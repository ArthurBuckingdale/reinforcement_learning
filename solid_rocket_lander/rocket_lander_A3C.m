
%the purpose of this script is to implement an A3C algorithm for training
%the 2D rocket lander. I was irresponsible and deleted the PPO in favor of
%a deterministic one and I could not obtain convergence. I'm moving back to
%a stochastic approach in the form of A3C(Asynchronus advantage action-cirtic).
%It will become asynchronus when we enable parallel training of the
%network.

close all
clear all



%% creating the environment
%I've imported the RocketLander class from the PPO example in MATLAB we can
%go ahead and modify the properties of this like the center of gravity and
%all that. This also tells the wrapper functions what it needs to do. This is
%the thing that I will need to edit in order to have a different rocket
%type. I will continue setting up this environment and script for now though.

% %this is the environment that i'm using to DEBUG
% env2 = RocketLander;
% actionInfo2 = getActionInfo(env2);
% observationInfo2 = getObservationInfo(env2);
% numObs2 = observationInfo2.Dimension(1);
% %numAct2 = numel(actionInfo2.Elements);

%this is the environment that I want.
env = SolidRocketLander;
actionInfo = getActionInfo(env);
observationInfo = getObservationInfo(env);
numObs = observationInfo.Dimension(1);
%numAct = numel(actionInfo.Elements);


%% editing the lander parameters
%this is the pysical parameters of the class and the environment, we'll play
%around with these after training in order to see how the agent reacts.

env.Mass=1; %mass [kg]
env.L1=10; %center of gravity top/bottom [m]
env.L2=5;% center of gravity left/right [m]
env.Gravity=9.806; %gravitational acceleration [m/s^2]
%env.Thrust=10; %thrust of our solid rocket [N]
env.Ts=0.05; %sample time [s] the amount that we step by
%env.State=[10;10;10;1;1;1]; %state vector [m;m;m;m/s;m/s;m/s]
env.LastAction=[0;0];
env.TimeCount=0; %time since event start [s]

validateEnvironment(env)

%% create the RL agent: 1. critic network
% so this type of RL agent uses a value representation and not a Q-value.
% It will take the state only to estimate future rewards.

layer_size=[200 300];
criticNetwork = [
    imageInputLayer([7 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC')
    leakyReluLayer('Name','leaky1')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC2')
    leakyReluLayer('Name','leaky2')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC3')
    leakyReluLayer('Name','leaky3')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC4')
    leakyReluLayer('Name','leaky4')
    fullyConnectedLayer(layer_size(1),'Name','CriticFC5')
    leakyReluLayer('Name','leaky5')
    fullyConnectedLayer(1,'Name','CriticFC6')];
% action_input = [
%     imageInputLayer([2 1 1],'Normalization','none','Name','action')
%     fullyConnectedLayer(layer_size(2),'Name','CriticAFC')
%     leakyReluLayer('Name','leaky11')
%     fullyConnectedLayer(layer_size(1),'Name','CriticAFC2')];
% 
% comPath = [additionLayer(2,'Name','add') fullyConnectedLayer(1,'Name','output')];
% 
% net = addLayers(layerGraph(criticNetwork),action_input);
% net = addLayers(net,comPath);
% 
% net = connectLayers(net,'CriticFC6','add/in1');
% net = connectLayers(net,'CriticAFC2','add/in2');
% 
% figure
% plot(net)
% 
% set some options for the critic
criticOpts = rlRepresentationOptions('LearnRate',0.05e-3,'UseDevice','gpu');

% create the critic
critic = rlValueRepresentation(criticNetwork,observationInfo,'Observation',{'state'},criticOpts);


%% create the RL agent: 2. actor network
%for this our actor is stochastic. This means it will determine a mean and
%standard deviation for each action possible. We will therefore have
%splitting branches for our

% observation path layers (6 by 1 input and a 2 by 1 output)
inPath = [ imageInputLayer([7 1 1], 'Normalization','none','Name','myobs')
    fullyConnectedLayer(layer_size(2),'Name','actorFC1')
    leakyReluLayer('Name','leaky1')
    fullyConnectedLayer(layer_size(2),'Name','actorFC2')
    leakyReluLayer('Name','leaky2')
    fullyConnectedLayer(layer_size(2),'Name','actorFC3')
    leakyReluLayer('Name','leaky3')
    fullyConnectedLayer(layer_size(2),'Name','actorFC5')
    leakyReluLayer('Name','leaky4')
    fullyConnectedLayer(layer_size(1),'Name','actorFC4')
    leakyReluLayer('Name','leaky5')
    fullyConnectedLayer(2,'Name','infc')];

% path layers for mean value (2 by 1 input and 2 by 1 output)
% using scalingLayer to scale the range
meanPath = [ tanhLayer('Name','tanh');
    scalingLayer('Name','scale','Scale',actionInfo.UpperLimit) ];

% path layers for variance (2 by 1 input and output)
% using softplus layer to make it non negative)
variancePath =  softplusLayer('Name', 'splus');

% conctatenate two inputs (along dimension #3) to form a single (4 by 1) output layer
outLayer = concatenationLayer(3,2,'Name','gaussPars');

% add layers to network object
net = layerGraph(inPath);
net = addLayers(net,meanPath);
net = addLayers(net,variancePath);
net = addLayers(net,outLayer);

% connect layers
net = connectLayers(net,'infc','tanh/in');              % connect output of inPath to meanPath input
net = connectLayers(net,'infc','splus/in');             % connect output of inPath to variancePath input
net = connectLayers(net,'scale','gaussPars/in1');       % connect output of meanPath to gaussPars input #1
net = connectLayers(net,'splus','gaussPars/in2');       % connect output of variancePath to gaussPars input #2

plot(net)

actorOpts = rlRepresentationOptions('LearnRate',0.05e-3,'UseDevice','gpu');
actor = rlStochasticActorRepresentation(net,observationInfo,actionInfo,...
    'Observation',{'myobs'},actorOpts);

%% create the RL agent: 3. agent options.
%we now need to create the agent options itself. If i'm undestanding this
%correctly, what the agent can do is defined in the environement. We need
%to change the environment in order to edit this. This current sript is
%just essentially the meta parameters and which agent we want in the
%network, plus architecture of the network.

%so these are the options for the RL agent. If i recall correctly, this is
%what the parameters do:
%       ExperienceHorizon: keeps track of the previous steps so it
%           understands how the reward changes over time and can work out
%           which actions are the best to take.
%       ClipFactor: I have no idea...
%       EntropyLossWeight: no idea
%       .....stuff that's trivial DL things
%       AdvantageEstimateMethod: i need to watch the video on what this is
%       DiscountFactor: this is what ecourages long term reward if it's
%       lower we get less reward from previous steps, i need to watch this
%       video again as well it seems.

delete(gcp)
pool = parpool(8);


agentOptions = rlPPOAgentOptions(...
    'AdvantageEstimateMethod', 'GAE', ...
    'ClipFactor', 0.1, 'MiniBatchSize',256,...
    'NumEpoch',20,'DiscountFactor',0.9995,...
    'EntropyLossWeight',0.02,...
    'SampleTime',env.Ts);
agent = rlPPOAgent(actor,critic,agentOptions);

%% configure training options and train.
%we now just need to configure the options for the training of the agent.
%These are for training specifically.



trainOpts = rlTrainingOptions(...
    'MaxEpisodes',30000,...
    'MaxStepsPerEpisode',2000,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10000,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',11000, ...
    'UseParallel',true);
trainOpts.ParallelizationOptions.StepsUntilDataIsSent=1999;
trainOpts.ParallelizationOptions.Mode='async';
trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';


trainingStats = train(agent,env,trainOpts);
%% second instance of training for longer duration rewards
%this is some testing for me. I'm trying to figure out the steps util data
%sent and imporving it all

% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',8000,...
%     'MaxStepsPerEpisode',3000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageReward',...
%     'StopTrainingValue',10000,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=2999;
% trainOpts.ParallelizationOptions.Mode='async';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
% 
% 
% trainingStats2 = train(agent,env,trainOpts);
%
% %% third instance extremely long term training
% %we now must push on to long term training and update steps until data sent
% %again
%
% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',20000,...
%     'MaxStepsPerEpisode',1000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageReward',...
%     'StopTrainingValue',10000,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=999;
% trainOpts.ParallelizationOptions.Mode='sync';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
%
%
% trainingStats3 = train(agent,env,trainOpts);
%

% %% more deep training
%
% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',5000,...
%     'MaxStepsPerEpisode',3000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageSteps',...
%     'StopTrainingValue',1600,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=1600;
% trainOpts.ParallelizationOptions.Mode='sync';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
%
%
% trainingStats4 = train(agent,env,trainOpts);
%
% %
% %
% %% longer training here
% %so, we are iteratively increasing the maximum average steps since the RL
% %agent has different things to learn at the beginning than the end. We can
% %shape this with varying the stepsUntilDataIsSent parameter.
%
% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',4000,...
%     'MaxStepsPerEpisode',3000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageSteps',...
%     'StopTrainingValue',2000,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=2000;
% trainOpts.ParallelizationOptions.Mode='sync';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
%
%
% trainingStats5 = train(agent,env,trainOpts);
%
%
%
% %% synchronus.
% %by this point we should have a well converging solution and we can average
% %the results from each worker to push to the 10000 benchmark.
%
%
% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',5000,...
%     'MaxStepsPerEpisode',3000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageSteps',...
%     'StopTrainingValue',2800,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=2999;
% trainOpts.ParallelizationOptions.Mode='sync';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
%
%
% trainingStats6 = train(agent,env,trainOpts);
%
% %% longer episodes still
%
%
% trainOpts = rlTrainingOptions(...
%     'MaxEpisodes',5000,...
%     'MaxStepsPerEpisode',5000,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','AverageReward',...
%     'StopTrainingValue',10000,...
%     'ScoreAveragingWindowLength',100,...
%     'SaveAgentCriteria',"EpisodeReward",...
%     'SaveAgentValue',11000, ...
%     'UseParallel',true);
% trainOpts.ParallelizationOptions.StepsUntilDataIsSent=4000;
% trainOpts.ParallelizationOptions.Mode='sync';
% trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';
%
%
% trainingStats7 = train(agent,env,trainOpts);




