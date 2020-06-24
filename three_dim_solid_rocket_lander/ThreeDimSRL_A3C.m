
%the purpose of this script is to implement an A3C algorithm for training
%the 2D rocket lander. I was irresponsible and deleted the PPO in favor of
%a deterministic one and I could not obtain convergence. I'm moving back to
%a stochastic approach in the form of A3C(Asynchronus advantage action-cirtic).
%It will become asynchronus when we enable parallel training of the
%network.

close all
clear all



%% creating the environment

%this is the environment that I want.
env = ThreeDimSolidRocketLander;
actionInfo = getActionInfo(env);
observationInfo = getObservationInfo(env);
numObs = observationInfo.Dimension(1);
%numAct = numel(actionInfo.Elements);


%% editing the lander parameters
%this is the pysical parameters of the class and the environment, we'll play
%around with these after training in order to see how the agent reacts.

env.Mass=2.0; %mass [kg]
env.L1=10; %center of gravity top/bottom [m]
env.L2=5;% center of gravity left/right [m]
env.L3=5; %center of gravity for inwards/outwards
env.Gravity=9.806; %gravitational acceleration [m/s^2]
env.Thrust=20; %thrust of our solid rocket [N]
env.Ts=0.05; %sample time [s] the amount that we step by
%env.State=[10;10;10;1;1;1]; %state vector [m;m;m;m/s;m/s;m/s]
%env.LastAction=[0;0];
env.TimeCount=0; %time since event start [s]

validateEnvironment(env)

%% create the RL agent: 1. critic network
% so this type of RL agent uses a value representation and not a Q-value.
% It will take the state only to estimate future rewards.

layer_size=[400 600];
criticNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC1')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC12')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC13')
    leakyReluLayer(0.1,'Name','leaky1')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC21')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC22')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC23')
    leakyReluLayer(0.1,'Name','leaky2')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC31')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC32')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC33')
    leakyReluLayer(0.1,'Name','leaky3')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC41')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC42')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC43')
    leakyReluLayer(0.1,'Name','leaky4')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC51')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC52')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC53')
    leakyReluLayer(0.1,'Name','leaky5')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC61')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC62')
    fullyConnectedLayer(layer_size(2),'Name','CriticFC63')
    leakyReluLayer(0.1,'Name','leaky6')
    fullyConnectedLayer(layer_size(1),'Name','CriticFC71')
    fullyConnectedLayer(layer_size(1),'Name','CriticFC72')
    fullyConnectedLayer(layer_size(1),'Name','CriticFC73')
    leakyReluLayer(0.1,'Name','leaky7')
    fullyConnectedLayer(1,'Name','CriticFC8')];

criticOpts = rlRepresentationOptions('LearnRate',0.1e-3,'UseDevice','gpu');

% create the critic
critic = rlValueRepresentation(criticNetwork,observationInfo,'Observation',{'state'},criticOpts);


%% create the RL agent: 2. actor network
%for this our actor is stochastic. This means it will determine a mean and
%standard deviation for each action possible. We will therefore have
%splitting branches for our

% observation path layers (6 by 1 input and a 2 by 1 output)
inPath = [ imageInputLayer([numObs 1 1], 'Normalization','none','Name','myobs')
    fullyConnectedLayer(layer_size(2),'Name','actorFC1')
    fullyConnectedLayer(layer_size(2),'Name','actorFC12')
    fullyConnectedLayer(layer_size(2),'Name','actorFC13')
    tanhLayer('Name','leaky1')
    fullyConnectedLayer(layer_size(2),'Name','actorFC2')
    fullyConnectedLayer(layer_size(2),'Name','actorFC21')
    fullyConnectedLayer(layer_size(2),'Name','actorFC23')
    tanhLayer('Name','leaky2')
    fullyConnectedLayer(layer_size(2),'Name','actorFC3')
    fullyConnectedLayer(layer_size(2),'Name','actorFC32')
    fullyConnectedLayer(layer_size(2),'Name','actorFC33')
    tanhLayer('Name','leaky3')
    fullyConnectedLayer(layer_size(2),'Name','actorFC5')
    fullyConnectedLayer(layer_size(2),'Name','actorFC52')
    fullyConnectedLayer(layer_size(2),'Name','actorFC53')
    tanhLayer('Name','leaky4')
    fullyConnectedLayer(layer_size(2),'Name','actorFC4')
    fullyConnectedLayer(layer_size(2),'Name','actorFC42')
    fullyConnectedLayer(layer_size(2),'Name','actorFC43')
    tanhLayer('Name','leaky5')
    fullyConnectedLayer(layer_size(2),'Name','actorFC8')
    fullyConnectedLayer(layer_size(2),'Name','actorFC82')
    fullyConnectedLayer(layer_size(2),'Name','actorFC83')
    tanhLayer('Name','leaky6')
    fullyConnectedLayer(layer_size(1),'Name','actorFC6')
    fullyConnectedLayer(layer_size(1),'Name','actorFC62')
    fullyConnectedLayer(layer_size(1),'Name','actorFC63')
    tanhLayer('Name','leaky7')
    fullyConnectedLayer(5,'Name','infc')];

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

delete(gcp)
pool = parpool(8);


agentOptions = rlPPOAgentOptions(...
    'AdvantageEstimateMethod', 'GAE', ...
    'ClipFactor', 0.1, 'MiniBatchSize',128,...
    'NumEpoch',18,'DiscountFactor',0.9995,...
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
trainOpts.ParallelizationOptions.StepsUntilDataIsSent=1000;
trainOpts.ParallelizationOptions.Mode='async';
trainOpts.ParallelizationOptions.DataToSendFromWorkers='Experiences';


trainingStats = train(agent,env,trainOpts);

%% simulate and plot the results
run_sim_and_plot(agent,env)
