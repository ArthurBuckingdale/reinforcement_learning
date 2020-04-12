%the purpose of this script is to modify the rocket landing example which
%has been provided by MATLAB. In their example, they have a pair of rockets
%which have vairable thrust to land a body on the ground from some height.
%The goal for me is to change the dynamics of this. I am going to assume
%that we have a solid rocket. This implies that we cannot change the thrust
%we produce. What istead will change is the angle the thrust is relative to
%the ground. Liquid fuel rockets are complicated and require lots of
%engineering. Solid rockets however are much easier to design and are
%cheap. Eventually, we will also play around with the moments of inertia
%and the centers of gravity to see the effect that poor engineering has on
%out ability to land a rocket. We can also have our solid rockets run out
%of fuel after a certain amount of time has passed to ensure that it's not
%performing silly landing manoeuvers(ideally we don't want it to sit on the
%ground with its rockets pointing 90 degrees from vertical if the package
%does not need all the fuel(this corresponds to using its fuel higher up and
%bringing the object in slower so it can consume all the fuel.

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


%% the non trivial part. Modifying the RocketLander class
%up till now, we've just copied code from the MATLAB example and added some
%additional commentary for context and hopefully better understanding. We
%now however need to make the less trivial changes to this RocketLander
%class. I've created a SolidRocketLander class which is a copy of the
%MATLAB one, but we are going to change the parameters that we require to
%have constant thrust output from the boosters, but change the angle
%relative to the ground where they're producing thrust in order to see what
%happens. Taking a look at the class, we can see the properties. This is
%where we will make edits to the non RL parameters.
%also cool sidebar it looks like this thing is actually computing integrals
%from the force parameters. there is no cheating here and we're actually
%evaluating the dynamics of the body. This is cool as all hell.

%the physical parameters of the situation:
%these can be seen by the user as designed in the class. The properties
%section of the class defines what the user can see.

env.Mass=1; %mass [kg]
env.L1=10; %center of gravity top/bottom [m]
env.L2=5;% center of gravity left/right [m]
env.Gravity=9.806; %gravitational acceleration [m/s^2]
%env.Thrust=10; %thrust of our solid rocket [N]
env.Ts=0.10; %sample time [s] the amount that we step by
env.State=[10;10;10;1;1;1]; %state vector [m;m;m;m/s;m/s;m/s]
env.LastAction=[0;0;0];
env.TimeCount=0; %time since event start [s]

validateEnvironment(env)

%% create the RL agent: 1. critic network
L=100;
K=200;
observationPath = [
    imageInputLayer([7 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(K,'Name','CriticObsFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(K,'Name','CriticObsFC2')
     reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(K,'Name','CriticObsFC3')
     reluLayer('Name','CriticRelu3')
    fullyConnectedLayer(L,'Name','CriticObsFC4')];
actionPath = [
    imageInputLayer([3 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(K,'Name','CriticActFC1')
    reluLayer('Name','CriticRelu5')
    fullyConnectedLayer(K,'Name','CriticObsFC0')
    reluLayer('Name','CriticRelu6')
    fullyConnectedLayer(K,'Name','CriticObsFC9')
    reluLayer('Name','CriticRelu7')
    fullyConnectedLayer(L,'Name','CriticObsFC8')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','output')];
criticNetwork = layerGraph(observationPath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticObsFC4','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticObsFC8','add/in2');

figure 
plot(criticNetwork)

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',100,'L2RegularizationFactor',1e-4);
critic1 = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
critic2 = rlQValueRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'action'},criticOptions);
%% create the RL agent: 2. actor network
actorNetwork = [
    imageInputLayer([7 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(K,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(K,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(K,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(K,'Name','fc4')
    reluLayer('Name','relu8')
    fullyConnectedLayer(L,'Name','fc5')
    reluLayer('Name','relu7')
    fullyConnectedLayer(3,'Name','fc6')
    tanhLayer('Name','tanh1')
    ];

actorOptions = rlRepresentationOptions('LearnRate',0.5e-3,'GradientThreshold',100,'L2RegularizationFactor',1e-4);
actor = rlDeterministicActorRepresentation(actorNetwork,observationInfo,actionInfo,...
    'Observation',{'observation'},'Action',{'tanh1'},actorOptions);
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

%pool = parpool(8);


agentOptions = rlTD3AgentOptions(...
    'SampleTime',env.Ts,...
    'TargetSmoothFactor',1e-4,...
    'ExperienceBufferLength',10000,...
    'DiscountFactor',0.9995,...
    'MiniBatchSize',64,...
    'NumStepsToLookAhead',2000);
% agentOptions.NoiseOptions.Variance = 0.6;
% agentOptions.NoiseOptions.VarianceDecayRate = 1e-5;
agent = rlTD3Agent(actor,[critic1 critic2],agentOptions);
%% configure training options and train.
%we now just need to configure the options for the training of the agent.
%These are for training specifically.



trainOpts = rlTrainingOptions(...
    'MaxEpisodes',20000,...
    'MaxStepsPerEpisode',2000,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10000,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',11000, ...
    'UseParallel',false);
%trainOpts.ParallelizationOptions.StepsUntilDataIsSent=200;
%trainOpts.ParallelizationOptions.Mode='async';

trainingStats = train(agent,env,trainOpts);





















