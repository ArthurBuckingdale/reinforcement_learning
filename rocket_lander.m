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

% env.Mass=1; %mass [kg]
% env.L1=10; %center of gravity top/bottom [m]
% env.L2=5;% center of gravity left/right [m]
% env.Gravity=9.806; %gravitational acceleration [m/s^2]
% env.Thrust=10; %thrust of our solid rocket [N]
% env.Ts=0.10; %sample time [s] the amount that we step by
% env.State=[10;10;10;1;1;1]; %state vector [m;m;m;m/s;m/s;m/s]
% env.LastAction=[0;0];
% env.TimeCount=0; %time since event start [s]

%% create the RL agent: 1. critic network
%MATLAB's example uses a PPO. If I understand correctly, this works on
%discreet action space, this is not going to work for me, so i'm going to
%choose a different agent. I'll need to perouse through the different type
%of agents to find the best suited one for this or just try a bunch of
%different ones. For now let's assemble the nets that will be doing the
%"thinking".

%here we initalise the size and the network weights
criticLayerSizes = [200 100];
actorLayerSizes = [200 100];
%createNetworkWeights;

%this is defining the critic network architecture
criticNetwork = [imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(criticLayerSizes(1),'Name','CriticFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(criticLayerSizes(2),'Name','CriticFC2')
    reluLayer('Name','CriticRelu2')
    fullyConnectedLayer(1,'Name','CriticOutput')];

%this is calling the critic options and building the network.
criticOpts = rlRepresentationOptions('LearnRate',1e-3);
critic = rlValueRepresentation(criticNetwork,env.getObservationInfo, ...
    'Observation',{'observation'},criticOpts);


%% create the RL agent: 2. actor network
%so we've initialized the critic network, let's do the same for the actor,
%both will be needed,

% %defining the architecture of the actor network. 
% actorNetwork = [imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
%     fullyConnectedLayer(actorLayerSizes(1),'Name','ActorFC1')
%     reluLayer('Name','ActorRelu1')
%     fullyConnectedLayer(actorLayerSizes(2),'Name','ActorFC2')
%     reluLayer('Name','ActorRelu2')
%     fullyConnectedLayer(2 ,'Name','Action')];
% 
% %setting the options and initializing the actor network.
% actorOptions = rlRepresentationOptions('LearnRate',1e-3);
% actor = rlStochasticActorRepresentation(actorNetwork,env.getObservationInfo,env.getActionInfo,...
%     'Observation',{'observation'}, actorOptions);


% observation path layers (6 by 1 input and a 2 by 1 output)
inPath = [ imageInputLayer([7 1 1], 'Normalization','none','Name','myobs') 
           fullyConnectedLayer(200,'Name','infc')
           reluLayer('Name','relu1')
           fullyConnectedLayer(200,'name','infc3')
           reluLayer('Name','relu2')
           fullyConnectedLayer(200,'name','infc4')
           reluLayer('Name','relu3')
           fullyConnectedLayer(100,'name','infc5')
           reluLayer('Name','relu5')
           fullyConnectedLayer(2,'Name','infc2') ];

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
net = connectLayers(net,'infc2','tanh/in');              % connect output of inPath to meanPath input
net = connectLayers(net,'infc2','splus/in');             % connect output of inPath to variancePath input
net = connectLayers(net,'scale','gaussPars/in1');       % connect output of meanPath to gaussPars input #1
net = connectLayers(net,'splus','gaussPars/in2');% connect output of variancePath to gaussPars input #2
%net = connectLayers(net,'relu2','gaussPars/in3');

figure 
plot(net)

actorOptions = rlRepresentationOptions('LearnRate',1e-3);

actor = rlStochasticActorRepresentation(net,env.getObservationInfo,env.getActionInfo, 'Observation','myobs',actorOptions);
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

opt = rlPPOAgentOptions('ExperienceHorizon',1024,...
                        'ClipFactor',0.1,...
                        'EntropyLossWeight',0.02,...
                        'MiniBatchSize',128,...
                        'NumEpoch',10,...
                        'AdvantageEstimateMethod','gae',...
                        'GAEFactor',0.95,...
                        'SampleTime',env.Ts,...
                        'DiscountFactor',0.9995);
                    
agent = rlPPOAgent(actor,critic,opt);

%% configure training options and train.
%we now just need to configure the options for the training of the agent.
%These are for training specifically.

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',20000,...
    'MaxStepsPerEpisode',2500,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10000,...
    'ScoreAveragingWindowLength',100,...
    'SaveAgentCriteria',"EpisodeReward",...
    'SaveAgentValue',11000);

trainingStats = train(agent,env,trainOpts);





















