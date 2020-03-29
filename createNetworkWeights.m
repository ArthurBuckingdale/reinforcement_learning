% Randomly initialize weights
weights.criticFC1 = sqrt(2/numObs)*(rand(criticLayerSizes(1),numObs)-0.5);
weights.criticFC2 = sqrt(2/criticLayerSizes(1))*(rand(criticLayerSizes(2),criticLayerSizes(1))-0.5);
weights.criticOut = sqrt(2/criticLayerSizes(2))*(rand(1,criticLayerSizes(2))-0.5);

bias.criticFC1 = 1e-3*ones(criticLayerSizes(1),1);
bias.criticFC2 = 1e-3*ones(criticLayerSizes(2),1);
bias.criticOut = 1e-3;

weights.actorFC1 = sqrt(2/numObs)*(rand(actorLayerSizes(1),numObs)-0.5);
weights.actorFC2 = sqrt(2/actorLayerSizes(1))*(rand(actorLayerSizes(2),actorLayerSizes(1))-0.5);
weights.actorOut = sqrt(2/actorLayerSizes(2))*(rand(numAct,actorLayerSizes(2))-0.5);

bias.actorFC1 = 1e-3*ones(actorLayerSizes(1),1);
bias.actorFC2 = 1e-3*ones(actorLayerSizes(2),1);
bias.actorOut = 1e-3*ones(numAct,1);