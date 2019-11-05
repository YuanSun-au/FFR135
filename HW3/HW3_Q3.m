%% HW3 - Question 3
% Author: Jonas Hejderup
clear all; clc;
%% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);
%% Define the layers for all of the networks
layers_1 = [...
    imageInputLayer([32 32 3])
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

layers_2 = [...
    imageInputLayer([32 32 3])
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

layers_3 = [...
    imageInputLayer([32 32 3])
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Define the training options for the networks
options_1 = trainingOptions('sgdm',...
    'MaxEpochs',400,...
    'MiniBatchSize',8192, ...
    'InitialLearnRate',0.001, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',{xValid,tValid});
options_2 = trainingOptions('sgdm', ...
    'MaxEpochs',400,...
    'MiniBatchSize',8192, ...
    'InitialLearnRate',0.003, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',{xValid,tValid});
options_3 = trainingOptions('sgdm', ...
    'MaxEpochs',400,...
    'MiniBatchSize',8192, ...
    'InitialLearnRate',1e-3, ...
    'ValidationFrequency',30, ...
    'L2Regularization',0.2,...
    'ValidationPatience',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',{xValid,tValid});

%% Training the networks
net_1 = trainNetwork(xTrain,tTrain,layers_1,options_1);
net_2 = trainNetwork(xTrain,tTrain,layers_2,options_2);
net_3 = trainNetwork(xTrain,tTrain,layers_3,options_3);

%% Calculate classification error for training, test and validation set

YTrain_1 = net_1.classify(xTrain);
YValid_1 = net_1.classify(xValid);
YTest_1 = net_1.classify(xTest);

Training_accuracy_1 = 1-sum(YTrain_1 == tTrain)/length(tTrain)
Validation_accuracy_1 = 1-sum(YValid_1 == tValid)/length(tValid)
Test_accuracy_1 = 1-sum(YTest_1 == tTest)/length(tTest)


YTrain_2 = net_2.classify(xTrain);
YValid_2 = net_2.classify(xValid);
YTest_2 = net_2.classify(xTest);

Training_accuracy_2 = 1-sum(YTrain_2 == tTrain)/length(tTrain)
Validation_accuracy_2 = 1-sum(YValid_2 == tValid)/length(tValid)
Test_accuracy_2 = 1-sum(YTest_2 == tTest)/length(tTest)

YTrain_3 = net_3.classify(xTrain);
YValid_3 = net_3.classify(xValid);
YTest_3 = net_3.classify(xTest);

Training_accuracy_3 = 1-sum(YTrain_3 == tTrain)/length(tTrain)
Validation_accuracy_3 = 1-sum(YValid_3 == tValid)/length(tValid)
Test_accuracy_3 = 1-sum(YTest_3 == tTest)/length(tTest)