%% HW3 - Question 4
% Author: Jonas Hejderup
clear all; clc;
%% Load data
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(4);
%% Define the layers for all of the networks
layers_1 = [...
    imageInputLayer([32 32 3])
    convolution2dLayer(5,20,'Stride',2,'Padding',1)
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

layers_2 = [...
    imageInputLayer([32 32 3])
    convolution2dLayer(3,20,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,30,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2,'Padding',0)
    convolution2dLayer(3,50,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% Define the training options for the networks
options_1 = trainingOptions('sgdm', ...
    'MaxEpochs',120,...
    'MiniBatchSize',8192, ...
    'InitialLearnRate',0.001, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',{xValid,tValid});

options_2 = trainingOptions('sgdm', ...
    'MaxEpochs',120,...
    'MiniBatchSize',8192, ...
    'InitialLearnRate',0.001, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',3, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress',...
    'ValidationData',{xValid,tValid});

%% Training the networks
net_1 = trainNetwork(xTrain,tTrain,layers_1,options_1);
net_2 = trainNetwork(xTrain,tTrain,layers_2,options_2);



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