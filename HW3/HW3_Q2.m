%% HW3 - Question 2
% Author: Jonas Hejderup
clear all; clc; close all;
%% Load the data for the question
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(2);
% imshow(reshape(xTrain(:,25081),[32 32 3]));
% tTrain(:,25081)
%% Normalization
norm_x = sum(xTrain,2)*(1/size(xTrain,2));
xTrain_norm = xTrain - norm_x;
xValid_norm = xValid - norm_x;
xTest_norm = xTest - norm_x;
%% Parameters
eta = 0.01;
training_size = size(xTrain,2);
input_size = size(xTrain,1);
output_size = 10;
epochs = 100;
batch_size = 100;
g_prime = @(x) exp(-x)./((1+exp(-x)).^2);
g = @(x) 1./(1+exp(-x));
%% Define weigths for all layers
M1 = 20;
W1 = normrnd(0, 1/sqrt(3072), [M1, input_size]);
W2 = normrnd(0, 1/sqrt(M1), [M1, M1]);
W3 = normrnd(0, 1/sqrt(M1), [M1, M1]);
W4 = normrnd(0, 1/sqrt(M1), [M1, M1]);
W5 = normrnd(0, 1/sqrt(M1), [output_size, M1]);
bias_1 = zeros(M1, 1);
bias_2 = zeros(M1, 1);
bias_3 = zeros(M1, 1);
bias_4 = zeros(M1, 1);
bias_5 = zeros(output_size, 1);
% Intialize vectors for plotting
Cost = zeros(1,epochs);
u1 = zeros(epochs,1);
u2 = zeros(epochs,1);
u3 = zeros(epochs,1);
u4 = zeros(epochs,1);
u5 = zeros(epochs,1);
for i = 1:epochs
    % Shuffle for the mini batch
    shuffle = randperm(training_size);
    xTrain_shuffle = xTrain_norm(:,shuffle);
    tTrain_shuffle = tTrain(:,shuffle);
   % Vecorized implementation of mini batch
    for j = 1:batch_size:training_size
        % Select the inputs and targets according to batch size
        xTrain_temp = xTrain_shuffle(:,j:j+batch_size-1);
        tTrain_temp = tTrain_shuffle(:,j:j+batch_size-1);
        % Feed forward
        b1 = W1*xTrain_temp - bias_1;
        V1 = g(b1);
        b2 = W2*V1 - bias_2;
        V2 = g(b2);
        b3 = W3*V2 - bias_3;
        V3 = g(b3);
        b4 = W4*V3 - bias_4;
        V4 = g(b4);
        b5 = W5*V4 - bias_5;
        O = g(b5);
        % Back-prop
        delta_5 = (tTrain_temp-O).*g_prime(b5);
        delta_4 = (W5'*delta_5).*g_prime(b4);
        delta_3 = (W4'*delta_4).*g_prime(b3);
        delta_2 = (W3'*delta_3).*g_prime(b2);
        delta_1 = (W2'*delta_2).*g_prime(b1);
        % Update weights and biases
        W5 = W5 + eta*delta_5*V4';
        W4 = W4 + eta*delta_4*V3';
        W3 = W3 + eta*delta_3*V2';
        W2 = W2 + eta*(delta_2*V1');
        W1 = W1 + eta*(delta_1*xTrain_temp');
        bias_1 = bias_1 - eta*sum(delta_1,2);
        bias_2 = bias_2 - eta*sum(delta_2,2);
        bias_3 = bias_3 - eta*sum(delta_3,2);
        bias_4 = bias_4 - eta*sum(delta_4,2);
        bias_5 = bias_5 - eta*sum(delta_5,2);
    end
    % Feed forward for entire trainig set
    b1 = W1*xTrain_shuffle - bias_1;
    V1 = g(b1);
    b2 = W2*V1 - bias_2;
    V2 = g(b2);
    b3 = W3*V2 - bias_3;
    V3 = g(b3);
    b4 = W4*V3 - bias_4;
    V4 = g(b4);
    b5 = W5*V4 - bias_5;
    O = g(b5);
    % Find the cost
    Cost(i) = (1/2)*sum(sum((tTrain_shuffle-O).^2,2));
    % Calculate the gradients for the biases
    delta_5 = (tTrain_shuffle-O).*g_prime(b5);
    delta_4 = (W5'*delta_5).*g_prime(b4);
    delta_3 = (W4'*delta_4).*g_prime(b3);
    delta_2 = (W3'*delta_3).*g_prime(b2);
    delta_1 = (W2'*delta_2).*g_prime(b1);
    
    u1(i) = norm(sum(delta_1,2));
    u2(i) = norm(sum(delta_2,2));
    u3(i) = norm(sum(delta_3,2));
    u4(i) = norm(sum(delta_4,2));
    u5(i) = norm(sum(delta_5,2));
end
%% Plotting the results
figure(1)
subplot(1,2,1)
plot(1:epochs,Cost)
title('Energy function')
xlabel('epochs')
ylabel('loss')
subplot(1,2,2)
semilogy(1:epochs,u1,1:epochs,u2,1:epochs,u3,1:epochs,u4,1:epochs,u5)
legend('Layer 1','Layer 2','Layer 3','Layer 4','Layer 5','Location','southeast')
xlabel('epochs')
ylabel('gradient')
title('Gradient for biases')