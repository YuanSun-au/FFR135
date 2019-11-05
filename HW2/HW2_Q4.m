%% HW2 - Q4
% Author: Jonas Hejderup
clear all; clc;
%% Load training and validation data
train_data = csvread("training_set.csv");
validation_data = csvread("validation_set.csv");

train_x = train_data(:,1:2);
train_y = train_data(:,3);

val_x = validation_data(:,1:2);
val_y = validation_data(:,3);

%% Parameters
M1 = 8;
M2 = 4;
eta = 0.02;
epochs = 1000;
error_criteria = 0.12;

%% Intialize weights and bias
% Intialize weight with Random Gaussian number
W1 = normrnd(0,0.7,[M1,2]);
W2 = normrnd(0,0.7,[M2,M1]);
W3 = normrnd(0,0.7,[1,M2]);
% Set the biases to zero
bias_1 = zeros(1,M1);
bias_2 = zeros(1,M2);
bias_3 = 0;

%% Main loop
for i = 1:epochs
    % Stochastic gradient descent algorithm
    for j = 1:length(train_x)
        % Random training example
        r = randi([1,length(train_y)]);
        % Propagate forward
        V1 = tanh(-bias_1' + W1*train_x(r,:)');
        V2 = tanh(-bias_2' + W2*V1);
        O = tanh(-bias_3 + W3*V2);
        % Backpropagate
        delta_3 = (train_y(r)-O)*(1-O.^2);
        delta_2 = delta_3 * W3.*(1-V2.^2)';
        delta_1 = delta_2*W2.*(1-V1.^2)';
        % Update the weights
        W3 = W3 + eta*delta_3*V2';
        W2 = W2 + eta*(V1*delta_2)';
        W1 = W1 + eta*delta_1'.*train_x(r,:);
        % Update the bias
        bias_1 = bias_1 - eta*delta_1;
        bias_2 = bias_2 - eta*delta_2;
        bias_3 = bias_3 - eta*delta_3;
    end
    % Feedforward for the validation data
    V1_val = tanh(-bias_1 + (W1*val_x')');
    V2_val = tanh(-bias_2 + (W2*V1_val')');
    O_val = tanh(-bias_3 + (W3*V2_val')');
    % Compute the classification error
    C = (1/(2*length(val_y)))*sum(abs(sign(O_val)-val_y));
    % Break if the error criteria is met
    if C < error_criteria
        disp(['Epoch: ',num2str(i),' Classification error: ',num2str(C)])
        disp('Criteria is met');
        break
    else
        disp(['Epoch: ',num2str(i),' Classification error: ',num2str(C)])
    end     
end

%% Output the results
csvwrite('w1.csv',W1);
csvwrite('w2.csv',W2);
csvwrite('w3.csv',W3);
csvwrite('t1.csv',bias_1);
csvwrite('t2.csv',bias_2);
csvwrite('t3.csv',bias_3);



