%% HW2 - Q2
% Author: Jonas Hejderup
clear all; clc;
%% Load training and target data data
x_train = csvread("input_data_numeric.csv");
x_train = x_train(:,2:end);

E = [1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1];
B = [1, -1, 1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, -1];
A = [1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1];
D = [-1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1];
C = [-1, -1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1];
F = [1, -1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1];

targets = [E' B' A' D' C' F'];
%% Parameters
eta = 0.02;
max_iteration = 10^5;
max_tries = 10;
final_output = zeros(size(targets));
linearly_separable = zeros(size(targets,2),1);
%% Main loop
for i = 1:size(targets,2)
    target = targets(:,i);
    
    for k = 1:max_tries
        % Intialize weights
        W = -0.02 + rand(4,1)*(0.02-(-0.02));
        bias = -1 + rand*(1-(-1));
        for j = 1:max_iteration
            % Forward prop
            b = sum(-bias + x_train*W);
            output =  tanh(0.5*(-bias + x_train*W));
            % Stochastic grad
            r = randi([1,size(targets,1)]);
            delta_w = eta*(target(r)-output(r))*(1-tanh(b*0.5).^2)*0.5* x_train(r,:)';
            delta_bias = -eta*(target(r)-output(r))*(1-tanh(b*0.5).^2)*0.5;
            W = W + delta_w;
            bias = bias + delta_bias;
            % Check if lin. sep. break early
            if sign(tanh(0.5*(-bias + x_train*W))) == target
                break
            end
        end
        
        % Check if lin. sep. break early
        if sign(tanh(0.5*(-bias + x_train*W))) == target
            break
        end  
    end
    
    final_output(:,i) = sign(tanh(0.5*(-bias + x_train*W)));
    if isequal(final_output(:,i),target)
        linearly_separable(i) = true;
    else
        linearly_separable(i) = false;
    end   
end

linearly_separable