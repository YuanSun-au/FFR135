%% HW1 Question 1 - FFR135
% Author: Jonas Hejderup
clear all; clc
%% Parameters
N = 120;
p = [12,24,48,70,100,120];
trials = 10^5;
W_diag_zero  = false; % Set false for Q1b
%% Main loop for calcuating the one-step error
error = zeros(1,length(p));
parfor i = 1:length(p) % Loop through all the p
    
    for j = 1:trials % Do all the trials
        %Generate all the patterns with +1 or -1
        rand_pattern = round(randi([1,2],N,p(i))-1.5);
        
        % Calculate the weight matrix according to the lecture notes
        W = zeros(N);
        for k = 1:size(rand_pattern,2)
            W = W + rand_pattern(:,k)*rand_pattern(:,k)';
        end
        W = W.*(1/N);
        % Set the diagonal to zero for Q1a
        if W_diag_zero
            W = W - diag(diag(W));
        end
        
        % Select random pattern and neuron
        rand_p = randi([1,p(i)]);
        rand_N = randi([1,N]);
        
        % Do the async update
        z = W(rand_N,:)*rand_pattern(:,rand_p);
        S = sign(z);
        if S == 0
            S = 1;
        end
        
        % Check for error
        if S ~= rand_pattern(rand_N,rand_p)
            error(i) = error(i) + 1;
        end     
    end
end
%% Calculate percentage and printing results
error = error./trials;
if W_diag_zero
    disp('The one step error for Wii = 0 is the following: ')
    error
else
    disp('The one step error when diagonal is not zero is the following: ')
    error
end