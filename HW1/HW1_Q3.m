%% HW1 Question 3 - FFR135
% Author: Jonas Hejderup
clear all; clc
%% Parameters
N = 200;
p = [7,45];
trials = 100;
T = 2*10^5;
beta = 2;
% Initialize the order vectors
order_exp = zeros(1,T);
order_avg = zeros(1,trials);
order_ensemble = zeros(1,length(p));
% Function for stochastic update
g = @(b,beta) 1/(1+exp(-2*beta*b));
%% Main loop
for i = 1:length(p) %loop through p
    for t = 1:trials
        % generate the random pattern
         rand_pattern = round(randi([1,2],N,p(i))-1.5);%Change to somethig else
         % Define the weight matrix
         W = zeros(N);
         for k = 1:size(rand_pattern,2)
            W = W + rand_pattern(:,k)*rand_pattern(:,k)';
         end
         W = W.*(1/N);
         W = W - diag(diag(W));
         % Feed the first pattern
         state = rand_pattern(:,1);
         % Do all the async update
         for j = 1:T 
            % Pick random neuron
            rand_N = randi([1,N]); 
            % Do the stochastic Hopfield update
            z = W(rand_N,:)*state;
            S = g(z,beta);
            r = rand(1);
            if S > r
                S = 1;
            else
                S = -1;
            end
            state(rand_N) = S;
            % Calculate the order parameter
            order_exp(j) = (rand_pattern(:,1)'*state)*(1/N);
         end
         % Order for each experiment
         order_avg(t) = mean(order_exp);
    end
    % The ensemble average order para
    order_ensemble(i) = mean(order_avg); 
end

disp('The ensemble averrage of the order parameter for p=7 and p=45:')
order_ensemble
    
    


