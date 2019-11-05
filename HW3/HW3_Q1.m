%% HW3 - Question 1
% Author: Jonas Hejderup
clear all; clc;
%% Load the data for the question
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(1);
% imshow(reshape(xTrain(:,25081),[32 32 3]));
% tTrain(:,25081)
%% Normalization
norm_x = sum(xTrain,2)*(1/size(xTrain,2));
xTrain_norm = xTrain - norm_x;
xValid_norm = xValid - norm_x;
xTest_norm = xTest - norm_x;

%% Parameters for all networks
eta = 0.1;
training_size = size(xTrain,2);
input_size = size(xTrain,1);
output_size = 10;
epochs = 20;
batch_size = 100;
g_prime = @(x) exp(-x)./((1+exp(-x)).^2);
g = @(x) 1./(1+exp(-x));

%% Network 1 - No hidden layer
% Parameters for Network 1 and intialize weights and biases
W1_1 = normrnd(0, 1/sqrt(3072), [output_size, input_size]);
bias_1_1 = zeros(1, output_size);
% Intialize vector to save class error
C_train_network_1 = zeros(1,epochs);
C_val_network_1 = zeros(1,epochs);
C_test_network_1 = zeros(1,epochs);
% Training Loop
 for i = 1:epochs
     % Shuffle for the mini batch
     shuffle = randperm(training_size);
     xTrain_shuffle = xTrain_norm(:,shuffle);
     tTrain_shuffle = tTrain(:,shuffle);
     % Vectorized implementation, lets hope everything works
     for j = 1:batch_size:training_size
         % Select the inputs and targets according to batch size
         xTrain_temp = xTrain_shuffle(:,j:j+batch_size-1);
         tTrain_temp = tTrain_shuffle(:,j:j+batch_size-1);
         % Feed-forward
         b1 = W1_1*xTrain_temp - bias_1_1';
         O = g(b1);
         % Back prop
         delta_1 = (tTrain_temp-O).*g_prime(b1);
         % Update the weights and biases
         W1_1 = W1_1 + eta*(delta_1*xTrain_temp');
         bias_1_1 = bias_1_1 - eta*sum(delta_1,2)';
     end
     % Calculate the training classification
     output_train_temp = g(W1_1*xTrain_norm - bias_1_1');
     output_train = zeros(output_size, length(output_train_temp));
     % Pick the class with the highest score
     for k = 1:length(output_train)
         [argvalue, argmax] = max(output_train_temp(:,k));
         output_train(argmax,k) = 1;
     end
     C_train_network_1(i) = (1/(2*length(output_train)))*sum(sum(abs(output_train-tTrain)));
     
     % Calculate the validation classification
     output_val_temp = g(W1_1*xValid_norm - bias_1_1');
     output_val = zeros(output_size, length(output_val_temp));
     % Pick the class with the highest score
     for k = 1:length(output_val)
         [argvalue, argmax] = max(output_val_temp(:,k));
         output_val(argmax,k) = 1;
     end
     C_val_network_1(i) = (1/(2*length(output_val)))*sum(sum(abs(output_val-tValid)));
     
     % Calculate the test classification
     output_test_temp = g(W1_1*xTest_norm - bias_1_1');
     output_test = zeros(output_size, length(output_test_temp));
     % Pick the class with the highest score
     for k = 1:length(output_test)
         [argvalue, argmax] = max(output_test_temp(:,k));
         output_test(argmax,k) = 1;
     end
     C_test_network_1(i) = (1/(2*length(output_test)))*sum(sum(abs(output_test-tTest)));
 end
[argvalue, argmin] = min(C_val_network_1);
disp('-------------------------------------')
disp('Network 1:')
disp('-------------------------------------')
disp("Lowest validation classification error: "+num2str(argvalue))
disp("Epoch: "+num2str(argmin))
disp("Training classification error: "+num2str(C_train_network_1(argmin)))
disp("Test classification error: "+num2str(C_test_network_1(argmin)))
disp('-------------------------------------')

%% Network 2 - One hidden layer - 10 neurons
% Parameters for Network 2 and intialize weights and biases
M1_2 = 10;
W1_2 = normrnd(0, 1/sqrt(3072), [M1_2, input_size]);
W2_2 = normrnd(0, 1/sqrt(10), [output_size, M1_2]);
bias_1_2 = zeros(1, M1_2);
bias_2_2 = zeros(1, output_size);
% Intialize vector to save class error
C_train_network_2 = zeros(1,epochs);
C_val_network_2 = zeros(1,epochs);
C_test_network_2 = zeros(1,epochs);
% Training loop
for i = 1:epochs
    % Shuffle for the mini batch
    shuffle = randperm(training_size);
    xTrain_shuffle = xTrain_norm(:,shuffle);
    tTrain_shuffle = tTrain(:,shuffle);
   % Vectorized implementation, lets hope everything works
    for j = 1:batch_size:training_size
        % Select the inputs and targets according to batch size
        xTrain_temp = xTrain_shuffle(:,j:j+batch_size-1);
        tTrain_temp = tTrain_shuffle(:,j:j+batch_size-1);
        % Feed-forward
        b1 = W1_2*xTrain_temp - bias_1_2';
        V1 = g(b1);
        b2 = W2_2*V1 - bias_2_2';
        O = g(b2);
        % Back prop
        delta_2 = (tTrain_temp-O).*g_prime(b2);
        delta_1 = (W2_2'*delta_2).*g_prime(b1);
        %Update weight and biases
        W2_2 = W2_2 + eta*delta_2*V1';
        W1_2 = W1_2 + eta*(delta_1*xTrain_temp');
        bias_1_2 = bias_1_2 - eta*sum(delta_1,2)';
        bias_2_2 = bias_2_2 - eta*sum(delta_2,2)';
    end
    % Calculate the training classification
    V1_train = g(W1_2*xTrain_norm - bias_1_2');
    output_train_temp = g(W2_2*V1_train - bias_2_2');
    output_train = zeros(output_size, length(output_train_temp));
    % Pick the class with the highest score
    for k = 1:length(output_train)
        [argvalue, argmax] = max(output_train_temp(:,k));
        output_train(argmax,k) = 1;
    end
    C_train_network_2(i) = (1/(2*length(output_train)))*sum(sum(abs(output_train-tTrain)));
    
    % Calculate the validation classification
    V1_val = g(W1_2*xValid_norm - bias_1_2');
    output_val_temp = g(W2_2*V1_val - bias_2_2');
    output_val = zeros(output_size, length(output_val_temp));
    % Pick the class with the highest score
    for k = 1:length(output_val)
        [argvalue, argmax] = max(output_val_temp(:,k));
        output_val(argmax,k) = 1;  
    end
    C_val_network_2(i) = (1/(2*length(output_val)))*sum(sum(abs(output_val-tValid))); 
    
    % Calculate the test classification
    V1_test = g(W1_2*xTest_norm - bias_1_2');
    output_test_temp = g(W2_2*V1_test - bias_2_2');
    output_test = zeros(output_size, length(output_test_temp));
    % Pick the class with the highest score
    for k = 1:length(output_test)
        [argvalue, argmax] = max(output_test_temp(:,k));
        output_test(argmax,k) = 1;  
    end
    C_test_network_2(i) = (1/(2*length(output_test)))*sum(sum(abs(output_test-tTest)));
end
[argvalue, argmin] = min(C_val_network_2);
disp('Network 2:')
disp('-------------------------------------')
disp("Lowest validation classification error: "+num2str(argvalue))
disp("Epoch: "+num2str(argmin))
disp("Training classification error: "+num2str(C_train_network_2(argmin)))
disp("Test classification error: "+num2str(C_test_network_2(argmin)))
disp('-------------------------------------')

%% Network 3 - One hidden layer - 50 neurons
% Parameters for Network 3 and intialize weights and biases
M1_3 = 50;
W1_3 = normrnd(0, 1/sqrt(3072), [M1_3, input_size]);
W2_3 = normrnd(0, 1/sqrt(50), [output_size, M1_3]);
bias_1_3 = zeros(1, M1_3);
bias_2_3 = zeros(1, output_size);
% Intialize vector to save class error
C_train_network_3 = zeros(1,epochs);
C_val_network_3 = zeros(1,epochs);
C_test_network_3 = zeros(1,epochs);
% Training loop
for i = 1:epochs
    % Shuffle for the mini batch
    shuffle = randperm(training_size);
    xTrain_shuffle = xTrain_norm(:,shuffle);
    tTrain_shuffle = tTrain(:,shuffle);
   % Vectorized implementation, lets hope everything works
    for j = 1:batch_size:training_size
        % Select the inputs and targets according to batch size
        xTrain_temp = xTrain_shuffle(:,j:j+batch_size-1);
        tTrain_temp = tTrain_shuffle(:,j:j+batch_size-1);
        % Feed forward
        b1 = W1_3*xTrain_temp - bias_1_3';
        V1 = g(b1);
        b2 = W2_3*V1 - bias_2_3';
        O = g(b2);
        % Backprop
        delta_2 = (tTrain_temp-O).*g_prime(b2);
        delta_1 = (W2_3'*delta_2).*g_prime(b1);
        % Update weights biases
        W2_3 = W2_3 + eta*delta_2*V1';
        W1_3 = W1_3 + eta*(delta_1*xTrain_temp');
        bias_1_3 = bias_1_3 - eta*sum(delta_1,2)';
        bias_2_3 = bias_2_3 - eta*sum(delta_2,2)';
    end
    % Calculate the training classification
    V1_train = g(W1_3*xTrain_norm - bias_1_3');
    output_train_temp = g(W2_3*V1_train - bias_2_3');
    output_train = zeros(output_size, length(output_train_temp));
    % Pick the class with the highest score
    for k = 1:length(output_train)
        [argvalue, argmax] = max(output_train_temp(:,k));
        output_train(argmax,k) = 1;
    end
    C_train_network_3(i) = (1/(2*length(output_train)))*sum(sum(abs(output_train-tTrain)));

    % Calculate the validation classification
    V1_val = g(W1_3*xValid_norm - bias_1_3');
    output_val_temp = g(W2_3*V1_val - bias_2_3');
    output_val = zeros(output_size, length(output_val_temp));
    % Pick the class with the highest score
    for k = 1:length(output_val)
        [argvalue, argmax] = max(output_val_temp(:,k));
        output_val(argmax,k) = 1;  
    end
    C_val_network_3(i) = (1/(2*length(output_val)))*sum(sum(abs(output_val-tValid))); 
    
    % Calculate the test classification
    V1_test = g(W1_3*xTest_norm - bias_1_3');
    output_test_temp = g(W2_3*V1_test - bias_2_3');
    output_test = zeros(output_size, length(output_test_temp));
    % Pick the class with the highest score
    for k = 1:length(output_test)
        [argvalue, argmax] = max(output_test_temp(:,k));
        output_test(argmax,k) = 1;  
    end
    C_test_network_3(i) = (1/(2*length(output_test)))*sum(sum(abs(output_test-tTest))); 
end
[argvalue, argmin] = min(C_val_network_3);
disp('Network 3:')
disp('-------------------------------------')
disp("Lowest validation classification error: "+num2str(argvalue))
disp("Epoch: "+num2str(argmin))
disp("Training classification error: "+num2str(C_train_network_3(argmin)))
disp("Test classification error: "+num2str(C_test_network_3(argmin)))
disp('-------------------------------------')

%% Network 4 - Two hidden layers - 50 neutrons each
% Parameters for Network 4 and intialize weights and biases
M1_4 = 50;
M2_4 = 50;
W1_4 = normrnd(0, 1/sqrt(3072), [M1_4, input_size]);
W2_4 = normrnd(0, 1/sqrt(50), [M1_4, M2_4]);
W3_4 = normrnd(0, 1/sqrt(50), [output_size, M2_4]);
bias_1_4 = zeros(1, M1_4);
bias_2_4 = zeros(1, M2_4);
bias_3_4 = zeros(1, output_size);

% Intialize vector to save class error
C_train_network_4 = zeros(1,epochs);
C_val_network_4 = zeros(1,epochs);
C_test_network_4 = zeros(1,epochs);
% Training loop
for i = 1:epochs
    % Shuffle for the mini batch
    shuffle = randperm(training_size);
    xTrain_shuffle = xTrain_norm(:,shuffle);
    tTrain_shuffle = tTrain(:,shuffle);
   % Vectorized implementation, lets hope everything works
    for j = 1:batch_size:training_size
        % Select the inputs and targets according to batch size
        xTrain_temp = xTrain_shuffle(:,j:j+batch_size-1);
        tTrain_temp = tTrain_shuffle(:,j:j+batch_size-1);
        % Feed forward
        b1 = W1_4*xTrain_temp - bias_1_4';
        V1 = g(b1);
        b2 = W2_4*V1 - bias_2_4';
        V2 = g(b2);
        b3 = W3_4*V2 - bias_3_4';
        O = g(b3);
        % Back-prop
        delta_3 = (tTrain_temp-O).*g_prime(b3);
        delta_2 = (W3_4'*delta_3).*g_prime(b2);
        delta_1 = (W2_4'*delta_2).*g_prime(b1);
        % Update weights and biases
        W3_4 = W3_4 + eta*delta_3*V2';
        W2_4 = W2_4 + eta*(delta_2*V1');
        W1_4 = W1_4 + eta*(delta_1*xTrain_temp');
        bias_1_4 = bias_1_4 - eta*sum(delta_1,2)';
        bias_2_4 = bias_2_4 - eta*sum(delta_2,2)';
        bias_3_4 = bias_3_4 - eta*sum(delta_3,2)';
    end
    % Calculate the training classification
    V1_train = g(W1_4*xTrain_norm - bias_1_4');
    V2_train = g(W2_4*V1_train - bias_2_4');
    output_train_temp = g(W3_4*V2_train - bias_3_4');
    output_train = zeros(output_size, length(output_train_temp));
    % Pick the class with the highest score
    for k = 1:length(output_train)
        [argvalue, argmax] = max(output_train_temp(:,k));
        output_train(argmax,k) = 1;
    end
    C_train_network_4(i) = (1/(2*length(output_train)))*sum(sum(abs(output_train-tTrain)));

    % Calculate the valid classification
    V1_val = g(W1_4*xValid_norm - bias_1_4');
    V2_val = g(W2_4*V1_val - bias_2_4');
    output_val_temp = g(W3_4*V2_val - bias_3_4');
    output_val = zeros(output_size, length(output_val_temp));
    % Pick the class with the highest score
    for k = 1:length(output_val)
        [argvalue, argmax] = max(output_val_temp(:,k));
        output_val(argmax,k) = 1;  
    end
    C_val_network_4(i) = (1/(2*length(output_val)))*sum(sum(abs(output_val-tValid)));
    
    % Calculate the test classification
    V1_test = g(W1_4*xTest_norm - bias_1_4');
    V2_test = g(W2_4*V1_test - bias_2_4');
    output_test_temp = g(W3_4*V2_test - bias_3_4');
    output_test = zeros(output_size, length(output_test_temp));
    % Pick the class with the highest score
    for k = 1:length(output_test)
        [argvalue, argmax] = max(output_test_temp(:,k));
        output_test(argmax,k) = 1;  
    end
    C_test_network_4(i) = (1/(2*length(output_test)))*sum(sum(abs(output_test-tTest)));
end
[argvalue, argmin] = min(C_val_network_4);
disp('Network 4:')
disp('-------------------------------------')
disp("Lowest validation classification error: "+num2str(argvalue))
disp("Epoch: "+num2str(argmin))
disp("Training classification error: "+num2str(C_train_network_4(argmin)))
disp("Test classification error: "+num2str(C_test_network_4(argmin)))
disp('-------------------------------------')

%% Plotting
figure(1)
semilogy(1:epochs,C_train_network_1,'b','LineWidth',2)
hold on
semilogy(1:epochs,C_val_network_1,'--b','LineWidth',2)
semilogy(1:epochs,C_train_network_2,'r','LineWidth',2)
semilogy(1:epochs,C_val_network_2,'--r','LineWidth',2)
semilogy(1:epochs,C_train_network_3,'y','LineWidth',2)
semilogy(1:epochs,C_val_network_3,'--y','LineWidth',2)
semilogy(1:epochs,C_train_network_4,'g','LineWidth',2)
semilogy(1:epochs,C_val_network_4,'--g','LineWidth',2)
xlabel('Epochs')
ylabel('Classification error')
title('Validation classification error')
legend('Network 1 (train)','Network 1 (val)','Network 2 (train)','Network 2 (val)','Network 3 (train)','Network 3 (val)','Network 4 (train)','Network 4 (val)')