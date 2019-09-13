clear all; clc;
%% The stored patterns
x1=[[ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [ -1, 1, 1, 1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]];

x2=[[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],[ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [ -1, -1, -1, 1, 1, 1, 1, -1, -1, -1]];

x3=[[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],...
    [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],...
    [ -1, -1, -1, -1, -1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],...
    [ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, -1, -1, -1, -1, -1, -1, -1],[ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [ 1, 1, 1, 1, 1, 1, 1, 1, -1, -1]];

x4=[[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, 1, 1, 1, -1],[ -1, -1, 1, 1, 1, 1, 1, 1, 1, -1],...
    [ -1, -1, 1, 1, 1, 1, 1, 1, -1, -1]];

x5=[[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],...
    [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],...
    [ -1, 1, 1, -1, -1, -1, -1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],[ -1, 1, 1, 1, 1, 1, 1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],[ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1],...
    [ -1, -1, -1, -1, -1, -1, -1, 1, 1, -1]];

pattern_q1 = [[-1, -1, -1, -1, -1, -1, -1, -1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],...
    [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1],...
    [-1, -1, -1, -1, -1, 1, 1, 1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [1, 1, 1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1, -1, -1, -1],...
    [1, 1, 1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, -1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1]];

pattern_q2 = [[1, -1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, 1, 1, 1, 1, -1, -1, 1],...
    [1, -1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, 1, 1, 1, 1, -1, -1, 1],...
    [1, -1, -1, 1, 1, 1, 1, -1, -1, 1], [1, -1, -1, -1, -1, -1, -1, -1, -1, 1], [1, -1, -1, -1, -1, -1, -1, -1, -1, 1],...
    [1, 1, 1, 1, 1, 1, 1, -1, -1, 1], [1, 1, 1, 1, 1, 1, 1, -1, -1, 1], [1, 1, 1, 1, 1, 1, 1, -1, -1, 1],...
    [1, 1, 1, 1, 1, 1, 1, -1, -1, 1], [1, 1, 1, 1, 1, 1, 1, -1, -1, 1], [1, 1, 1, 1, 1, 1, 1, -1, -1, 1],...
    [1, 1, 1, 1, 1, 1, 1, -1, -1, 1]];

pattern_q3 = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],...
    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1],...
    [-1, 1, 1, 1, -1, -1, 1, 1, 1, -1], [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, 1, 1, 1, 1, -1, -1, -1],...
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]];

stored_pattern = [x1',x2',x3',x4',x5'];
feed_pattern = [pattern_q1',pattern_q2',pattern_q3'];

%% Parameters and weight matrix
N = length(x1);

W = zeros(N);
for k = 1:size(stored_pattern,2)
    W = W + stored_pattern(:,k)*stored_pattern(:,k)';
end
W = W.*(1/N);
W = W - diag(diag(W));

%% Q1 pattern
output_pattern = zeros(size(feed_pattern));
output_digit = zeros(1,size(feed_pattern,2));
for i = 1:size((feed_pattern),2)
    state = feed_pattern(:,i);
    state_old = zeros(size(state));
    while state ~= state_old
        state_old = state;
        for j = 1:length(state)
            z = W(j,:)*state;
            S = sign(z);
            if S == 0
                S = 1;
            end
            state(j) = S;
        end
    end
    output_pattern(:,i) = state;
    
    for k = 1:size((stored_pattern),2)
        if state == stored_pattern(:,k)
            output_digit(i) = k;
            break
        elseif state == (-stored_pattern(:,k))
            output_digit(i) = -k;
            break
        elseif k == size((stored_pattern),2)
            output_digit(i) = k+1;
        end    
    end
end

for i = 1:size(output_pattern,2)
    disp(['The classified digit for Q',num2str(i),' is: ',num2str(output_digit(i))])
    disp(['The steady state pattern for Q',num2str(i),' is:'])
    reshape(output_pattern(:,i),[10,16])'
    
end


