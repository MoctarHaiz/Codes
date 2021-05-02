%%%%%%% Testing Gradient Descent Function
%https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html
%https://www.codeproject.com/Articles/879043/Implementing-Gradient-Descent-to-Solve-a-Linear-Re
%https://towardsdatascience.com/understanding-the-mathematics-behind-gradient-descent-dde5dc9be06e
function main()
close all; clc; clear all;
load accidents
x =hwydata(:,14); % Must be Nomalized
y =hwydata(:,4);
format long
x =normalize(x);
y =normalize(y);
% y =ax+b ->
x = [x ones(length(x), 1)];


%% Gradient Descent
weights = zeros(2,1);%a = weights(1,1); b = weights(2,1); 
learning_rate = 0.1;
iteration=1000;
[weight_updated, cost_history]= GradientDescent_LossMSE_UpdateWeight(x,y,learning_rate,iteration,weights);

%% Plot Data
subplot(2, 1, 1);
scatter(x(:,1),y);
hold on;

%% Plot Estimated Regression Line
a = x\y; % y =ax+b
y_estimated = a'*x';
plot(x(:,1),y_estimated,'color','green', 'LineWidth',10);

%% Plot Results
y_estimated = weight_updated'*x';
plot(x(:,1),y_estimated,'color','blue');

% Plotting our cost function on a different figure to see how we did
subplot(2, 1, 2);
plot(cost_history, 1:iteration);
end




function [weight_updated,cost_history] = GradientDescent_LossMSE_UpdateWeight(x,y, learning_rate, iteration, weights)
% Gradient Descent -> Loss decreases fastest if one goes in the negative
% direction (-Derivative(Loss)).
% - y_estimated = weight*x
% - Loss=1/n(sum(yi-yi_estimated)2
% - Derivative(Loss) = -2/n(sum(yi-yi_estimated)
% - yi_estimated_new = yi_estimated -learning_rate*Derivative(Loss)
n = length(y);
if(n==size(x,1))
    cost_history = zeros(iteration,1);
    for i=1:iteration
        y_estimated = weights'*x';   
        weights(1) = weights(1) - learning_rate* (-2/n)*(sum((y-y_estimated')'*x(:,1)));
        weights(2) = weights(2) - learning_rate* (-2/n)*(sum((y-y_estimated')));
        %Generally  weights(i) = weights(i) - learningRate * (-2/n) * (sum((y-y_estimated) * x(:, i));
        cost_history(i) = (1/n)*sum((y'- weights'*x').*(y'- weights'*x')); 
    end
    weight_updated =weights;
else
    weight_updated= 0;
    cost_history =0;
    disp('y & y_estimated sizes must be equal'); 
end
end



