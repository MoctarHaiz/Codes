%%Input
clc; clear all; 
data = [
        5 1500 5 480;
        11 2030 12 1090;
        14 1442 6 350;
        8 2501 4 1310;
        12 1300 9 400;
        10 1789 11 500]
gradient_boosting_(data(:,1:3),data(:,4),1000);
    
%%

function model = gradient_boosting_(training_x, training_y, iteration)
    %% Initialize model with a constant value
    F0  = mean(training_y)
%     nb_constants = 10;
%     constants =  min(training_y) + max(training_y)*rand(nb_constants); % Random Constant
%     for i=1:size(constants)
%         mse(i) = LossMeanSquareError(training_y, constants);
%     end
%     [minimum_value, minimun_indice] = min(mse); % argmin
%     F0 = constants(minimun_indice);
    
    learning_rate=0.1;
    %% Iteration
    F{1} = F0*ones(size(training_y))
    for m = 2:iteration+1
            % Compute so-called pseudo-residuals:
              r = DerivativeLossMeanSquareError(training_y, F{m-1});
            
%             %   Fit a base learner (or weak learner, e.g. tree) to pseudo-residuals, i.e. train it using the training set
               h{m-1} = [training_x,r];
               F{m} = F{m-1} +  learning_rate*r;
%              h{m-1}
%             % Compute multiplier by solving the following one-dimensional optimization problem:
%             for i=1:size(constants)
%                 mse(i) = LossMeanSquareError(training_y, F(3-1) + constants*h(m));
%             end
%             [minimum_value, minimun_indice] = min(mse); % argmin
%             multipler = constants(minimun_indice);
% 
%             % Update the model:
%             F(m) = F(m-1) +  multipler*h(m);
    end
    model = F{iteration}
end


%%
function [mse] =  LossMeanSquareError(y,y_estimated)
    if size(y) == size(y_estimated)
        mse = mean((1/2)*pow((y_estimated - y),2));
    end
end


%%
function [results] = DerivativeLossMeanSquareError(y,y_estimated)
    if size(y) == size(y_estimated)
        results =y- y_estimated;
    end
end