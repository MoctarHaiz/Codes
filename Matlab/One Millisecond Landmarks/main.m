close all; clc; clear all;

%%%%%%% Testing Gradient Descent Function
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




%% Read files
close all; clc; clear all;
drawFaces =false;
Files=dir('HELEN/annotation/*.*');
data ={}
for k=1:length(Files)-2
   fid = fopen(strcat(Files(k+2).folder,'\',Files(k+2).name), 'rt');
   imageName = fgetl(fid);
   datacell = textscan(fid, '%f%f', 'Delimiter',',', 'CollectOutput', 1);
   fclose(fid);
   imagepath = strcat('HELEN\images','\',strsplit(imageName),'.jpg');
   data{k,1} = imread(imagepath{1});
   data{k,2} = datacell{1};
  
   if drawFaces
       I =data{k,2};
       frame = insertShape(I,'circle',[data{k,1} 4*ones(1,size(data{1,1},1))'],'LineWidth',5);
       imshow(frame);
   end
end
%%


%%Regressor
rng('default')
R = 5; % Number of initialization per image
n = size(data,1); % Number of training data
N = n*R;
triplets ={}
for k=1:N
    datatemp = data;
    i = randi(n);  
    I = data{i,1};
    datatemp(i,:) =[];
    i = randi(n-1); 
    S =  datatemp{i,2};
    DS = data{i,2} - S;
    triplets{k,1}=I;
    triplets{k,2}=S;
    triplets{k,3}=DS;
end

