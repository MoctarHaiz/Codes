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

