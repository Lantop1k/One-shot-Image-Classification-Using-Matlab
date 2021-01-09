clc
clear all
close all

%load training image folder
imageFolder=fullfile('Train'); %Train
imds=imageDatastore(imageFolder,'LabelSource','foldernames','IncludeSubfolders',true); %data store

%load test labels
load testLabel

Labels={};
for i=1:length(testLabel)
    Labels{i}=testLabel(i,:); 
end


%load testing image folder
imageFolder=fullfile('Test'); %Test
imdsTesting=imageDatastore(imageFolder,'Labels',categorical(Labels)); %data store

%resize image store in smaller size 
inputsize=[64 64]; %input image size
imds.ReadFcn=@(loc) imresize(imread(loc),inputsize); %resize the training image folders
imdsTesting.ReadFcn=@(loc) imresize(imread(loc),inputsize); %resize the testing image folder


%create deep learning layers
inputimagesize=[64 64 3];

%specify the number of image class
nImage=100;

%create a deep learning layer for image classification
layers = [
    imageInputLayer(inputimagesize)

    convolution2dLayer(5,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
       
    maxPooling2dLayer(2,'Stride',2)
       
    convolution2dLayer(5,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(nImage)
    softmaxLayer
    classificationLayer];

%specify training options 
options = trainingOptions('sgdm', ...
    'MaxEpochs',30, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsTesting, ...
    'ValidationFrequency',20, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imds,layers,options);

