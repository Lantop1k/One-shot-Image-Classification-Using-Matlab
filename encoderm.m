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

%training labels
trainLabels=imds.Labels;

%size of all images
h=28;
w=28;


%number of train images
numtrain=length(imds.Labels);

images={};
%loop through all the images and store their image array and vector
 for i=1:numtrain
     
        %read image
        im=double(imresize(imread(imds.Files{i}),[h,w]));   
       
        images{i}=double(im);
 end
 

%number of train images
numtrain=length(imds.Labels);

%number of test images
numtest=length(imdsTesting.Labels);

%first hidden layer size
hiddenSize1 = 100;

epoch1=400; %epoch

%Train first autoencoder
autoenc1 = trainAutoencoder(images,hiddenSize1, ...
    'MaxEpochs',epoch1, ...
    'EncoderTransferFunction','satlin','DecoderTransferFunction','purelin');

%first layer
view(autoenc1)

%plot autoencoder weights 
figure(1)
plotWeights(autoenc1);
title('weights')

%encode the first layer
encoded1 = encode(autoenc1,images);

%create second layer
hiddenSize2 = 100; %number of neurons in layer
epoch2=100;    %number of epochs

%train the second layer
autoenc2 = trainAutoencoder(encoded1,hiddenSize2, ...
    'MaxEpochs',epoch2, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);

%encode second layer 
encoded2 = encode(autoenc2,encoded1);

%train a softmaxtrix classifier for the multiclass classification
epoch3=400;
softmaxnetwork = trainSoftmaxLayer(encoded2,  dummyvar(imds.Labels),'MaxEpochs',epoch3);

%stack all network 
stacknetwork = stack(autoenc1,autoenc2,softmaxnetwork);
view(stacknetwork)

X= zeros(h*w*3,length(images));
for i = 1:length(images)
   X(:,i) = images{i}(:);
end
y=dummyvar(imds.Labels);

%Perform fine tuning
stacknetwork = train(stacknetwork,X,y);

%Testing 
%initialize labels
predLabels={};
%loop through all test images 
for t=1:numtest
           
          %load image and resize  
          im=imresize(imread(imdsTesting.Files{t}),[h,w]);
          
      
          %compute the class probabilities for each class
          y=stacknetwork(double(im(:)));
          
          %find the class with the maximum probabilities
          [~,idx]=max(y);
          
          %obtain the predicted label
          predLabels{end+1}=char(trainLabels(idx));
end

%convert the predicted labels and actual labels to categorical 
predLabels=categorical(predLabels);
Labels=categorical(Labels);

%compute accuracy
accuracy = 100*sum(predLabels==  Labels)/numel( Labels)



