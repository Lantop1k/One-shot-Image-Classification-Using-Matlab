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


%size of all images
h=300; %height
w=300; %width

%number of train images
numtrain=length(imds.Labels);

%number of test images
numtest=length(imdsTesting.Labels);

%store all extracted images
images=zeros(h,w,numtrain); %image size * total number of images  
vec=zeros(h*w,numtrain);    %vector of all images
    
%loop through all the images and store their image array and vector
 for i=1:numtrain
        im=rgb2gray(imread(imds.Files{i}));
        
        images(:,:,i)=imadjust(imresize(im,[h,w]));
        vec(:,i)=reshape(images(:,:,i),h*w,1);
 end
  
%calculate mean face
meanface=mean(vec,2); 

%create face space by subtracting vectors of each images
facespace=vec-repmat(meanface,1,numtrain);
     
%compute the eigen vector and values
L=facespace'*facespace;
[eigenvector,~]=eig(L);

%compute the convariance matrix of the matrix
U=facespace*eigenvector;

% vector projection
omega=U'*facespace;

predLabels={};
for t=1:numtest
    
    %load the test image to be recognized
    im=rgb2gray(imread(imdsTesting.Files{t}));
    testIm=imadjust(imresize(im,[h,w]));
        
    im=reshape(testIm,h*w,1);
    imtest=double(im);
    
    imd=imtest-meanface;
    
    % projection of the test face on the eigenfaces
    om=U'*imd;
   
    d=repmat(om,1,numtrain)-omega;
    
    distance=zeros(numtrain,1);
     
    % find the distance from all training faces
    for i=1:numtrain
        distance(i,1)=norm(d(:,i));
    end
    
    %find minimum distance
    [~,idx]=min(distance);
    predLabels{end+1}=char(imds.Labels(idx));
end

 predLabels=categorical(predLabels);
 Labels=categorical(Labels);

accuracy = 100*sum(predLabels==  Labels)/numel( Labels)