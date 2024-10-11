% This code runs the LMVSC algorithm on Caltech7 dataset and records its
% performance. The results are output in *.txt format.

% Notice: The dataset is organized in a cell array with each element being
% a view. Each view is represented by a matrix, each row of which is a
% sample.

% The core of LMVSC is encapsulated in an independent matlab function.
% Visit lmv.m directly, if you want to learn the details of its
% implementation.

clear;

addpath('./datasets');
% addpath(genpath('D:\XDU_Master\WritePaper\Multi-view Dataset'));
load('HW1256');

% 2022.02.23 change record: the structrue of dataset
data = X';
labels = Y;

X=data;
y=labels;

nV=length(X);
    
ns=length(unique(y));

%% Data preprocessing
% Select a data preprocessing method, or no data preprocessing
%% Data pre-processing A
disp('------Data preprocessing------');
tic
for v=1:nV
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end
toc

%% Data pre-processing B
% disp('------Data preprocessing------');
% tic
% for v=1:nV
%     XX = X{v};
%     for n=1:size(XX,1)
%         XX(n,:) = XX(n,:)./norm(XX(n,:),'fro');
%     end
%     X{v} = double(XX);
% end
% toc

%% Data pre-processing C
% for v=1:nV
%     X{v} = ( X{v} - mean(X{v},2) )./ std( X{v},0,2);
% end

%% Parameter 1: number of anchors (tunable)
% numanchor=[ns 100 200];
numanchor = [60];

% Parameter 2: alpha (tunable)
% alpha=[0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000];
alpha = [1];

for j=1:length(numanchor)
    
    % Perform K-Means on each view
    parfor i=1:nV
        rand('twister',5489);
        [~, H{i}] = litekmeans(X{i},numanchor(j),'MaxIter', 100,'Replicates',10);
    end

    for i=1:length(alpha)
        fprintf('params:\tnumanchor=%d\t\talpha=%f\n',numanchor(j),alpha(i));
        tic;
        
        % Core part of this code (LMVSC)
        [F,ids] = lmv(X',y,H,alpha(i));
        
        % Performance evaluation of clustering result
        result=ClusteringMeasure1(y,ids);
        
        t=toc;
        fprintf('result:\t%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n',[result t]);
        
        % Write the evaluation results to a text file
        dlmwrite('MSRC.txt',[alpha(i) numanchor(j) result t],'-append','delimiter','\t','newline','pc');
        
    end
end