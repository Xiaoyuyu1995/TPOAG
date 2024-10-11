clear;
addpath([pwd, '/funs']);
% addpath([pwd, '/datasets']);

ds = {'Caltech101-20','CCV','Caltech101-all_fea','SUNRGBD_fea','NUSWIDEOBJ','AwA_fea','YoutubeFace_sel_fea'...
'scene15Big','MNIST_fea'};
datapath = '/home/zpp/scholar/dataset/MultiView/';
addpath(genpath('D:\XDU_Master\WritePaper\Multi-view Dataset'));
% X:n*d
for di = 1:1
    dataname=ds{di};
%     load(strcat(datapath,dataname,'.mat'));
    load('scene15Big.mat');
    nV = length(X);
    
    
    % Select a data preprocessing method, or no data preprocessing
    %% Data pre-processing A
%     disp('------Data preprocessing------');
%     tic
%     for v=1:nV
%         a = max(X{v}(:));
%         X{v} = double(X{v}./a);
%     end
%     toc

    %% Data pre-processing B
%     disp('------Data preprocessing------');
%     tic
%     for v=1:nV
%         XX = X{v};
%         for n=1:size(XX,1)
%             XX(n,:) = XX(n,:)./norm(XX(n,:),'fro');
%         end
%         X{v} = double(XX);
%     end
%     toc

    %% Data pre-processing C
%     for v=1:nV
%         X{v} = ( X{v} - mean(X{v},2) )./ std( X{v},0,2);
%     end

    
    anchor_rate=0.1:0.1:1;
    projev = 1.5;
    i = 1;
    c = length(unique(Y));
    opt1. style = 1;
    opt1. IterMax =50;
    opt1. toy = 0;
%     t1=clock;
for iii=1:length(anchor_rate)
    tic;
    [P1, alpha, y(:,i)] = FastmultiCLR(X, c, anchor_rate(iii), opt1,10);
%     t2=clock;
    [result(i,:)] = Clustering8Measure(Y, y(:,i));
%     time(i) = etime(t2,t1);
    time(i) = toc;
    
    fprintf('Dataset:%s\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',dataname,result(1),result(2),result(3),result(4),result(5),result(6),result(7),time);
    fid = fopen('allresult.txt','a');
    fprintf(fid,'Dataset:%s\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t Time:%.4f\n\n',dataname,result(1),result(2),result(3),result(4),result(5),result(6),result(7),time);
    fclose(fid);
end
    clear X result time y
end
