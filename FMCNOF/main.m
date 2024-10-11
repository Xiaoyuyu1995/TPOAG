clc
clear 
tic
%% 读入数据
% load mnist4.mat
%load Caltech101-7.mat
%load Caltech101-20.mat
%load Caltech101-all
load AWA.mat
%load NUSWIDEOBJ
% load Reuters
% load HW1256.mat   


%% 数据连接
viewcell=X;
Y = double(Y);
%Y=cell2mat(truelabel(:,1));
viewmat=[];
V=size(viewcell,2);
for i=1:V
    viewcell{i}=double(viewcell{i});
    viewcell{i}=mapstd(viewcell{i});
    viewcell0{i}= viewcell{i}; % 展开矩阵为一列，然后转置为一行。
    viewcell{i} = reshape(viewcell0{i}, size(viewcell{i}));
    viewmat = [viewmat viewcell{i}];
end
s=length(unique(Y));  % s为数据类的个
[n,~]=size(viewmat);
acc=[];nmi=[];purity=[];per=[];rec=[];fscore=[];ari=[];
m=s+7; %mnist4=s+3;cal7=s+7;cal20=s+7;calall=s+7;nus=s+9;nusw=s+9;awa=s+7;reuters=s+7
tic

for II=1:5
%% k-Means
[label, cluster_centers] = litekmeans(viewmat, m);

%% Get B
sigma=10;
Bcell = GetB2(viewmat,cluster_centers,sigma,s,viewcell);
Bmat=cell2mat(Bcell);
for i=1:V
    Bcell{i}= Bcell{i}';
end
Bmat=cell2mat(Bcell);
%% 初始化
Z=zeros(n,m);
B0=[];
alphaV1=[];
J=[];
alphaV=1/V*ones(1,V);%6
for i=1:V
    Z=Z+alphaV(i).*Bcell{i};
end

G=initializeG(m,s,1e-1);
F=Z*G;
for i=1:n
    F(i,:)=F(i,:)./sum(F(i,:));
end
for i=1:V
        Bmat=(Bcell{i})';
        Btemp=reshape(Bmat,m*n,1);
        B0=[B0 Btemp];
end
B=B0'*B0;
BB = max(B,B');
[vIM,dIM] = eig(BB);%求矩阵AA的全部特征值，构成对角阵dIM，并求AA的特征向量构成vIM的列向量
if(min(min(dIM)) < 0)
    disp('Nonegative Definite Matrix;');
end

%% 交互迭代
for iter=1:30  
    %得到G
    [U, S, W] = svd(F'*Z,'econ');
    G = (U*W')';
    %得到F
    F=Z*G;
    F(F<0)=0;
    %得到alpha 
    temp5=reshape(F*G',m*n,1);
    b=2*temp5'*B0;
    [alphaV, val,p] = SimplexQP_ALM(B, b', 1e-2,1.05,1);
    alphaV1=[alphaV1 alphaV];
    Z=zeros(n,m);
    for i=1:V
        Z=Z+alphaV(i)*Bcell{i};
    end
%     J1=trace(Z'*Z)-2*trace(F'*Z*G)+trace(G*F'*F*G');
%     J=[J J1];
end
[maxv,ind]=max(F,[],2);
results = ClusteringMeasure1(Y, ind);
% [ACC,NMI,Purity,PER,REC,Fscore,ARI] = results;
acc=[acc,results(1)];
nmi=[nmi,results(2)];
purity=[purity,results(3)];
per=[per,results(4)];
rec=[rec,results(5)];
fscore=[fscore,results(6)];
ari=[ari,results(7)];
end
t=toc/5
[mean(acc),std(acc);mean(nmi),std(nmi);mean(purity),std(purity);mean(per),std(per);mean(rec),std(rec);mean(fscore),std(fscore);mean(ari),std(ari);]



