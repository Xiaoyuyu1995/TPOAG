clear all;
addpath([pwd, '/funs']);
addpath([pwd, '/datasets']);

%% load data
dataname='MSRC';
load(strcat(dataname,'.mat'));


nv = length(X);
nc = length(unique(Y));

%% Data pre-processing A
disp('------Data preprocessing------');
tic
for v = 1:nv
    a = max(X{v}(:));
    X{v} = double(X{v}./a);
end
toc


%% setting
%% MSRC
anchor_rate = 0.6;
p = 0.8;
lambda1 = 10;
lambda2 = 100;
beta = 10;
%% mai1
IterMax = 150;
filename=['result_main1-' dataname '.txt'];
fid = fopen(filename,'a');
for num1 = 1:length(anchor_rate)
    for num2 = 1:length(p)
        for num3 = 1:length(lambda1)
           for num4 = 1: length(lambda2)
                for num5 = 1: length(beta)
                   [final_result,label,time,iter] = main_new1(X,Y,nv,nc,anchor_rate(num1),p(num2),lambda1(num3),lambda2(num4),beta(num5),IterMax);
                   %[final_result,label,time,iter] = main_new2(X,Y,nv,nc,anchor_rate(num1),p(num2),lambda1(num3),lambda2(num4),beta(num5),IterMax);
                   for n_result = 1:length(final_result)
                        fprintf(fid, '%f ' ,final_result(n_result));
                        fprintf('%f ' ,final_result(n_result));
                   end
                   fprintf('anchor_rate=%f_p=%f_lambda1=%f_lambda2=%f_beta=%f\n', anchor_rate(num1),p(num2),lambda1(num3),lambda2(num4),beta(num5));
                   fprintf(fid, 'anchor_rate=%f_p=%f_lambda1=%f_lambda2=%f_beta=%f\n', anchor_rate(num1),p(num2),lambda1(num3),lambda2(num4),beta(num5));
%                    fprintf(fid, 'Time_all=%f s\n',time);
%                    fprintf(fid, 'Time_average=%f s\n',time/iter);
                end
           end
        end 
    end
end

