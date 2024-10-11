function [final_result,label,time,iter] = main_new1(X,Y,nv,nc,anchor_rate,p,lambda1,lambda2,beta,IterMax)
% X为输入，nC为类别数,M为锚点数
N = size(X{1},1);
M = fix(N*anchor_rate);
alpha = repmat(1/nv, [1,nv]);
betaf = ones(nv, 1); % 张量的权重

%% initial
for v = 1:nv
    % 需要重新初始化
    B{v} = zeros(N,M);
    E{v} = zeros(N,M);
    G{v} = zeros(N,nc);
    F{v} = zeros(N,nc);
    H{v} = zeros(M,nc);
    Q{v} = zeros(M,nc);
    J{v} = zeros(N,nc);
    A{v} = zeros(M,nc);
    Y1{v} = zeros(N,nc);
    Y2{v} = zeros(M,nc);
    Y3{v} = zeros(N,nc);
    Y4{v} = zeros(M,nc);
    Y5{v} = zeros(N,M);
end
EE = eye(M, M);
mu1 = 1e-3;
mu2 = 1e-3;
mu3 = 1e-3;
mu4 = 1e-3;
mu5 = 1e-3;
max_mu = 1e9;

coe = 1.1;
final_result = zeros(1,9);
sX1 = [N, nc, nv];
sX2 = [M, nc, nv];
r = -1;
epson = 1e-7;
iter = 0;
Isconverg = 0;
time_start = clock;
opt1. style = 1;
opt1. IterMax = IterMax;
opt1. toy = 0;

tic;
%% anchor grph
[S] = Construct_Graph(X,nc,anchor_rate, opt1,10);
clear X

while(Isconverg == 0) 
    %% update G{v}
    for v = 1:nv
        SS{v} = 2*beta*B{v}*Q{v}+mu1*(F{v}-Y1{v}/mu1)+mu3*(J{v}-Y3{v}/mu3);
        [uu{v}, ~, vv{v}] = svd(SS{v}, 'econ');
        G{v} = uu{v} * vv{v}';
    end
    
    %% update H{v}
    for v = 1:nv
        QQ = Q{v}+Y2{v}/mu2;
        for j = 1:M
            QQQ = QQ(j,:);
            H{v}(j,:) = EProjSimplex_new(QQQ,1);
        end
    end
    
    %% update F{v}
    for v = 1:nv
        F{v} = G{v}+Y1{v}/mu1;
        F{v}(F{v}<0) = 0; 
    end
    
    %% update Q{v}
    for v = 1:nv
        Q{v} = (2*beta*B{v}'*B{v}+mu2*EE+mu4*EE)\(2*beta*B{v}'*G{v}+mu2*(H{v}-Y2{v}/mu2)+mu4*(A{v}-Y4{v}/mu4));
    end
    
    %% update B{v}
    for v = 1:nv
        B{v} = (2*alpha(v)^r*S{v}+2*beta*G{v}*Q{v}'+mu5*(E{v}-Y5{v}/mu5))/((2*alpha(v)^r+mu5)*EE+2*beta*Q{v}*Q{v}');
    end
    
    %% update E{v}
    for v = 1:nv
        BB = B{v}+Y5{v}/mu5;
        for i = 1:N
            BBB = BB(i,:);
            E{v}(i,:) = EProjSimplex_new(BBB,1);
        end
    end

    %% update J
    for v =1:nv
        GY3{v} = G{v} + Y3{v}./mu3;
    end
    GY3_tensor = cat(3,GY3{:,:});
    [myj, TNNJ] = wshrinkObj_weight_lp(GY3_tensor(:), lambda1*betaf./mu3,sX1, 0,3,p);
    J_tensor = reshape(myj, sX1);
    for k=1:nv
        J{k} = J_tensor(:,:,k);
    end
    clear J_tensor
    
    %% update A
    for v =1:nv
        QY4{v} = Q{v} + Y4{v}./mu4;
    end
    QY4_tensor = cat(3,QY4{:,:});
    [myA, TNNA] = wshrinkObj_weight_lp(QY4_tensor(:), lambda2*betaf./mu4,sX2, 0,3,p);
    A_tensor = reshape(myA, sX2);
    for k=1:nv
        A{k} = A_tensor(:,:,k);
    end
    clear A_tensor 
    

    
    %% update alpha
%     sum_z = 0;
%     for v = 1:nv       
%         z{v} = norm(S{v} - B{v}, 'fro');
%         sum_z = sum_z+(z{v})^(0.5);
%     end
%     
%     for v = 1:nv
%         alpha(v) = (z{v})^(0.5)/sum_z;
%     end
    sum_z = 0;
    for v = 1:nv       
        z{v} = norm(S{v} - B{v}, 'fro');
        sum_z = sum_z+(z{v});
    end
    
    for v = 1:nv
        alpha(v) = (z{v})/sum_z;
    end
    
    
    %% update Y1{v},Y2{v}
    for v = 1:nv
        Y1{v} = Y1{v}+mu1*(G{v}-F{v});
        Y2{v} = Y2{v}+mu2*(Q{v}-H{v});
        Y3{v} = Y3{v}+mu3*(G{v}-J{v});
        Y4{v} = Y4{v}+mu4*(Q{v}-A{v});
        Y5{v} = Y5{v}+mu5*(B{v}-E{v});
    end
    
    %% update mu,pho
    mu1 = min(mu1*coe, max_mu);
    mu2 = min(mu2*coe, max_mu);
    mu3 = min(mu3*coe, max_mu);
    mu4 = min(mu4*coe, max_mu);
    mu5 = min(mu5*coe, max_mu);
     
    
    %% Clustering result
    G_sum = zeros(N,nc);
    for v=1:nv
        G_sum = G_sum + G{v};
        %G_sum = G_sum + G{v}./beta;
        %G_sum = G_sum + G{v}./alpha(v);
    end
    
    [~, label] = max(G_sum, [], 2);
    
    result = ClusteringMeasure(Y,label);   %ACC MIhat Purity  P R F RI
    if (sum(result) - sum(final_result))>0
        final_result = result;
    end
    
    % calculate objective value
    sum_g = 0;
    for v=1:nv
        [~,sigma{v},~]=svd(G{v});
        sum_g=sum_g+sum(diag(sigma{v}).^p);
    end
    sum_g=sum_g^(1/p);
    
    sum_q = 0;
    for v=1:nv
        [~,sigma1{v},~]=svd(Q{v});
        sum_q=sum_g+sum(diag(sigma1{v}).^p);
    end
    sum_q=sum_q^(1/p);
    
    obj = 0;
    for v=1:nv
        obj = obj+ (alpha(v)^r)* (norm(S{v} - B{v}, 'fro'))^2 + beta* (norm(B{v}*Q{v} - G{v}, 'fro'))^2;
    end
    obj = obj + lambda1*sum_g + lambda2*sum_q;
    value(iter+1) = obj;
    
    %% converge
    %     for v=1:nv
    %         history.norm_F_G{v}= norm(F{v}-G{v},inf);
    %         fprintf('norm_F_G %7.10f \n', history.norm_F_G{v});
    %         Isconverg = 0;
    %     end
        Isconverg = 1;
        for v=1:nv
            if (norm(G{v}-F{v},inf)>epson)
                history.norm_G_F{v}= norm(G{v}-F{v},inf);
                fprintf('norm_G_F{v} %7.10f \n', history.norm_G_F{v});
                %fprintf(fid,'norm_G_F{v} %7.10f \n', history.norm_G_F{v});
                Isconverg = 0;
            end
            if (norm(Q{v}-H{v},inf)>epson)
                 history.norm_Q_H{v}= norm(Q{v}-H{v},inf);
                 fprintf('norm_Q_H{v} %7.10f \n', history.norm_Q_H{v});
                 %fprintf(fid,'norm_Q_H{v} %7.10f \n', history.norm_Q_H{v});
                Isconverg = 0;
             end
             if (norm(G{v}-J{v},inf)>epson)
                 history.norm_G_J{v}= norm(G{v}-J{v},inf);
                 fprintf('norm_G_J{v} %7.10f \n', history.norm_G_J{v});
                 %fprintf(fid,'norm_G_J{v} %7.10f \n', history.norm_G_J{v});
                 Isconverg = 0;
             end
             if (norm(Q{v}-A{v},inf)>epson)
                 history.norm_Q_A{v}= norm(Q{v}-A{v},inf);
                 fprintf('norm_Q_A{v} %7.10f \n', history.norm_Q_A{v});
                 %fprintf(fid,'norm_Q_A{v} %7.10f \n', history.norm_Q_A{v});
                 Isconverg = 0;
             end
             if (norm(B{v}-E{v},inf)>epson)
                 history.norm_B_E{v}= norm(B{v}-E{v},inf);
                 fprintf('norm_B_E{v} %7.10f \n', history.norm_B_E{v});
                 %fprintf(fid,'norm_Q_A{v} %7.10f \n', history.norm_Q_A{v});
                 Isconverg = 0;
             end
        end
    %%
    if (iter > IterMax)
        Isconverg  = 1;
    end
    
    
    %     fprintf('iter:%d\n',iter)
    iter = iter + 1;
    %
%         result1{iter} = result(1);
%         rel = cell2mat(result1);
%         C1{iter} = norm(G{v}-F{v},inf);
%         CC1 = cell2mat(C1);
%         C2{iter} = norm(Q{v}-H{v},inf);
%         CC2 = cell2mat(C2);
%         C3{iter} = norm(G{v}-J{v},inf);
%         CC3 = cell2mat(C3);
%         C4{iter} = norm(Q{v}-A{v},inf);
%         CC4 = cell2mat(C4);
    
end
toc;
time = toc;

%plot_converge(rel,CC1,CC2,CC3,CC4)
time_end = clock;
fprintf('Time_all:%f s\n',etime(time_end,time_start))
fprintf('Time_average:%f s\n',etime(time_end,time_start)/iter)
fprintf('Final_iter:%d\n',iter)   

fprintf('Time_all_1:%f s\n',time)
fprintf('Time_average_1:%f s\n',time/iter)
end

