function [x,tnn, trank] = prox_tnn(x,rho,sX,p,mode,W)
%this function is used to update E of our model,E is the te,sor

% The proximal operator of the tensor nuclear norm of a 3 way tensor
%
% min_X rho*||X||_*+0.5*||X-Y||_F^2
%
% Y     -    n1*n2*n3 tensor
%
% X     -    n1*n2*n3 tensor
% tnn   -    tensor nuclear norm of X
% trank -    tensor tubal rank of X
%

if ~exist('mode','var')
    % mode = 1是采用lateral slice的方法
    % mode = 2是采用front slice的方法
    % mode = 3是采用top slice的方法
    mode = 1;
end
X=reshape(x,sX);
if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1);
else
    Y = X;
end

[n1,n2,n3] = size(Y);
n12 = min(n1,n2);
Y = fft(Y,[],3);
U = zeros(n1,n12,n3);
V = zeros(n2,n12,n3);
S = zeros(n12,n12,n3);
trank = 0;
for i = 1 : n3
    [U(:,:,i),s,V(:,:,i)] = svd(Y(:,:,i),'econ');
     % [m n]=size(Y(:,:,i));
   % M=sqrt(m*n);
    s = diag(s);
     % rho=rho*(m./(s+w));
    rho=rho.* (s.^(W-1));
    s = solve_Lp_w(s, rho, p); 
    S(:,:,i) = diag(s);
    tranki = length(find(s~=0));
    trank = max(tranki,trank);
end
U = U(:,1:trank,:);
V = V(:,1:trank,:);
S = S(1:trank,1:trank,:);

U = ifft(U,[],3);
S = ifft(S,[],3);
V = ifft(V,[],3);

X = tprod( tprod(U,S), tran(V));
if mode == 1
    X = Yi2X(X,3);
elseif mode == 3
    X = shiftdim(X, 2);
else
    X = X;
end

x = X(:);

S = S(:,:,1);
tnn = sum(S(:)); % return the tensor nuclear norm of X
end
