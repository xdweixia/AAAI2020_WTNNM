%%--------------------------------------------------------------------------
% Reference:
%
% Quanxue Gao, Wei Xia, Zhizhen Wan, De-Yan Xie, Pu Zhang: 
% Tensor-SVD Based Graph Learning for Multi-View Subspace Clustering. 
% AAAI 2020: 3930-3937
%
% version 1.0 --Jan./2019
%
%--------------------------------------------------------------------------
% Written by (xd.weixia@gmail.com)
%--------------------------------------------------------------------------

%% Load Setting
clc;
clear all;
addpath([pwd, '/Dataset']);
addpath([pwd, '/funs']);

%% Load Dataset
dataname='yale';
load(strcat('Dataset/',dataname,'.mat'));
cls_num = length(unique(gt));
Label = double(gt);
num_views = length(X); N = size(X{1},2); % view number; sample number

%% Data preprocessing
for v = 1:num_views
    X{v}=NormalizeData(X{v});
end

%% Hyper-Parameters
lambda=[1];
beta = [1, 10, 100]';
fid=fopen('result_Yale.txt','wt');

%% Optimizataion

%% Initialize and Settings
for k=1:num_views
    Z{k} = zeros(N,N);
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N);
end
K = num_views;
w = zeros(N*N*K,1); g = zeros(N*N*K,1);
dim1 = N;dim2 = N;dim3 = K;
sX = [N, N, K];
epson = 1e-7; mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e10; pho_rho = 2;
converge_Z=[]; converge_Z_G=[];

iter = 0;Isconverg = 0; num = 0;
while(Isconverg == 0)
    fprintf('---------processing iter %d--------\n', iter+1);
    num = num + 1;
    for k=1:K
        %% Update Z^k
        tmp = (X{k}'*Y{k} + mu*X{k}'*X{k} - mu*X{k}'*E{k} - W{k})./rho +  G{k};
        Z{k}=pinv(eye(N,N)+ (mu/rho)*X{k}'*X{k})*tmp;
        
        %% Update E^k
        F = [X{1}-X{1}*Z{1}+Y{1}/mu;X{2}-X{2}*Z{2}+Y{2}/mu;X{3}-X{3}*Z{3}+Y{3}/mu];
        [Econcat] = solve_l1l2(F,lambda/mu);
        E{1} = Econcat(1:size(X{1},1),:);
        E{2} = Econcat(size(X{1},1)+1:size(X{1},1)+size(X{2},1),:);
        E{3} = Econcat(size(X{1},1)+size(X{2},1)+1:size(X{1},1)+size(X{2},1)+size(X{3},1),:);
        
        %% Update E^k
        Y{k} = Y{k} + mu*(X{k}-X{k}*Z{k}-E{k});
    end
    
    %% Update G
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:);
    [g, objV] = wshrinkObj_weight(z + 1/rho*w,beta/rho,sX,0,3);
    G_tensor = reshape(g, sX);
    %% Update W
    w = w + rho*(z - g);
    
    %% Record the iteration information
    history.objval(iter+1) = objV;
    %% Coverge condition
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z= norm(X{k}-X{k}*Z{k}-E{k},inf);
            fprintf('norm_Z %7.10f', history.norm_Z);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        G{k} = G_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-G{k},inf)>epson)
            history.norm_Z_G= norm(Z{k}-G{k},inf);
            fprintf('norm_Z_G %7.10f \n', history.norm_Z_G);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
    if (iter>200)
        Isconverg = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
    S = 0;
    for k=1:K
        S = S + abs(Z{k})+abs(Z{k}');
    end
    C = SpectralClustering(S./K, cls_num);
    [result(num,:)] = ClusteringMeasure1(Label, C);
end

My_result = result(num,:);
fprintf(fid,'lambda: %f ', lambda);
fprintf(fid,'beta: %g %g %g \n', beta);
fprintf(fid,'%g %g %g %g %g %g %g \n ', My_result);