function [Acc,acc_iter,Beta,Yt_pred] = MEDA(Xs,Ys,Xt,Yt,options)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d       :  dimension after manifold feature learning (default: 20)
%%%%% options.T       :  number of iteration (default: 10)
%%%%% options.lambda  :  lambda in the paper (default: 10)
%%%%% options.eta     :  eta in the paper (default: 0.1)
%%%%% options.rho     :  rho in the paper (default: 1.0)
%%%%% options.base    :  base classifier for soft labels (default: NN)
%%%%% options.w_method:  the method to get weights (default: B)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
    %fprintf('MEDA starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end
    if ~isfield(options,'w_method')
        options.w_method = 'S';
    end
    if ~isfield(options,'edit_mu')
        options.edit_mu = true;
    end
    if ~isfield(options,'mu_dim')
        options.mu_dim = 0;
    end


    % Manifold feature learning
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
    Xt = double(Xt_new');

    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];
 %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Generate soft labels for the target domain
    knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
    Cls = knn_model.predict(X(:,n + 1:end)');

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    E = diag(sparse([ones(n,1);zeros(m,1)]));
    
    % Compute weight on S
    if options.w_method == 'B'
        w = ones(1, C)/C;
    end
    if options.w_method == 'S'
        w = zeros(1,C);
        for c = 1:C
            w(1,c) = length(find(Ys == c)) ^ options.mu_dim;
        end
        w = w/sum(w);
    end
    
    for t = 1 : options.T
        % Update weights on T
        if options.w_method == 'T'
            w = zeros(1,C);
            for c = 1:C
                w(1,c) = length(find(Cls == c))^ options.mu_dim;
            end
            w(isinf(w)) = 0;
            w = w/sum(w);
        end
        if options.w_method == 'A'
            w = zeros(1,C);
            for c = 1:C
                w(1,c) = (length(find(Cls == c))/length(find(Ys == c)))^ options.mu_dim;
            end
            w = w/sum(w);
        end
        if options.w_method == 'V'
            w = zeros(1,C);
            for c = 1:C
                w(1,c) = (length(find(Cls == c))*length(find(Ys == c)))^ options.mu_dim;
            end
            w = w/sum(w);
        end
        % Estimate mu
        if options.edit_mu
            mu = estimate_mu(Xs',Ys,Xt',Cls,w);
        else
            mu = estimate_mu(Xs',Ys,Xt',Cls,ones(1, C)/C);
        end
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys));
        N = 0;
        
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = zeros(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(Cls == c)) = -1 / length(find(Cls == c));
            e(isinf(e)) = 0;
            N = N + e * e' * w(1,c) * C;
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
        %fprintf('Iteration:[%02d]>>mu=%.2f,Acc=%f\n',t,mu,Acc);
    end
    Yt_pred = Cls;
    %fprintf('MEDA ends!\n');
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end