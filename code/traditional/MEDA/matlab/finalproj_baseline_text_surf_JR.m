% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'books', 'dvd', 'elec', 'kitchen'};
w_methods = {'B', 'S', 'T','A' , 'V'};
imbalance_rate = 0.2;
resample = "S";  % 修改为"T" "ST" 分别尝试一下
list_acc = [];
fprintf('resample: %s \n\n', resample);
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load(['./Amazon_review/' src '_400.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xs = zscore(fts,1);    clear fts
        Ys = labels+1;           clear labels  % change labels from (0,1) to (1,2)
        
        load(['./Amazon_review/' tgt '_400.mat']);     % target domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts,1);     clear fts
        Yt = labels+1;            clear labels  % change labels from (0,1) to (1,2)
        
        if contains(resample,'S')
            pos_index = find(Ys== 2);
            sample_index = [find(Ys == 1);pos_index(randi(length(pos_index), round(length(pos_index)*imbalance_rate),1))];
            sample_index = sample_index(randperm(length(sample_index)));
            Xs = Xs(sample_index,:);
            Ys = Ys(sample_index,:);
        end
        if contains(resample,'T')
            pos_index = find(Yt== 2);
            sample_index = [find(Yt == 1);pos_index(randi(length(pos_index), round(length(pos_index)*imbalance_rate),1))];
            sample_index = sample_index(randperm(length(sample_index)));
            Xt = Xt(sample_index,:);
            Yt = Yt(sample_index,:);
        end
        
        % meda
        options.d = 20;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 10;
        options.mu_edit = false;
        fprintf('%s --> %s : \n', src, tgt);
        
        options.w_method = 'B';
        options.mu_dim = 0;
        options.mu_edit = false;
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('B: %.2f accuracy \n', Acc * 100);
        
        
        options.mu_edit = true;
        
        options.mu_dim = 2;
        options.w_method = 'T';
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s %d edit: %.2f accuracy \n',options.w_method, options.mu_dim, Acc * 100);
        
        options.mu_dim = 2;
        options.w_method = 'A';
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s %d edit: %.2f accuracy \n',options.w_method, options.mu_dim, Acc * 100);
        
        
        options.mu_dim = 2;
        options.w_method = 'V';
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s %d edit: %.2f accuracy \n',options.w_method, options.mu_dim, Acc * 100);
    end
end
