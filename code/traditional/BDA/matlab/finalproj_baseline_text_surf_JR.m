% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'books', 'dvd', 'elec', 'kitchen'};
w_methods = {'B', 'S', 'T'};
resample = true;
imbalance_rate = 0.2;
list_acc = [];
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
        
        if resample
            pos_index = find(Ys== 2);
            test1 = pos_index(randi(length(pos_index), round(length(pos_index)*imbalance_rate),1));
            sample_index = [find(Ys == 1);pos_index(randi(length(pos_index), round(length(pos_index)*imbalance_rate),1))];
            sample_index = sample_index(randperm(length(sample_index)));
            Xs = Xs(sample_index,:);
            Ys = Ys(sample_index,:);
        end
        
        % meda
        options.gamma = 1.0;
        options.lambda = 0.1;
        options.kernel_type = 'linear';
        options.T = 10;
        options.dim = 100;
        options.mu = 0.5;
        options.mode = 'W-BDA';
        fprintf('%s --> %s : \n', src, tgt);
        for k = 1:3
            options.w_method = w_methods{k};
            [Acc,acc_ite,~] = BDA(Xs,Ys,Xt,Yt,options);
            fprintf('%s: %.2f accuracy \n',options.w_method, Acc * 100);
            
        end
    end
end
