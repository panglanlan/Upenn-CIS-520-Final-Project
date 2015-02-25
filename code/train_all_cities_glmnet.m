    clear;
    
    load '../data/word_train.mat';
    load '../data/price_train.mat';
    load '../data/bigram_train.mat';
    load '../data/city_train.mat';

    load ../data/city_test.mat
    load ../data/word_test.mat
    load ../data/bigram_test.mat

    Y_train = price_train;
    X_train = [city_train word_train bigram_train];
    X_test = [city_test word_test bigram_test];
    
    K = 19;
    sigma = 6;
    pc = 31;
    
    
    X_train_ttl = X_train;
    X_test_ttl = X_test;
    Y_train_ttl = Y_train;
    
	[Ntrain, ~] = size(X_train);
	[Ntest, ~] = size(X_test);
    
	X_ttl = [X_train;X_test];
    
    Prob = sum(X_ttl, 1)/(Ntrain+Ntest);
    Prob(Prob==0)=0.0001;
    info = -log2(Prob);
    for i = 1:length(info)
        X_ttl(:, i) = info(i) * X_ttl(:, i);
    end
    
	dwX_train= X_ttl(1:Ntrain, :);
	dwX_test= X_ttl(Ntrain+1:end, :);
    
    coeff.info_ttl =info;

    glmnet_obj = cvglmnet(dwX_train, Y_train, 'gaussian');
	[~,min_idx] = min(glmnet_obj.cvm);

    coeff.beta_ttl = glmnet_obj.glmnet_fit.beta(:, min_idx);
	coeff.alpha_ttl = glmnet_obj.glmnet_fit.a0(min_idx);
    pred_train_price = dwX_train*coeff.beta_ttl + coeff.alpha_ttl;
	model.residual_ttl = Y_train - pred_train_price;
    
    [U,D,V] = svdsecon([dwX_train; dwX_test],pc);
    X_trunc = U*D;
    model.x_svd_ttl = X_trunc(1:Ntrain, :);
    model.x_svd_ttl_loading = V;
    model.residual_city = cell(7,1);    
    model.x_svd_city =  cell(7,1);
    model.x_svd_city_loading = cell(7,1);
    coeff.info_city = [];
    for i = 1:7
        X_train = X_train_ttl(city_train(:,i)==1,:);
        Y_train = Y_train_ttl(city_train(:,i)==1);
        X_test = X_test_ttl(city_test(:,i)==1,:);
        
        [Ntrain, ~] = size(X_train);
        [Ntest, ~] = size(X_test);

        X_ttl = [X_train;X_test];

        Prob = sum(X_ttl, 1)/(Ntrain+Ntest);
        Prob(Prob==0)=0.0001;
        info = -log2(Prob);

        coeff.info_city(:,i) =info;
        
        for j = 1:length(info)
            X_ttl(:, j) = info(j) * X_ttl(:, j);
        end

        dwX_train= X_ttl(1:Ntrain, :);
        dwX_test= X_ttl(Ntrain+1:end, :);


        glmnet_obj = cvglmnet(dwX_train, Y_train, 'gaussian');
        [~,min_idx] = min(glmnet_obj.cvm);
        
        
        coeff.beta_city(:,i) = glmnet_obj.glmnet_fit.beta(:, min_idx);
        coeff.alpha_city(i) = glmnet_obj.glmnet_fit.a0(min_idx);
        pred_train_price = dwX_train*coeff.beta_city(:,i) + coeff.alpha_city(i);
        

        model.residual_city{i} = Y_train - pred_train_price;

        [U,D,V] = svdsecon([dwX_train; dwX_test],pc);
        X_trunc = U*D;
        model.x_svd_city_loading{i} = V;
        model.x_svd_city{i} = X_trunc(1:Ntrain, :);
        
    end
    
    lasso_coeff=coeff;
    knn_model=model;
    save('model.mat', 'lasso_coeff', 'knn_model');