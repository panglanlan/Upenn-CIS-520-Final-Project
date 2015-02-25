function Y_pred = idf_svd_knn_glmnet(X_train, Y_train, X_test, pc, K, on, IDF, sigma)
	% input:
	%      X_train: train samples
	% 	   Y_train: train lables
	% 	   X_test: test samples
	% 	   pc: # of principle components (30 for lastest submit)
		%      on: 1 for add predicted residual to prediction made by glmnet (1 seems always better)
	% 		   0 for original best CV-ed glmnet
	%          this is for comparing purpose;
	%      IDF:1 use inverse document frequency (1 seems always better)
    %          0 do not use inverse document frequency
    % 	   K:  # of nearest neighbors (19 for latest submit)
    %      sigma: kernel knn bandwidth (8 for lastest submit)
	% ouput:
	%      predictions
	[Ntrain, ~] = size(X_train);
	[Ntest, ~] = size(X_test);
    
	X_ttl = [X_train;X_test];
    if IDF == 1
        Prob = sum(X_ttl, 1)/(Ntrain+Ntest);
        info = -log2(Prob);
    % 	info = info./max(info);
        for i = 1:length(info)
            X_ttl(:, i) = info(i) * X_ttl(:, i);
        end
    end
    
	dwX_train= X_ttl(1:Ntrain, :);
	dwX_test= X_ttl(Ntrain+1:end, :);
	
	glmnet_obj = glmnet(dwX_train, Y_train, 'gaussian');
	coeff.beta = glmnet_obj.beta(:, 52);
	coeff.alpha = glmnet_obj.a0(52);
	pred_train_price = dwX_train*coeff.beta + coeff.alpha;
	residual = Y_train - pred_train_price;
	
	pred_price = dwX_test*coeff.beta + coeff.alpha;
	
	if on == 1
		[U,D,~] = svdsecon([dwX_train; dwX_test],pc);
		X_trunc = U*D;
		Xtrain_trunc = X_trunc(1:Ntrain, :);
		Xtest_trunc = X_trunc(Ntrain+1:end, :);
		[nnidx,dist] = knnsearch(Xtrain_trunc, Xtest_trunc, 'K', K);
		knn_pred_residual = 0;
        kw = exp(-dist.^2/sigma^2);
        nl = sum(kw, 2);
		for i = 1:size(nnidx, 2)
			knn_pred_residual = knn_pred_residual + kw(:,i) .* residual(nnidx(:,i));
		end
		knn_pred_residual = knn_pred_residual./nl;
        %%%newadd%%
%         m = mean(residual);
%         c = cov(residual);
%         w = 1-normpdf(knn_pred_residual, m, c);
        %%%newadd%%
		Y_pred = pred_price + knn_pred_residual;
	else
		Y_pred = pred_price;
	end
end
