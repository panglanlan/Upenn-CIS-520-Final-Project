function nfold_rmse = cv_idf_svd_knn_glment(X_train, Y_train, nfold, pc, K, on, idf, sigma)
	%%%%
	% n-fold crossvalidation for svd-knn-glment
%     (X_train, Y_train, X_test, pc, K, on, IDF, sigma)
	indices = crossvalind('Kfold', size(X_train,1), nfold);
	mse = 0;
	for i = 1:nfold
		test = (indices == i); train = ~test;
		Y_pred = idf_svd_knn_glmnet(X_train(train,:), Y_train(train,:), ...
                                X_train(test, :), pc, K, on, idf, sigma);
		mse = mse + (mean((Y_pred - Y_train(test,:)).^2));
	end
	nfold_rmse = sqrt(mse/nfold);
end