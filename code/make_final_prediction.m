function prediction = make_final_prediction(model,X_test,X_text)

% Input
% X_test : a 1xp vector representing "1" test sample.
% X_test=[city word bigram] a 1-by-10007 vector
% model : what you initialized from init_model.m
%
% Output
% prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.
K = 19;
sigma = 6;

city_idx = find(X_test(1:7));
[idx, dist] = knnsearch(model.knn.x_svd_ttl,(model.lasso.info_ttl.*X_test)*model.knn.x_svd_ttl_loading,'K',K);
kw = exp(-dist.^2/sigma^2);
ttl_pred = model.lasso.alpha_ttl+(model.lasso.info_ttl.*X_test)*model.lasso.beta_ttl+kw*model.knn.residual_ttl(idx)/sum(kw);

[idx, dist] = knnsearch(model.knn.x_svd_city{city_idx},(model.lasso.info_city(:,city_idx)'.*X_test)*model.knn.x_svd_city_loading{city_idx},'K',K);
kw = exp(-dist.^2/sigma^2);
city_pred = model.lasso.alpha_city(city_idx)+(model.lasso.info_city(:,city_idx)'.*X_test)*model.lasso.beta_city(:,city_idx)+kw*model.knn.residual_city{city_idx}(idx)/sum(kw);

prediction=(city_pred+ttl_pred)/2;
