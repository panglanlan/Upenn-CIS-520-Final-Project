clear;
load ../data/city_train.mat
load ../data/city_test.mat
load ../data/word_train.mat
load ../data/word_test.mat
load ../data/bigram_train.mat
load ../data/bigram_test.mat
load ../data/price_train.mat
tic
X_train =[city_train word_train bigram_train];
Y_train = price_train;
glmnet_obj = glmnet(X_train, Y_train, 'gaussian');
coeff.beta = glmnet_obj.beta(:, 52);
coeff.alpha = glmnet_obj.a0(52);
predict = X_train*coeff.beta + coeff.alpha;
sqrt(mean((predict-Y_train).^2))
toc