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
X_test =[city_test word_test bigram_test];
Y_train = price_train;
% n-fold crossvalidation for dw-svd-knn-glment
nfold = 15;
%%%fixed paras%%%
K = 19;
on = 1;
idf = 1;
%%%%%%%%%%%%%%%%%
sigma = 8:20;
pc = 15:33;
nfold_rmse = zeros(length(sigma), length(pc));
for i = 1:length(sigma)
    for j = 1:length(pc)
        nfold_rmse(i,j) = cv_idf_svd_knn_glment(X_train, Y_train, nfold, pc(j), K, on, idf, sigma(i));
    end
end
save nfold_rmse n_fold_rmse;