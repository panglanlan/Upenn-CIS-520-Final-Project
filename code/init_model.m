function model = init_model()

load('model.mat');
model.lasso = lasso_coeff;
model.knn = knn_model;

% Example:
% tmp = load('magic.mat');
% model.regW = tmp.w;


