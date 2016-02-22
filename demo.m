% Copyright (c) University of Glasgow in UK  - All Rights Reserved
% Author: Li Sun (Kevin) <lisunsir@gmail.com>
% Institute: University of Glasgow
% Details: This is a demo script of multi-calss Gaussian Process
% Classification, in which hyperparameters are optimized by maximization of
% marginal likelihood, posterior is estimated by Laplace Approximation
% Reference: 
% 1. <Gaussian Process for Machine Learning> 
% 2. <Recognising the Clothing Categories from Free-Configuration using Gaussian-Process-Based Interactive Perception>


clear all
close all
warning off
clc

load('data.mat');

%% training GP
Dim = size(Xtrain,2);
kernel = @covSEiso; % rbf kernel
para.kernel = kernel;
para.hyp = log([ones(1,1)*2, 10]); % initilization of hyper-parameters
para.S = 1e4; % sample number
para.c = nClass; % numble of categories
para.Ncore = 12; % multiple CPU cores parallelizing
para.flag = true; % plotting flag
hyp = para.hyp;
gp_para = para;

% hyper-parameter optimization
[ hyp ] = modelSelection(para, Xtrain, ytrain);

% compute multi-class GP kernel
[ K ] = covMultiClass(hyp, para, Xtrain, []);

% estimate the posterior probility of p(f|X,Y)
[ gp_model ] = LaplaceApproximation(hyp, para, K, Xtrain, ytrain);

% save GP parameters
save('classifier_gp_demo.mat','gp_model','gp_para');

% prediction p(y*|X,y,x*)
[ ypredict prob fm ] = predictGPC(hyp, para, Xtrain, ytrain, gp_model, Xtest);

if para.flag
    scrsz = get(0,'ScreenSize');
    fig3 = figure(3);
    set(fig3, 'name', 'The confusion matrix', 'Position',[1000 scrsz(4) 500 400]);
    c = para.c;
    [ ytest ] = label2binary(ytest, c, 'mat'); % convert label to binary form
    [ ypredict ] = label2binary(ypredict, c, 'mat'); % convert label to binary form
    plotconfusion(ytest', ypredict');
end
