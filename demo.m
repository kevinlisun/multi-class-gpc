clear all
close all
warning off
clc

load('data.mat');

%% training GP
Dim = size(Xtrain,2);
kernel = @covSEiso;
para.kernel = kernel;
para.hyp = log([ones(1,1)*2, 10]);
para.S = 1e4;
para.c = nClass;
para.Ncore = 12;
para.flag = true; 
hyp = para.hyp;
gp_para = para;

% hyper-parameter optimization
[ hyp ] = modelSelection(para, Xtrain, ytrain);

% estimate the posterior probility of p(f|X,Y)
[ K ] = covMultiClass(hyp, para, Xtrain, []);
[ gp_model ] = LaplaceApproximation(hyp, para, K, Xtrain, ytrain);
save('classifier_gp_demo.mat','gp_model','gp_para');

[ ypredict prob fm ] = predictGPC_classic(hyp, para, Xtrain, ytrain, gp_model, Xtest);

if para.flag
    scrsz = get(0,'ScreenSize');
    fig3 = figure(3);
    set(fig3, 'name', 'The confusion matrix', 'Position',[1000 scrsz(4) 500 400]);
    c = para.c;
    [ ytest ] = label2binary(ytest, c, 'mat');
    [ ypredict ] = label2binary(ypredict, c, 'mat');
    plotconfusion(ytest', ypredict');
end
