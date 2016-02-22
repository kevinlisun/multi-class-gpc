function [ y prob fm ] = predictGPC(hyp, para, X, Y, model, x)

% Copyright (c) University of Glasgow in UK  - All Rights Reserved
% Author: Li Sun (Kevin) <lisunsir@gmail.com>
% Institute: University of Glasgow
% Details: Prediction for Multiple Class GP classification p(y*|X,Y,x*)
% Reference: 
% 1. <Gaussian Process for Machine Learning> 
% 2. <Recognising the Clothing Categories from Free-Configuration using Gaussian-Process-Based Interactive Perception>


K = model.K;
W = model.W;
f = model.f;
Pi = model.Pi;
TT = model.TT;

n = length(Y);
c = para.c;
kernel = para.kernel;

[ Ybin ] = label2binary(Y);

Ncore = para.Ncore;

for j = 1:Ncore
    batch_list{j} = j:Ncore:size(x,1);
end

Y = cell(1,1,Ncore);
FM = cell(1,1,Ncore);
PROB = cell(1,1,Ncore);

for j = 1:Ncore
    
    tmp_list = batch_list{j};
    y = zeros(size(x,1), 1);
    fm = zeros(size(x,1), c);
    prob = zeros(size(x,1), c);
    
    for i = 1:length(batch_list{j})
        
        Mu = zeros(1,c);
        Cov = zeros(c,c);
        
        disp(['predicting the ', num2str(tmp_list(i)),'th of ', num2str(size(x,1)), ' testing example ...']);
        index = repmat(1:c, [n 1]);
        
        for ci = 1:c
            xi = x(tmp_list(i),:);
            kc = feval(kernel, hyp, X, xi);
            %kc = computeCov(X, xi, para);
            yc = Ybin(index == ci);
            pic = Pi(index == ci);
            
            Mu(ci) = kc' * (yc - pic);
        end
        
        Q = covMultiClass(hyp, para, X, xi);
        
        Cov = eye(c) * feval(kernel, hyp, xi, xi) - Q' * inv(K + inv(W+10e-8*eye(size(W)))) * Q;
        %%-----------------fast------------------%%
        % %     D = diag(Pi);
        % %     R = inv(D) * TT;
        % %     E = D^(0.5) * inv(eye(size(K)) + D^(0.5)*K*D^(0.5)) *  D^(0.5);
        % %     KWinv_inv = E - E * R * inv(R'*E*R) * R' * E;
        % %     Cov = eye(c) * computeCov(xi, xi, para) - Q' * KWinv_inv * Q;
        %------------------------------------------
        
        % get the probility estimation from sampling
        S = para.S;
        fs = mgd(S, c, Mu, Cov);
        fm(tmp_list(i),:) = Mu;
        
        fs = exp(fs);
        p = fs ./ repmat(sum(fs,2),[1,c]);
        p = sum(p,1) / S;
        prob(tmp_list(i),:) = p;
        
        [ tmp y(tmp_list(i)) ] = max(p);
    end
    Y{j} = y;
    FM{j} = fm;
    PROB{j} = prob;
end

y = sum(cell2mat(Y),3);
fm = sum(cell2mat(FM),3);
prob = sum(cell2mat(PROB),3);

%% predictive confidence (largest predictive probablity)
max_prob = max(prob, [], 2);



