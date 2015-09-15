function [ F f ] = objectivefunc(hyp)

    global logML;
    global marker;
    
    load('parameters.mat');
    
    [ K ] = covMultiClass(hyp, para, X, []);
    [ model ] = LaplaceApproximation(hyp, para, K, X, y);
    
    if isempty(model)
        F = 1e3;
        f = zeros(size(hyp));
        return;
    end
    
    F = logMarginalLikelihood(hyp, para, model);
    
    logML = [ logML, F ];
    
    figure(1);
    subplot(1,3,1);
    title('hyper-paramethers');
    plot(1:length(hyp), exp(hyp), 'xb');
    
    figure(1)
    subplot(1,3,2); 
    title('log marignal likelihood');
    hold on;
    if length(logML) > 1
        plot(length(logML)-1:length(logML), -logML(end-1:end), marker);
    else
        plot(1:length(logML), -logML, marker);
    end

    
    % caculate gridient
    if nargout > 1
        f = logMarginalLikelihood(hyp, para, model, []);
        figure(1);
        subplot(1,3,3);
        title('gradient')
        plot(1:length(f), f, 'r^');
        pause(0.01);
    end
    
