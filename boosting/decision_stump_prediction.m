function [y_pred] = decision_stump_prediction(model, X)
% Apply decision stump model from decision_stump.m to predict labels from X

if(strcmp(model.pos_neg, 'pos'))
    y_pred = X(:,model.dim) >= model.min_error_thr;
else
    y_pred = X(:,model.dim) < model.min_error_thr;
end

y_pred = (y_pred-0.5) * 2;  % convert to (1,-1)