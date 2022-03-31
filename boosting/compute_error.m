function [error] = compute_error(y_true, y_pred, w_obs)
% Calculate the error rate of a weak classifier t
% y_true: actual target value
% y_pred: predicted value by weak classifier
% w_obs: observation weights

error = sum(w_obs(y_true ~= y_pred));

end