function [w_obs_new] = update_weights(y_true,final_pred)
% Update observation weights after a boosting iteration
% y_true: actual target value
% final_pred: predicted value by ensemble model at iter t-1

w_obs_new = exp(-y_true .* final_pred);
w_obs_new = w_obs_new ./ sum(w_obs_new);  % normalize to 1

end