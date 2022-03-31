function [w_c] = compute_w_c(error)
% Calculate the weight of the weak classifier t in the majority vote of the
% final classifier
% error: error rate from weak classifier t

w_c = .5 * log( (1-error)/error );

end