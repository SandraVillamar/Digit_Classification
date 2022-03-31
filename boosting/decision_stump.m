function [model] = decision_stump(train_set, sample_weights, labels)
% Selects the optimal weak classifier (decision stump) by finding the best
% threshold, dimension, and polarity that divides the data into 2 classes
% with the smallest error.
%   train_set: an NxD-matrix, each row is a training sample in the D 
%   dimensional feature space
%   sample_weights: an Nx1-vector, each entry is the weight of the 
%   corresponding training sample
%   labels: Nx1 dimensional vector, each entry is the corresponding label 
%   (either -1 or 1)
%
%   model: the ouput model. It consists of
%       1) min_error: training error
%       2) min_error_thr: threshold value
%       3) pos_neg: whether up-direction shows the positive region (label:1, 'pos') or
%          the negative region (label:-1, 'neg')
%       4) dim: feature index

model = struct('min_error',[],'min_error_thr',[],'pos_neg',[],'dim',[]);
model.min_error = inf;
thresholds = (0:50) / 50;

% for each dimension
for dim=1:size(train_set,2)
	
	% for each threshold
	for thr_ind=1:length(thresholds)

        ind1 = train_set(:,dim) >= thresholds(thr_ind);  % xj >= t, class 1 
        ind1 = (ind1-0.5) * 2;  % convert to (1,-1)
		
        tmp_err = - sample_weights' * (labels.*ind1);  % error of model

		if(tmp_err < model.min_error)
			model.min_error = tmp_err;
			model.min_error_thr = thresholds(thr_ind);
			model.pos_neg = 'pos';
			model.dim = dim;
		end
		
        % check opposite direction
        ind1 = train_set(:,dim) < thresholds(thr_ind);  % xj < t, class 1
        ind1 = (ind1-0.5) * 2;  % convert to (1,-1)
      
        tmp_err = - sample_weights' * (labels.*ind1);  % error of model

        if(tmp_err < model.min_error)
            model.min_error = tmp_err;
            model.min_error_thr = thresholds(thr_ind);
            model.pos_neg = 'neg';
			model.dim = dim;
        end
	end
end