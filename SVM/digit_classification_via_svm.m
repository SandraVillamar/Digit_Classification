% Classify Digits Via SVM

%% ---------- Loading Data ----------

% add path to access readMNIST fn
addpath('../data')

% get data
[imgs_train, labels_train] = readMNIST( ...
    '../data/training set/train-images.idx3-ubyte', ...
    '../data/training set/train-labels.idx1-ubyte', 20000, 0);
[imgs_test, labels_test] = readMNIST( ...
    '../data/test set/t10k-images.idx3-ubyte', ...
    '../data/test set/t10k-labels.idx1-ubyte', 10000, 0);
X_train = imgs_train;
X_test = imgs_test;

% variables
n = size(X_train,1);  % num of obs in train data
n_test = size(X_test,1); % num of obs in test data
d = size(X_train,2);  % dim of data
digits = 1:10;  % 10 represents 0 digit
n_digits = length(digits);

%% ---------- PROBLEM AB ----------

C = [2,4,8];  % SVM parameter
% results: 3rd dim is test err, # of SVs, 3 SV with largest lagrange (+)
% and 3 SV with largest lagrange (-)
results = zeros(length(C),n_digits,8);  
margins = zeros(length(C),n_digits,n);
g = zeros(n_digits,n_test);  % store output per digit and sample for overall error
final_test_error = zeros(length(C));  % overall classification error per C


for C_i=1:length(C)
%for C_i=1
    
    for digit = digits
    %for digit=1
    
        % convert labels to {1,-1} if label=digit
        if digit~=10
            y_train = ones(size(labels_train));
            y_train(labels_train ~= digit) = -1;
            y_test = ones(size(labels_test));
            y_test(labels_test ~= digit) = -1;
        else
            % 10th index represents digit 0
            y_train = ones(size(labels_train));
            y_train(labels_train ~= 0) = -1;
            y_test = ones(size(labels_test));
            y_test(labels_test ~= 0) = -1;
        end
    
        % train SVM
        model_linear = svmtrain(y_train, X_train, sprintf('-s 0 -t 0 -c %i',C(C_i)));
    
        % predict
        [predict_label_L, accuracy_L, d_vals] = svmpredict(y_test, X_test, model_linear);
    
        % store variables
        results(C_i,digit,1) = 100 - accuracy_L(1);  % binary test error (%)
        results(C_i,digit,2) = sum(model_linear.nSV);  % # of SVs
        [~,max_ind] = maxk(model_linear.sv_coef,3);  % 3 SVs with largest lagrange (+)
        results(C_i,digit,3:5) = model_linear.sv_indices(max_ind);
        [~,min_ind] = mink(model_linear.sv_coef,3);  % 3 SVs with largest lagrange (-)
        results(C_i,digit,6:8) = model_linear.sv_indices(min_ind);

        % get w and b
        w = model_linear.SVs' * model_linear.sv_coef;
        b = -model_linear.rho;
        if model_linear.Label(1) == -1
          w = -w;
          b = -b;
        end
        
        margins(C_i,digit,:) = y_train .* (X_train*w + b);  % margin

        % store g(x) for overall error
        g(digit,:) = X_test*w + b;

    end

    % overall test error per C: argmax across all binary classifiers, per index
    [~,digit_pred] = max(g);
    digit_pred(digit_pred==10) = 0;  % adjust digit back to 0
    final_test_error(C_i) = sum(digit_pred' ~= labels_test) / n_test;

end

save('variables_linear')

%% Plot 3 SVs with largest Lagrange mult. on each side of the boundary

        for C_i=1:length(C)
            figure(C_i)
            for digit=digits
                for i=3:8
                    subplot(10,6,6*(digit-1)+i-2)
                    colormap(gray(255));
                    sample_ind = results(C_i,digit,i);
                    imagesc( reshape( X_train(sample_ind,:), 28, 28)' );
                end       
            end
        end

%% Plot CDFs

figure(4)
for digit=digits
    
    subplot(2,5,digit)
    
    % plot cdf per C
    for C_i=1:length(C)
        cdfplot(margins(C_i,digit,:));
        hold on
    end
    hold off

    % plot labels
    if digit ~= 10
        title(sprintf('Digit %i',digit))
    else
        title(sprintf('Digit %i',0))
    end
    xlabel('X')
    ylabel('P(margin <= X)')
    if digit==1
        legend('C=2','C=4','C=8','southeast')
    end
end

%% ---------- PROBLEM C ----------

% grid search to find best C and gamma (one vs one classification strategy)
bestcv = 0;
for log2c = -1:1
    for log2g = -4:-2
        cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
        cv = svmtrain(labels_train, X_train, cmd);
        if (cv >= bestcv)
            bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
        end
    fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
    end
end

% run SVM with radial basis fn:
results2 = zeros(n_digits,8);  
margins2 = zeros(n_digits,n);
g2 = zeros(n_digits,n_test);  % store output per digit and sample for overall error
    
for digit = digits

    % convert labels to {1,-1} if label=digit
    if digit~=10
        y_train = ones(size(labels_train));
        y_train(labels_train ~= digit) = -1;
        y_test = ones(size(labels_test));
        y_test(labels_test ~= digit) = -1;
    else
        % 10th index represents digit 0
        y_train = ones(size(labels_train));
        y_train(labels_train ~= 0) = -1;
        y_test = ones(size(labels_test));
        y_test(labels_test ~= 0) = -1;
    end

    % train SVM
    model_radial = svmtrain(y_train, X_train, sprintf('-c %g -g %g',bestc,bestg));

    % predict
    [predict_label_L, accuracy_L, d_vals] = svmpredict(y_test, X_test, model_radial);

    % store variables
    results2(digit,1) = 100 - accuracy_L(1);  % binary test error (%)
    results2(digit,2) = sum(model_radial.nSV);  % # of SVs
    [~,max_ind] = maxk(model_radial.sv_coef,3);  % 3 SVs with largest lagrange (+)
    results2(digit,3:5) = model_radial.sv_indices(max_ind);
    [~,min_ind] = mink(model_radial.sv_coef,3);  % 3 SVs with largest lagrange (-)
    results2(digit,6:8) = model_radial.sv_indices(min_ind);

    % get w and b
    w2 = model_radial.SVs' * model_radial.sv_coef;
    b2 = -model_radial.rho;
    if model_radial.Label(1) == -1
      w2 = -w2;
      b2 = -b2;
    end
    
    margins2(digit,:) = y_train .* (X_train*w2 + b2);  % margin

    % store g(x) for overall error
    g2(digit,:) = X_test*w2 + b2;

end

% overall test error: argmax across all binary classifiers, per index
[~,digit_pred] = max(g2);
digit_pred(digit_pred==10) = 0;  % adjust digit back to 0
final_test_error = sum(digit_pred' ~= labels_test) / n_test;


save('variables_radial')

%% Plot 3 SVs with largest Lagrange mult. on each side of the boundary

figure(5) 
for digit=digits
    for i=3:8
        subplot(10,6,6*(digit-1)+i-2)
        colormap(gray(255));
        sample_ind = results2(digit,i);
        imagesc( reshape( X_train(sample_ind,:), 28, 28)' );
    end       
end

%% Plot CDFs

figure(6)
for digit=digits
    
    subplot(2,5,digit)
    
    % plot cdf
    cdfplot(margins2(digit,:));

    % plot labels
    if digit ~= 10
        title(sprintf('Digit %i',digit))
    else
        title(sprintf('Digit %i',0))
    end
    xlabel('X')
    ylabel('P(margin <= X)')
    
end
