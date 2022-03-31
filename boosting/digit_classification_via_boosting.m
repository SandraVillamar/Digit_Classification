% Classify Digits Via Boosting

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

%% Create binary classifiers: digits 0-9

% variables for all digit classifiers
K = 250;  % iterations
x_largest_w_obs = zeros(K,n_digits);  % index of sample with largest w_obs per iteration
final_pred = zeros(n,n_digits);  % final pred g(x) per binary classifier
final_pred_test = zeros(n_test,n_digits);
error_train = zeros(K,n_digits);  % error per iter per bin classifier
error_test = zeros(K,n_digits);
margins = zeros(5,n,n_digits);  % store margin per train sample at iter: [5,10,50,100,250]
a = 128*ones(d,n_digits); % array for part D

% for each digit i.e. binary classifier
for digit=digits

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

    % initialize
    w_c = zeros(K,1);  % weights for weak learners (classifiers)
    w_obs = ones(n,1)/n;  % observation weights; at t=1 weights are all equal
    margin_cnt = 1;

    % find K weak learners to obtain boosting model
    for t=1:K

        fprintf('digit: %i, iter: %i\n', digit, t);

        if t==1
            % fit first weak learner and add to models
            models = decision_stump(X_train, w_obs, y_train);
            % store weak learner behavior
            if strcmp(models.pos_neg,'pos')
                a(models.dim,digit) = 255;
            else
                a(models.dim,digit) = 0;
            end
            % predict labels
            y_pred = decision_stump_prediction(models, X_train);
            y_pred_test = decision_stump_prediction(models, X_test);
        else
            % compute observation weights
            w_obs = update_weights(y_train,final_pred(:,digit));
            % fit weak learner i.e. find optimal u(x;j,t)
            models(t) = decision_stump(X_train, w_obs, y_train);
            % store weak learner behavior
            if strcmp(models(t).pos_neg,'pos')
                a(models(t).dim,digit) = 255;
            else
                a(models(t).dim,digit) = 0;
            end
            % predict labels
            y_pred = decision_stump_prediction(models(t), X_train);
            y_pred_test = decision_stump_prediction(models(t), X_test);
        end
    
        % compute error of classifier t
        error_c = compute_error(y_train, y_pred, w_obs);
        % compute weak learner weight
        w_c(t) = compute_w_c(error_c);
    
        % add to final prediction
        final_pred(:,digit) = final_pred(:,digit) + w_c(t)*y_pred;
        final_pred_test(:,digit) = final_pred_test(:,digit) + w_c(t)*y_pred_test;
    
        % store error of model at iter t
        curr_iter_pred = sign(final_pred(:,digit));
        error_train(t,digit) = sum(curr_iter_pred ~= y_train) / n;
        curr_iter_pred_test = sign(final_pred_test(:,digit));
        error_test(t,digit) = sum(curr_iter_pred_test ~= y_test) / n_test;
    
        % store index of sample with largest w_obs
        [~,x_largest_w_obs(t,digit)] = max(w_obs);
    
        % store margin
        if ismember(t,[5,10,50,100,250])
            margins(margin_cnt,:,digit) = y_train .* final_pred(:,digit);
            margin_cnt = margin_cnt + 1;
        end  
    end
end

%% test error of final classifier

% argmax of final_pred_test (across all binary classifiers, per index)
[~,digit_pred] = max(final_pred_test,[], 2);
digit_pred(digit_pred==10) = 0;  % adjust digit back to 0
error_test_final = sum(digit_pred ~= labels_test) / n_test;

%% plot train and test errors per binary classifier
figure(1)
for digit=digits
    subplot(2,5,digit)
    plot(error_train(:,digit),'r')  % training error
    hold on
    plot(error_test(:,digit), 'b')  % testing error
    
    % plots labels
    if digit ~= 10
        title(sprintf('Binary Classifier: %i',digit))
    else
        title(sprintf('Binary Classifier: %i',0))
    end
    xlabel('iteration')
    ylabel('error (%)')
    if digit==1
        legend('Training Error', 'Test Error')
    end
end

%% cdf of margins

figure(2)
for digit=digits
    
    subplot(2,5,digit)
    
    % plot cdf per stored iteration
    for i=1:size(margins,1)
        cdfplot(margins(i,:,digit));
        hold on
    end
    hold off

    % plot labels
    if digit ~= 10
        title(sprintf('Binary Classifier: %i',digit))
    else
        title(sprintf('Binary Classifier: %i',0))
    end
    xlabel('X')
    ylabel('P(margin <= X)')
    if digit==1
        legend('Iter 5', 'Iter 10', 'Iter 50', 'Iter 100', 'Iter 250', 'Location','southeast')
    end
end

%% sample index of largest w_obs per iter

figure(3)
for digit=digits
    subplot(2,5,digit)
    plot(x_largest_w_obs(:,digit),'.')

    % plot labels
    if digit ~= 10
        title(sprintf('Binary Classifier: %i',digit))
    else
        title(sprintf('Binary Classifier: %i',0))
    end
    xlabel('iteration')
    ylabel('sample index with largest weight')
end

%% plot 3 'heaviest' examples

figure(4)
for digit=digits
    [count,example] = groupcounts(x_largest_w_obs(:,digit));  % counts of examples
    [~,cnt_ind] = maxk(count,3);  % index of 3 highest counts
    heaviest_ex = example(cnt_ind);  % examples of 3 highest counts

    if digit~= 10
        plot(ones(3,1)*digit,heaviest_ex,'.','MarkerSize',10)
    else
        plot(zeros(3,1),heaviest_ex,'.','MarkerSize',10)
    end
    hold on

    % plot labels
    xlabel('digit')
    ylabel('sample index')
end
hold off
title('3 heaviest examples')

%% plot images of 3 'heaviest' examples

figure(5)
for digit=digits
    % get 3 heaviest examples
    [count,example] = groupcounts(x_largest_w_obs(:,digit));  % counts of examples
    [~,cnt_ind] = maxk(count,3);  % index of 3 highest counts
    heaviest_ex = example(cnt_ind);  % examples of 3 highest counts

    for i=1:3
        % plot corresponding image of 3 heaviest examples
        subplot(10,3,3*(digit-1)+i)
        colormap(gray(255));
        imagesc( reshape( X_train(heaviest_ex(i),:), 28, 28)' );
    end
end

%% plot array a

figure(6)

for digit=digits
    subplot(2,5,digit)
    colormap(gray(255));
    imagesc(reshape(a(:,digit),28,28)');

    % plot labels
    if digit ~= 10
        title(sprintf('Binary Classifier: %i',digit))
    else
        title(sprintf('Binary Classifier: %i',0))
    end

end


