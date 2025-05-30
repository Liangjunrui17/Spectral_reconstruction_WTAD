%**************Parameter settings*************
step = 0.02;                 % Wavelength channel spacing in nm
MinLambda = 1525;            % The smallest wavelength in nm
MaxLambda = 1565;            % The largest wavelength in nm
lambda = MinLambda:step:MaxLambda;    % Operating wavelength range

%**************Loading data*************
load I.mat    % Observed intensity
load S_Reference.mat    % Reference spectrum

load T1.mat    % Calibrated transmission matrix
load T2.mat
load T3.mat
load T4.mat

%**************Reconstruction*************
WTM = 0*T1+0.5*T2+0.5*T3+0*T4;    % Weighted transmission matrix

S = Tikhonov(WTM,I);    % Initial estimation by Tikhonov regularization

S = dlarray(S);    % Support automatic differentiation
lossFunc = @(S) sum((WTM * S - I).^2, 'all');      % Loss function constructed by high-quality data fidelity term

learningRate = 1e-6;    % Initial learning rate
numIterations = 1e6;    % Maximum number of iterations 
patience = 50;          % Patience
min_delta = 1e-5;       % Minimum change in loss
best_loss = Inf;        % Best loss
wait = 0;               % Current patience
best_S = S;             % Best S

%**************Optimized by automatic differentiation
for iter = 1:numIterations
    [L, gradients] = dlfeval(@(S) computeGradients(lossFunc, S), S);   % Calculating gradient and loss
    learningRate = learningRate*(1-1e-9)^iter;    % Decay the learning rate    
    S = max(0,(S - learningRate * gradients));    % Update S

    % Early stop
    current_loss = extractdata(L);
    if current_loss < best_loss - min_delta
        best_loss = current_loss;
        best_S = S;   
        wait = 0;     
    else
        wait = wait + 1; 
    end
    
    if wait >= patience
        fprintf('Early stopping at iteration %d\n', iter);
        break;
    end

    % Print loss
    if mod(iter, 10) == 0
        fprintf('Iteration %d, Loss: %.4f\n', iter, extractdata(L));
    end
end

S_optimized = extractdata(S);

figure;    % Visualization
plot(lambda,S_optimized);hold on
plot(lambda,spe);


%**************evaluation metrics*************
% Calculating L2_error
L2_error = (norm(spe-S_optimized'))/((norm(spe)));    
fprintf('L2_error: %.4f \n', L2_error);

% Calculating PSNR
N = length(S_optimized); 
MSE = sum((S_optimized' - spe).^2) / N;
MAX = max(max(S_optimized), max(spe)); 
PSNR = 10 * log10(MAX^2 / MSE);
fprintf('PSNR: %.4f dB\n', PSNR);


%**************Gradient calculation function*************
function [L, gradients] = computeGradients(lossFunc, S)
    L = lossFunc(S);     
    gradients = dlgradient(L, S); 
end

%**************GCV-regulated Tikhonov regularization*************
function [m_optimal,cv_mse] = Tikhonov(x,y)
    gamma = [1:5:50];    % Regularization parameter space
    
    % Tikhonov regularization
    for i = 1:length(gamma)
        m_tikhonov(:,i) = max(0,(x'*x + gamma(i)*eye(size(x,2))) \ x' * y);   % 矩阵A\B 等效：A（-1）*B
    end 
    
    % Cross-validation to find optimal regularization parameter
    k = 10;    % k-fold cross validation
    cv_mse = zeros(length(gamma),1);    % Initialize mean squared error vector
    cv_indices = crossvalind('Kfold', length(y), k); 
    for i = 1:length(gamma)
        mse = zeros(k,1);     % Initialize MSE for each fold
        for j = 1:k
            test_indices = (cv_indices == j);     % Select current fold as test set
                    train_indices = ~test_indices;     % Other folds as training set
        x_train = x(train_indices,:);     % Training set features
        y_train = y(train_indices,:);     % Training set labels
        x_test = x(test_indices,:);     % Test set features
        y_test = y(test_indices,:);     % Test set labels

        % Solve for m on training set with regularization
        m = (x_train'*x_train + gamma(i)*eye(size(x_train,2))) \ x_train' * y_train;
        m = max(0,m);     % Apply non-negativity constraint
        y_pred = x_test * m;     % Predict on test set
        mse(j) = mean((y_test - y_pred).^2);     % Calculate MSE
    end
    cv_mse(i,1) = mean(mse);     % Compute cross-validation error
    end

    % Find the regularization parameter with minimum CV error
    [min_cv_mse, min_cv_mse_index] = min(cv_mse);
    disp('Optimal regularization parameter:');
    disp(gamma(min_cv_mse_index));

    % Optimal spectral vector with optimal regularization parameter
    m_optimal = m_tikhonov(:,min_cv_mse_index);
end