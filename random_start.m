syms x y;

functions = {
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % Easom Function
};

Names = {
    'Easom Function';
};

% Parameters for the trust_region function
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-3; % Tolerance for convergence
deg = 2; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method
num_trials = 50; % Number of random trials

% Arrays to store results
tay_steps = [];
tay_values = [];
pad_steps = [];
pad_values = [];

% Loop through each function and apply the trust_region method
for i = 1:length(functions)
    expr = functions{i};
    
    for trial = 1:num_trials
        disp(['Current Trial: ', num2str(trial)]);
    

        % Random initial points
        x0 = -5 + 10*rand();
        y0 = -5 + 10*rand();
        
        % Perform the trust region method with Taylor approximation
        method = 'tay';
        [result_tay, history_tay] = trust_region_2d(expr, x, y, x0, y0, r0, deg, maxIter, tol, method, rf, eta);
        
        % Perform the trust region method with Padé approximation
        method = 'pad';
        [result_pad, history_pad] = trust_region_nd(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
        
        % Store the results
        tay_steps = [tay_steps; size(history_tay, 1)];
        tay_values = [tay_values; double(subs(expr, {x, y}, {double(result_tay(1)), double(result_tay(2))}))];
        
        pad_steps = [pad_steps; size(history_pad, 1)];
        pad_values = [pad_values; double(subs(expr, {x, y}, {double(result_pad(1)), double(result_pad(2))}))];
    end
end

% Plot box plots for the number of steps
figure;
h = boxplot([tay_steps, pad_steps], 'Labels', {'Taylor Steps', 'Padé Steps'});
set(h, 'LineWidth', 2);
title('Box Plot of Number of Steps to Convergence');
ylabel('Number of Steps');

% Plot box plots for the function values
figure;
h = boxplot([tay_values, pad_values], 'Labels', {'Taylor Values', 'Padé Values'});
set(h, 'LineWidth', 2);
title('Box Plot of Function Values at Convergence');
ylabel('Function Value');

% Calculate mean values
mean_tay_steps = mean(tay_steps);
mean_pad_steps = mean(pad_steps);
mean_tay_values = mean(tay_values);
mean_pad_values = mean(pad_values);

% Calculate standard errors
se_tay_steps = std(tay_steps) / sqrt(num_trials);
se_pad_steps = std(pad_steps) / sqrt(num_trials);
se_tay_values = std(tay_values) / sqrt(num_trials);
se_pad_values = std(pad_values) / sqrt(num_trials);

% Plot mean number of steps
figure;
errorbar([1, 2], [mean_tay_steps, mean_pad_steps], [se_tay_steps, se_pad_steps], 'o-', 'LineWidth', 2);
xticks([1, 2]);
xticklabels({'Taylor Steps', 'Padé Steps'});
title('Mean Number of Steps to Convergence');
ylabel('Mean Number of Steps');
grid on;

% Plot mean function values
figure;
errorbar([1, 2], [mean_tay_values, mean_pad_values], [se_tay_values, se_pad_values], 'o-', 'LineWidth', 2);
xticks([1, 2]);
xticklabels({'Taylor Values', 'Padé Values'});
title('Mean Function Values at Convergence');
ylabel('Mean Function Value');
grid on;