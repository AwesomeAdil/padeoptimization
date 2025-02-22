syms x y;

functions = {
    % exp(x+y);                                       % 0. Exponential Function
    % x^2 + y^2;                                      % 1. Quadratic Function
    % (x - 1)^2 + (y - 2)^2 + 1;                      % 2. Quadratic Function with Offset
    % sin(x) + cos(y);                                % 3. Trigonometric Function
    % exp(-x^2 - y^2);                                % 4. Gaussian Function
    % x^4 - 16*x^2 + y^4 - 16*y^2 + 64;               % 5. Fourth-Order Function
    % x^2 - 2*x*y + 2*y^2 + 2*x - 4*y + 4;            % 6. Quadratic with Linear Terms
    % (x^2 + y - 11)^2 + (x + y^2 - 7)^2;             % 7. Himmelblau's Function
    % 0.26*(x^2 + y^2) - 0.48*x*y;                    % 8. Beale's Function
    % sin(x) * cos(y);                                % 9. Sine-Cosine Function
    % x^2 * y^2;                                      % 10. Product Function
    % (x^2 + y^2)/4000 - (cos(x) + 1)*(cos(y)/sqrt(2) + 1); % 11. Griewank Function
    % 0.26*(x^2 + y^2) - 0.48*x*y;                    % 12. Matyas Function
    (1 - x)^2 + 100*(y - x^2)^2;                    % 13. Rosenbrock Function
    sin(x + y) + (x - y)^2 - 1.5*x + 2.5*y + 1;     % 14. McCormick Function
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 15. Easom Function
    (x + 2*y - 7)^2 + (2*x + y - 5)^2;              % 16. Booth Function
    (1 + (x + y + 1)^2*(19 - 14*x + 3*x^2 - 14*y + 6*x*y + 3*y^2)) * (30 + (2*x - 3*y)^2*(18 - 32*x + 12*x^2 + 48*y - 36*x*y + 27*y^2)) % 17. Goldstein-Price Function
};


Names = {
    % 'Exponential Function'
    % 'Quadratic Function';
    % 'Quadratic Function with Offset';
    % 'Trigonometric Function';
    % 'Gaussian Function';
    'Fourth-Order Function';
    % 'Quadratic with Linear Terms';
    % 'Himmelblau''s Function';
    % 'Beale''s Function';
    % 'Sine-Cosine Function';
    % 'Product Function';
    % 'Griewank Function';
    % 'Matyas Function';
    % 'Rosenbrock Function';
    % 'McCormick Function';
    % 'Easom Function';
    % 'Booth Function';
    % 'Goldstein-Price Function'
};

initial_loc = {
    % 0, 0; % 0
    % 0, 0; % 1
    % 1, 2; % 2
    % pi/2, pi; % 3
    % 0, 0; % 4
    4, 4; % 5
    % 2, 2; % 6
    % 0, 0; % 7
    % 0, pi; % 8
    % 0, 0; % 9
    % 0, 0; % 10
    % 0, 0; % 11
    % 0, 0; % 12
    % 1, 1; % 13
    % 0, 0; % 14
    % pi, pi; % 15
    % 1, 3; % 16
    % 0, 0; % 17
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
tay_times = [];
pad_steps = [];
pad_values = [];
pad_times = [];

method = 'tay';
% Loop through each function and apply the trust_region method
for i = 1:length(functions)
    tay_steps = [];
    tay_values = [];
    pad_steps = [];
    pad_values = [];
    tay_times = [];
    pad_times = [];
    expr = functions{i};
    disp(['Current Test: ', Names{i}]);
    for trial = 1:num_trials
        disp(['- Current Trial: ', num2str(trial)]);
    
        % Random initial points
        %initial_loc{i,1}
        x0 = -5 + 10*rand() + double(initial_loc{i, 1});
        y0 = -5 + 10*rand() + double(initial_loc{i, 2});
        
        % Perform the trust region method with Taylor approximation

        % Perform the trust region method with Taylor approximation
        tic;
        [result_tay, history_tay] = trust_region_2d(expr, x, y, x0, y0, r0, deg, maxIter, tol, 'tay', rf, eta);
        tay_times = [tay_times; toc];
        
        % Perform the trust region method with Padé approximation
        tic;
        [result_pad, history_pad] = trust_region_nd(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
        pad_times = [pad_times; toc];

        
        % Store the results
        tay_steps = [tay_steps; size(history_tay, 1)];
        tay_values = [tay_values; double(subs(expr, {x, y}, {double(result_tay(1)), double(result_tay(2))}))];
        
        pad_steps = [pad_steps; size(history_pad, 1)];
        pad_values = [pad_values; double(subs(expr, {x, y}, {double(result_pad(1)), double(result_pad(2))}))];
    end

    % Calculate mean values
    mean_tay_steps = mean(tay_steps);
    mean_pad_steps = mean(pad_steps);
    mean_tay_values = mean(tay_values);
    mean_pad_values = mean(pad_values);
    mean_tay_times = mean(tay_times);
    mean_pad_times = mean(pad_times);
    
    % Calculate standard errors
    se_tay_steps = std(tay_steps) / sqrt(num_trials);
    se_pad_steps = std(pad_steps) / sqrt(num_trials);
    se_tay_values = std(tay_values) / sqrt(num_trials);
    se_pad_values = std(pad_values) / sqrt(num_trials);
    se_tay_times = std(tay_times) / sqrt(num_trials);
    se_pad_times = std(pad_times) / sqrt(num_trials);

    % Create a figure and set its properties
    fig = figure;
    set(fig, 'Name', Names{i}, 'NumberTitle', 'off', 'Visible', 'on');

    % Plot box plots for the number of steps
    subplot(3, 2, 1);
    h = boxplot([tay_steps, pad_steps], 'Labels', {'Taylor Steps', 'Padé Steps'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Number of Steps to Convergence');
    ylabel('Number of Steps');
    
    % Plot box plots for the function values
    subplot(3, 2, 2);
    h = boxplot([tay_values, pad_values], 'Labels', {'Taylor Values', 'Padé Values'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Function Values at Convergence');
    ylabel('Function Value');
    
    % Plot box plots for the running times
    subplot(3, 2, 3);
    h = boxplot([tay_times, pad_times], 'Labels', {'Taylor Times', 'Padé Times'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Running Times');
    ylabel('Time (s)');
    
    % Plot mean number of steps
    subplot(3, 2, 4);
    errorbar([1, 2], [mean_tay_steps, mean_pad_steps], [se_tay_steps, se_pad_steps], 'o-', 'LineWidth', 2);
    xticks([1, 2]);
    xticklabels({'Taylor Steps', 'Padé Steps'});
    title('Mean Number of Steps to Convergence');
    ylabel('Mean Number of Steps');
    grid on;
    
    % Plot mean function values
    subplot(3, 2, 5);
    errorbar([1, 2], [mean_tay_values, mean_pad_values], [se_tay_values, se_pad_values], 'o-', 'LineWidth', 2);
    xticks([1, 2]);
    xticklabels({'Taylor Values', 'Padé Values'});
    title('Mean Function Values at Convergence');
    ylabel('Mean Function Value');
    grid on;
    
    % Plot mean running times
    subplot(3, 2, 6);
    errorbar([1, 2], [mean_tay_times, mean_pad_times], [se_tay_times, se_pad_times], 'o-', 'LineWidth', 2);
    xticks([1, 2]);
    xticklabels({'Taylor Times', 'Padé Times'});
    title('Mean Running Times');
    ylabel('Mean Time (s)');
    grid on;
    
    % Adjust layout
    sgtitle(['Convergence Analysis: ', Names{i}]);
    savefig(['Convergence Analysis/', Names{i},'2.fig']);
    %waitfor(gcf);
end