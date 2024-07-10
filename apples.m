syms x y;

functions = {
    exp(x+y);                                       % 0. Exponential Function
    x^2 + y^2;                                      % 1. Quadratic Function
    % (x - 1)^2 + (y - 2)^2 + 1;                      % 2. Quadratic Function with Offset
    sin(x) + cos(y);                                % 3. Trigonometric Function
    exp(-x^2 - y^2);                                % 4. Gaussian Function
    4 - 16*x^2 + y^4 - 16*y^2 + 64;               % 5. Fourth-Order Function
    % x^2 - 2*x*y + 2*y^2 + 2*x - 4*y + 4;            % 6. Quadratic with Linear Terms
    (x^2 + y - 11)^2 + (x + y^2 - 7)^2;             % 7. Himmelblau's Function
    0.26*(x^2 + y^2) - 0.48*x*y;                    % 8. Beale's Function
    sin(x) * cos(y);                                % 9. Sine-Cosine Function
    x^2 * y^2;                                      % 10. Product Function
    (x^2 + y^2)/4000 - (cos(x) + 1)*(cos(y)/sqrt(2) + 1); % 11. Griewank Function
    0.26*(x^2 + y^2) - 0.48*x*y;                    % 12. Matyas Function
    (1 - x)^2 + 100*(y - x^2)^2;                    % 13. Rosenbrock Function
    sin(x + y) + (x - y)^2 - 1.5*x + 2.5*y + 1;     % 14. McCormick Function
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 15. Easom Function
    (x + 2*y - 7)^2 + (2*x + y - 5)^2;              % 16. Booth Function
    (1 + (x + y + 1)^2*(19 - 14*x + 3*x^2 - 14*y + 6*x*y + 3*y^2)) * (30 + (2*x - 3*y)^2*(18 - 32*x + 12*x^2 + 48*y - 36*x*y + 27*y^2)); % 17. Goldstein-Price Function
    2*x^2 - 1.05*x^4 + (x^6)/6 + x*y + y^2; % 18 Three Hump Camel
    (4 - 2.1*x^2 + (x^4)/3)*x^2 + x*y + (-4 + 4*y^2)*y^2; % 19. Six Hump Camel Function
    (y - (5.1/(4*pi^2))*x + 5/pi * x - 6)^2 + 10*(1 - 1/(8*pi))*cos(x) + 10; % 20. Branin Function
    sin(3*pi*x)^2 + (x-1)^2*(1 + sin(3*pi*y)^2) + (y-1)^2*(1 + sin(2*pi*y)^2); % 21. Levy Function
    };


Names = {
    'Exponential Function'
    'Quadratic Function';
    % 'Quadratic Function with Offset';
    'Trigonometric Function';
    'Gaussian Function';
    'Fourth-Order Function';
    % 'Quadratic with Linear Terms';
    'Himmelblau''s Function';
    'Beale''s Function';
    'Sine-Cosine Function';
    'Product Function';
    'Griewank Function';
    'Matyas Function';
    'Rosenbrock Function';
    'McCormick Function';
    'Easom Function';
    'Booth Function';
    'Goldstein-Price Function';
    'Three Hump Camel';
    'Six Hump Camel';
    'Branin Function';
    'Levy Function';
};

initial_loc = {
    0, 0; % 0
    0, 0; % 1
    % 1, 2; % 2
    pi/2, pi; % 3
    0, 0; % 4
    4, 4; % 5
    % 2, 2; % 6
    0, 0; % 7
    0, pi; % 8
    0, 0; % 9
    0, 0; % 10
    0, 0; % 11
    0, 0; % 12
    1, 1; % 13
    0, 0; % 14
    pi, pi; % 15
    1, 3; % 16
    0, 0; % 17
    0, 0; % 18
    0, 0;
    2.5, 7.5;
    0, 0;
};

rad1 = {
    5; % 0
    5; % 1
    % 5; % 2
    5; % 3
    5; % 4
    5; % 5
    % 5; % 6
    5; % 7
    5; % 8
    5; % 9
    5; % 10
    5; % 11
    5; % 12
    5; % 13
    5; % 14
    5; % 15
    5; % 16
    5; % 17
    5;
    3; % 18
    7.5; % 19
    10 % 20
};

rad2 = {
    5; % 0
    5; % 1
    % 5; % 2
    5; % 3
    5; % 4
    5; % 5
    % 5; % 6
    5; % 7
    5; % 8
    5; % 9
    5; % 10
    5; % 11
    5; % 12
    5; % 13
    5; % 14
    5; % 15
    5; % 16
    5; % 17
    5;
    2; % 18
    7.5; % 19
    10 % 20
};



% Parameters for the trust_region function
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-3; % Tolerance for convergence
deg = 2; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method
num_trials = 2; % Number of random trials

% Arrays to store results

method = 'tay';
% Loop through each function and apply the trust_region method
for i = 1:length(functions)
    tay_steps = [];
    tay_values = [];
    tay_times = [];
    pad_steps = [];
    pad_values = [];
    pad_times = [];

    expr = functions{i};
    disp(['Current Test: ', Names{i}]);
    for trial = 1:num_trials
        disp(['- Current Trial: ', num2str(trial)]);
    
        % Random initial points
        %initial_loc{i,1}
        x0 = -rad1{i} + 2*rad1{i}*rand() + double(initial_loc{i, 1});
        y0 = -rad2{i} + 2*rad2{i}*rand() + double(initial_loc{i, 2});
        
        % Perform the trust region method with Taylor approximation

        % Perform the trust region method with Taylor approximation
        tic;
        [result_tay, history_tay] = trust_region_taylor(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
        tay_times = [tay_times; toc];
        
        % Perform the trust region method with Padé approximation
        tic;
        [result_pad, history_pad] = trust_region_pade(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
        pad_times = [pad_times; toc];

        
        % Store the results
        tay_steps = [tay_steps; size(history_tay, 1)];
        tay_values = [tay_values; double(subs(expr, {x, y}, {double(result_tay(1)), double(result_tay(2))}))];
        
        pad_steps = [pad_steps; size(history_pad, 1)];
        pad_values = [pad_values; double(subs(expr, {x, y}, {double(result_pad(1)), double(result_pad(2))}))];
        
    end
    save(['adsfasd/', Names{i}, '.mat'], 'pad_steps', 'pad_values', 'pad_times', 'tay_steps', 'tay_values', 'tay_times');
    % Calculate mean values


    boxplotdata(tay_steps, pad_steps, tay_values, pad_values, tay_times, pad_times, Names{i}, num_trials);

    
    %waitfor(gcf);
end