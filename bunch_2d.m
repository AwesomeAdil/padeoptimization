syms x y;
%-((1 + cos(12*sqrt(x^2 + y^2)))/(0.5*(x^2 + y^2) + 2)); % 11. Drop Wave Function
    
    %-0.0001*(abs(sin(x)*sin(y)*exp(abs(100 - (sqrt(x^2 + y^2))/pi) + 1)))^0.1; % 19. Cross-in-Tray Function
functions = {
    exp(x+y);                                       % 0. Exponential Function
    x^2 + y^2;                                      % 1. Quadratic Function
    (x - 1)^2 + (y - 2)^2 + 1;                      % 2. Quadratic Function with Offset
    sin(x) + cos(y);                                % 3. Trigonometric Function
    exp(-x^2 - y^2);                                % 4. Gaussian Function
    x^4 - 16*x^2 + y^4 - 16*y^2 + 64;               % 5. Fourth-Order Function
    x^2 - 2*x*y + 2*y^2 + 2*x - 4*y + 4;            % 6. Quadratic with Linear Terms
    (x^2 + y - 11)^2 + (x + y^2 - 7)^2;             % 7. Himmelblau's Function
    0.26*(x^2 + y^2) - 0.48*x*y;                    % 8. Beale's Function
    sin(x) * cos(y);                                % 9. Sine-Cosine Function
    x^2 * y^2;                                      % 10. Product Function
    (x^2 + y^2)/4000 - (cos(x) + 1)*(cos(y)/sqrt(2) + 1); % 12. Griewank Function
    0.26*(x^2 + y^2) - 0.48*x*y;                    % 13. Matyas Function
    (1 - x)^2 + 100*(y - x^2)^2;                    % 14. Rosenbrock Function
    sin(x + y) + (x - y)^2 - 1.5*x + 2.5*y + 1;     % 15. McCormick Function
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 16. Easom Function
    (x + 2*y - 7)^2 + (2*x + y - 5)^2;              % 17. Booth Function
    (1 + (x + y + 1)^2*(19 - 14*x + 3*x^2 - 14*y + 6*x*y + 3*y^2)) * (30 + (2*x - 3*y)^2*(18 - 32*x + 12*x^2 + 48*y - 36*x*y + 27*y^2)) % 20. Goldstein-Price Function
};


Names = {
    'Exponential Function'
    'Quadratic Function';
    'Quadratic Function with Offset';
    'Trigonometric Function';
    'Gaussian Function';
    'Fourth-Order Function';
    'Quadratic with Linear Terms';
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
    'Goldstein-Price Function'
};


% Parameters for the trust_region function
x0 = 5; % Initial point
y0 = 3; % Initial point for y
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-3; % Tolerance for convergence
method = 'tay';
deg = 2; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method

% Loop through each function and apply the trust_region method
for i = 1:length(functions)
    expr = functions{i};
    method = 'tay';
    % Perform the trust region method
    [result_tay, history_tay] = trust_region_2d(expr, x, y, x0, y0, r0, deg, maxIter, tol, method, rf, eta);
    % [result_tay, history_tay] = trust_region_2d_naive(expr, x, y, x0, y0, r0, deg, maxIter, tol, method, rf, eta);

    % Display results
    disp(['Function ', char(expr), ':']);
    disp(['Function ', num2str(i), ' with tay method:']);
    disp([num2str(double(result_tay(1))), ' ',num2str(double(result_tay(2)))]);

    % Repeat with the 'pad' method
    method = 'pad';
    
    profile on;
    [result_pad, history_pad] = trust_region_nd(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
    disp('Profiling is running. Press any key to continue after turning off profiling.');

    profile viewer;
    pause;

    % Display results for 'pad' method
    disp(['Function ', num2str(i), ' with pad method:']);
    disp([num2str(double(result_pad(1))), ' ', num2str(double(result_pad(2)))]);

    % Define the range for plotting
    % Define the range for plotting''';
    x_min = min([history_tay(:,1); history_pad(:,1)]) - 1;
    x_max = max([history_tay(:,1); history_pad(:,1)]) + 1;
    y_min = min([history_tay(:,2); history_pad(:,2)]) - 1;
    y_max = max([history_tay(:,2); history_pad(:,2)]) + 1;
    
    [X, Y] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
    Z = double(subs(expr, {x, y}, {X, Y}));
    
    % Evaluate the objective function at the final points
    f_tay = double(subs(expr, {x, y}, {double(result_tay(1)), double(result_tay(2))}));
    f_pad = double(subs(expr, {x, y}, {double(result_pad(1)), double(result_pad(2))}));
    
    % Plot the results in 3D
    fig = figure;
    surf(X, Y, Z, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
    plot3(history_tay(:,1), history_tay(:,2), double(subs(expr, {x, y}, {history_tay(:,1), history_tay(:,2)})), 'b-s', 'LineWidth', 5, 'MarkerSize', 10);
    plot3(history_pad(:,1), history_pad(:,2), double(subs(expr, {x, y}, {history_pad(:,1), history_pad(:,2)})), 'r-s', 'LineWidth', 5, 'MarkerSize', 10);
    
    title(['Function: ', Names(i), ' and its Approximations']);
    legend({
        'Original Function', 
        ['Taylor Approximation ', num2str(size(history_tay, 1)), ...
         ' steps (', num2str(double(result_tay(1))), ', ', num2str(double(result_tay(2))), ...
         '), f=', num2str(f_tay)], 
        ['Pade Approximation ', num2str(size(history_pad, 1)), ...
         ' steps (', num2str(double(result_pad(1))), ', ', num2str(double(result_pad(2))), ...
         '), f=', num2str(f_pad)]
    }, 'Location', 'Best');
    
    xlabel('x');
    ylabel('y');
    zlabel('f(x, y)');

    savefig(['ImprovedConv/', Names{i},'.fig']);
    grid on;
    hold off;
    %close(fig); 
end
