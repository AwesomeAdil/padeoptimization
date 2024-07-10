syms x y;

functions = {
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 16. Easom Function
};

Names = {
    'Easom Function';
};

% Parameters for plotting
x_min = -5;
x_max = 5;
y_min = -5;
y_max = 5;

% Parameters for the trust_region function
x0 = 1; % Initial point
y0 = 1; % Initial point for y
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-3; % Tolerance for convergence
method = 'tay';
deg = 2; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method
% Loop through each function and generate the plots

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

    % try
    %[result_pad, history_pad] = trust_region_nd_slicer_both(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
    
    [result_pad, history_pad] = trust_region_nd_slicer_both(expr, [x, y], [x0, y0], r0, maxIter, tol, rf, eta);
    % catch
    %     result_pad = [1,1];
    %     history_pad = [1, 1];
    % end

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
