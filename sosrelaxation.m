% Define the 2D rational function
f_num = @(x, y) x.^2 + y.^2 - 2*x - 4*y + 4;  % Numerator
f_den = @(x, y) x.^2 + y.^2 + 1;              % Denominator

% Use YALMIP to define variables and expressions
x = sdpvar(1,1);
y = sdpvar(1,1);

% Define the rational function
f = (x^2 + y^2 - 2*x - 4*y + 4) / (x^2 + y^2 + 1);

% Define the objective and constraints for SOS relaxation
gamma = sdpvar(1,1);  % The variable we want to maximize
objective = -gamma;   % Maximize gamma by minimizing -gamma
constraints = [sos(f_num(x,y) - gamma * f_den(x,y))];

% Set options for the solver
options = sdpsettings('solver', 'sedumi', 'verbose', 2);  % Example with SEDUMI
options = sdpsettings(options, 'sos.model', 2);  % Change SOS model setting

% Solve the problem
sol = optimize(constraints, objective, options);

% Check if the problem is feasible
if sol.problem == 0
    gamma_opt = value(gamma);
    x_opt = value(x);
    y_opt = value(y);
    fprintf('Optimal value found: gamma = %.4f\n', gamma_opt);
    fprintf('Optimal point found: x = %.4f, y = %.4f\n', x_opt, y_opt);
    
    % Plot the function and the optimal point
    [x_mesh, y_mesh] = meshgrid(linspace(-3, 3, 100));
    z = f_num(x_mesh, y_mesh) ./ f_den(x_mesh, y_mesh);

    figure;
    surf(x_mesh, y_mesh, z, 'EdgeColor', 'none');
    hold on;
    plot3(x_opt, y_opt, gamma_opt, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    title('2D Rational Function with Optimal Value');
    xlabel('x');
    ylabel('y');
    zlabel('f(x, y)');
    colorbar;
    grid on;
    legend('f(x, y)', 'Optimal Point');
    hold off;
else
    fprintf('Optimization failed. Solver message: %s\n', sol.info);
end