% Define the 2D rational function
f = @(x, y) (x.^2 + y.^2 - 2*x - 4*y + 4) ./ (x.^2 + y.^2 + 1);

% Define the range for x and y
[x, y] = meshgrid(linspace(-3, 3, 100));

% Calculate the corresponding z values
z = f(x, y);

% Plot the function
figure;
surf(x, y, z, 'EdgeColor', 'none');
title('2D Rational Function with Local Minimum');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
colorbar;
grid on;

% Define the function for fmincon
fmin = @(v) f(v(1), v(2));

% Define the initial guess for the minimum
initial_guess = [0, 0];

% Define bounds to prevent numerical issues
lb = [-10, -10];  % Lower bounds
ub = [10, 10];    % Upper bounds

% Use fmincon to find the minimum
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');
[x_opt, fval] = fmincon(fmin, initial_guess, [], [], [], [], lb, ub, [], options);

% Display the result
fprintf('Local minimum found at (x, y) = (%.4f, %.4f) with value f(x, y) = %.4f\n', x_opt(1), x_opt(2), fval);

% Plot the local minimum point
hold on;
plot3(x_opt(1), x_opt(2), fval, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('f(x, y)', 'Local Minimum');
hold off;