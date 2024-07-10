% Define the 3D rational function
f = @(x, y, z) (x.^2 + y.^2 + z.^2 - 2*x - 4*y - 6*z + 4) ./ (x.^2 + y.^2 + z.^2 + 1);

% Define the range for x, y, and z
[x, y, z] = ndgrid(linspace(-3, 3, 100), linspace(-3, 3, 100), linspace(-3, 3, 100));

% Calculate the corresponding values (z will be a 3D matrix, so for plotting we'll need to slice it)
w = f(x, y, z);

% Plotting the 3D function would require slicing, here we plot at z=0 for simplicity
z_slice = 0;
w_slice = f(x(:,:,1), y(:,:,1), z_slice*ones(size(x(:,:,1))));

figure;
surf(x(:,:,1), y(:,:,1), w_slice, 'EdgeColor', 'none');
title('3D Rational Function with Local Minimum (slice at z=0)');
xlabel('x');
ylabel('y');
zlabel('f(x, y, z)');
colorbar;
grid on;

% Define the function for fmincon
fmin = @(v) f(v(1), v(2), v(3));

% Define the initial guess for the minimum
initial_guess = [0, 0, 0];

% Define bounds to prevent numerical issues
lb = [-10, -10, -10];  % Lower bounds
ub = [10, 10, 10];    % Upper bounds

% Use fmincon to find the minimum
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');
[x_opt, fval] = fmincon(fmin, initial_guess, [], [], [], [], lb, ub, [], options);

% Display the result
fprintf('Local minimum found at (x, y, z) = (%.4f, %.4f, %.4f) with value f(x, y, z) = %.4f\n', x_opt(1), x_opt(2), x_opt(3), fval);

% Plot the local minimum point on the slice plot
hold on;
plot3(x_opt(1), x_opt(2), f(x_opt(1), x_opt(2), z_slice), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('f(x, y, z)', 'Local Minimum');
hold off;