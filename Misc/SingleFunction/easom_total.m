% Define the Easom function symbolically
syms xs ys

sx = 1.3;
sy = 1.3;

easom_sym = -cos(xs)*cos(ys)*exp(-((xs - pi)^2 + (ys - pi)^2));
%easom_sym = (xs^2 + ys^2)/4000 - (cos(xs) + 1)*(cos(ys)/sqrt(2) + 1);

% Evaluate the Easom function at (sx, sy)
easom_value_at_sx_sy = double(subs(easom_sym, {xs, ys}, {sx, sy}));

% Define the nested_pade function call to get the Padé approximation
f_pade_sym = nested_pade(vpa(easom_sym), xs, ys, sx, sy, 2);

% Convert the symbolic Padé approximation to a MATLAB function
f_pade = matlabFunction(f_pade_sym);

% Define the range for x and y
[x, y] = meshgrid(linspace(-10, 10, 400));

% Calculate the corresponding z values using the Padé approximation
z = f_pade(x, y);

% Plot the Padé approximated function
figure;
surf(x, y, z, 'EdgeColor', 'interp', 'FaceAlpha', 0.5);
title('Padé Approximated Easom Function');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
colorbar;
grid on;

% Define the function for fmincon
fmin = @(v) f_pade(v(1), v(2));

% Define the initial guess for the minimum
initial_guess = [sx, sy];

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
plot3(x_opt(1), x_opt(2), fval, 'ro', 'MarkerSize', 10, 'LineWidth', 5);
plot3(sx, sy, easom_value_at_sx_sy, 'bo', 'MarkerSize', 10, 'LineWidth', 5);
legend('f(x, y)', 'Local Minimum', 'Value at (sx, sy)');
hold off;