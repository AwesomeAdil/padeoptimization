% Step 1: Define the symbolic variables and the multivariate functions
syms x1 x2 x3 x4 t
xv = [x1, x2, x3, x4];  % Symbolic variable list

% Example: two complex multivariate functions
f = x1^2 + x2^2 - x3^2 + exp(x4);
m = x1 * x2 + sin(x3) - log(x4 + 5);

% Step 2: Choose a point in the domain
point = [1, 1, 1, 1];  % Example point in 4D

% Step 3: Define the direction vector
direction = [1, 0.5, -0.5, 2];  % Example direction vector in 4D

% Normalize the direction vector
direction = direction / norm(direction);

% Step 4: Create parameterized functions
% t will vary along the direction from the point
param_func_f = subs(f, xv, point + t * direction);
param_func_m = subs(m, xv, point + t * direction);

% Step 5: Evaluate the parameterized functions for a range of t values
t_values = linspace(-5, 5, 100);  % Define the range for t
y_values_f = double(subs(param_func_f, t, t_values));  % Evaluate f at each t
y_values_m = double(subs(param_func_m, t, t_values));  % Evaluate m at each t

% Step 6: Plot the parameterized functions in subplots
figure;

% Subplot for function f
subplot(2, 1, 1);
plot(t_values, y_values_f, 'LineWidth', 2);  % Set LineWidth to 2
xlabel('t');
ylabel('f(point + t * direction)');
title('2D Plot of Symbolic Function f along a Direction');
grid on;

% Subplot for function m
subplot(2, 1, 2);
plot(t_values, y_values_m, 'LineWidth', 2);  % Set LineWidth to 2
xlabel('t');
ylabel('m(point + t * direction)');
title('2D Plot of Symbolic Function m along a Direction');
grid on;