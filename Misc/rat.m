% Define the function
f = @(x) (x.^3 - 3*x) ./ (x.^2 + 1);

% Define the range for x
x = linspace(-3, 3, 1000);

% Calculate the corresponding y values
y = f(x);

% Plot the function
figure;
plot(x, y, 'LineWidth', 2);
title('Rational Function with Local Minimum');
xlabel('x');
ylabel('f(x)');
grid on;

% Hold the plot to add more details
hold on;

% Find the critical points
syms x_sym
f_sym = (x_sym^3 - 3*x_sym) / (x_sym^2 + 1);

% Convert the simplified symbolic function to a MATLAB function handle
f_handle = matlabFunction(f_sym, 'Vars', x_sym);

% Define the initial guess closer to the expected solution
initial_guess = [1];

% Define bounds to prevent numerical issues
lb = [-10; -10];  % Lower bounds
ub = [10; 10];    % Upper bounds

% Use fmincon with improved settings
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

% Define the objective function for fmincon
objective = @(v) f_handle(v(1));

% Call fmincon to find the minimum with bounds
[x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
x_opt
fval
df_sym = diff(f_sym, x_sym);
critical_points = solve(df_sym == 0, x_sym);

% Convert to numeric values
critical_points = double(critical_points);

% Evaluate the function at the critical points
critical_values = double(subs(f_sym, x_sym, critical_points));

% Plot the critical points
plot(critical_points, critical_values, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% Add a legend
legend('f(x)', 'Critical Points');

% Hold off the plot
hold off;