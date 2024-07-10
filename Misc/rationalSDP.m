% Define symbolic variables
syms x y

% Define the unsimplified rational function
f = (x^2 + 2*x*y + y^2) / (x + y) + (3*x^2 - y^2) / (2*x - y);

% Simplify the function
f_simplified = simplify(f);

% Display the simplified function
disp('Simplified function:');
disp(f_simplified);

% Convert the simplified symbolic function to a MATLAB function handle
f_handle = matlabFunction(f_simplified, 'Vars', [x, y]);

% Define the initial guess closer to the expected solution
initial_guess = [1, 1];

% Define bounds to prevent numerical issues
lb = [-10; -10];  % Lower bounds
ub = [10; 10];    % Upper bounds

% Use fmincon with improved settings
options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

% Define the objective function for fmincon
objective = @(v) f_handle(v(1), v(2));

% Call fmincon to find the minimum with bounds
[x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);

% Display the results
disp('Optimal solution:');
disp(x_opt);
disp('Objective value at optimal solution:');
disp(fval);