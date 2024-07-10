syms x y;

functions = {
    exp(x+y),                                       % 0. Exponential Function
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
    'Exponential Function';
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

% Define the values of x and y
x_val = 1;
y_val = 2;

% Convert the functions to function handles for evaluation
f_eval = cellfun(@matlabFunction, functions, 'UniformOutput', false);

% Evaluate functions at x_val, y_val
f_values = cellfun(@(f) f(x_val, y_val), f_eval);

% Compute gradients
grad_functions = cell(size(functions));
grad_pade = cell(size(functions));

npade = 2;

for i = 1:length(functions)
    grad_functions{i} = gradient(vpa(simplify(functions{i})), [x, y]);
    grad_pade{i} = gradient(simplify(nested_pade(functions{i}, x, y, x_val, y_val, npade)));
end

% Evaluate gradients at x_val, y_val
grad_values = cellfun(@(grad) double(subs(grad, [x, y], [x_val, y_val])), grad_functions, 'UniformOutput', false);
grad_pade_values = cellfun(@(grad) double(subs(grad, [x, y], [x_val, y_val])), grad_pade, 'UniformOutput', false);

% Compute Hessians
hess_functions = cell(size(functions));
hess_pades = cell(size(functions));

for i = 1:length(functions)
    hess_pades{i} = hessian(vpa(simplify(functions{i})), [x, y]);
    hess_functions{i} = hessian(simplify(nested_pade(functions{i}, x, y, x_val, y_val, npade)));
end

% Evaluate Hessians at x_val, y_val
hess_values = cellfun(@(hess) double(subs(hess, [x, y], [x_val, y_val])), hess_functions, 'UniformOutput', false);
hess_pade_values = cellfun(@(hess) double(subs(hess, [x, y], [x_val, y_val])), hess_pades, 'UniformOutput', false);

% Display results
for i = 1:length(functions)
    disp(['Function: ', Names{i}]);
    disp('Gradient:');
    disp(['[', num2str(grad_values{i}(1)), ', ', num2str(grad_values{i}(2)), ']']);
    disp(['[', num2str(grad_pade_values{i}(1)), ', ', num2str(grad_pade_values{i}(2)), ']']);
    disp('Hessian:');
    disp(['[', num2str(hess_values{i}(1,1)), ', ', num2str(hess_values{i}(1,2)), '; ', ...
                 num2str(hess_values{i}(2,1)), ', ', num2str(hess_values{i}(2,2)), ']']);
    disp(['[', num2str(hess_pade_values{i}(1,1)), ', ', num2str(hess_pade_values{i}(1,2)), '; ', ...
                 num2str(hess_pade_values{i}(2,1)), ', ', num2str(hess_pade_values{i}(2,2)), ']']);
    disp(' ');
end