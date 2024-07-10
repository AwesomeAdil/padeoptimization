syms x;

% Define the original functions
all_functions = {
    x^2 + 2*x + sin(2*x) + exp(-x),
    x^3 - 3*x + cos(x) + log(x + 1),
    sin(x) + cos(2*x) + exp(x),
    exp(-x^2) + x^4 - 4*x^2,
    x^5 - 5*x^3 + 4*x + sin(3*x),
    tan(x) + x^2 - log(x + 2),
    x^2 * sin(x) - cos(x) + exp(-x),
    sin(x),
    exp(sin(x)) + x^3 - x,
    cos(x^2) + x^3 - 2*x,
    x^2 * tan(x) - sin(x),
    exp(x^2 - x) - x^3,
    x^5 - 4*x^3 + 3*x + tan(x),
    exp(x) * sin(x) - x^3 + 2*x,
    tan(x^2) - x * exp(-x),
    x^4 - x^2 + exp(x) * log(x + 2),
    cos(sin(x)),
    exp(x),
    sin(x),
    cos(x),
    x^3 - 6*x^2 + 11*x - 6,
    x^4 - 4*x^3 + 6*x^2 - 4*x + 1,
    1 / (1 + x^2),
    x / (1 + x^2),
    tan(x),
    sec(x),  % sec(x)
    log(1 + x),
    x * exp(x),
    erf(x),
    gamma(x),
    1 / (1 + exp(-x)),  % Logistic (Sigmoid)
    0.5 * x * (1 + erf(x / sqrt(2))),  % GELU
    piecewise(x >= 0, x + 1, x < 0, exp(x) - 1),  % ELU with alpha=1.0
    x * (1 + exp(-x)),  % SiLU
    exp(-x^2)  % Gaussian
};

% Define the range for plotting
x_values = linspace(-2, 2, 500);

for i = 1:length(all_functions)
    func = all_functions{i};
    
    % Taylor series approximation (degree 5 for simplicity)
    taylor_approx = taylor(func, x, 'Order', 3);
    
    % Padé approximation (order [2, 2] for simplicity)
    pade_approximation = pade(func, x, 'Order', 1);
    
    % Plot the results
    figure('Name', ['Function ', num2str(i)]);
    hold on
    grid on

    % Original function
    fplot(func, 'k-', 'LineWidth', 2);
    
    % Taylor series approximation
    fplot(taylor_approx, 'b--', 'LineWidth', 2);
    
    % Pade approximation
    fplot(pade_approximation, 'r-.', 'LineWidth', 2);
    
    % Plot settings
    axis([-4 4 -4 4])
    legend('Original Function', 'Taylor Approximation', 'Pade Approximation', 'Location', 'Best');
    title(['Function: ', char(func)]);
    xlabel('x');
    ylabel('y');
    hold off;
end
