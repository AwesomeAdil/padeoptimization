syms x;

% Define the functions
functions = {
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
    cos(sin(x))
};

% Define the range for plotting
x_values = linspace(-2, 2, 500);

for i = 1:length(functions)
    func = functions{i};
    
    % Taylor series approximation (degree 5 for simplicity)
    taylor_approx = taylor(func, x, 'Order', 6);
    
    % Padé approximation (order [2, 2] for simplicity)
    pade_approximation = pade(func, x, 'Order', 4);
    
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
