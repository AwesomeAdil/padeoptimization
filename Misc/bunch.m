syms x;

% Define the functions in a cell array
% functions = {
%     x^2 + 2*x + sin(2*x) + exp(-x),
%     x^3 - 3*x + cos(x) + log(x + 1),
%     sin(x) + cos(2*x) + exp(x),
%     exp(-x^2) + x^4 - 4*x^2,
%     x^5 - 5*x^3 + 4*x + sin(3*x),
%     tan(x) + x^2 - log(x + 2),
%     sqrt(x) + sin(x^2) + exp(-x),
%     log(x + 10) + exp(x) - x^2,
%     x^2 * sin(x) - cos(x) + exp(-x),
%     x^3 - sin(x) + sqrt(x + 5),
%     sin(x),
%     x^4 - 2*x^2 + x*log(x + 1),
%     exp(sin(x)) + x^3 - x,
%     cos(x^2) + x^3 - 2*x,
%     log(x + 2) + x*sin(x),
%     x^4*exp(-x) - x^2 + cos(x),
%     x^2*tan(x) - sin(x),
%     exp(x^2 - x) - x^3,
%     cos(x) + x*log(x + 3) - exp(-x),
%     x^5 - 4*x^3 + 3*x + tan(x),
%     sqrt(x + 1) + x^2*cos(x) - log(x + 5),
%     exp(x)*sin(x) - x^3 + 2*x,
%     tan(x^2) - x*exp(-x),
%     x^4 - x^2 + exp(x)*log(x + 2)
% };

functions = {
    x^2 + 2*x + sin(2*x) + exp(-x),
    x^3 - 3*x + cos(x) + log(x + 1),
    sin(x) + cos(2*x) + exp(x),
    exp(-x^2) + x^4 - 4*x^2,
    x^5 - 5*x^3 + 4*x + sin(3*x),
    tan(x) + x^2 - log(x + 2),
    %5*log(x + 10) + exp(x) - x^2,
    x^2 * sin(x) - cos(x) + exp(-x),
    sin(x),
    %x^4 - 2*x^2 + x*log(x + 1),
    exp(sin(x)) + x^3 - x,
    cos(x^2) + x^3 - 2*x,
    %log(x + 2) + x*sin(x),
    %x^4*exp(-x) - x^2 + cos(x),
    x^2*tan(x) - sin(x),
    exp(x^2 - x) - x^3,
    %cos(x) + x*log(x + 3) - exp(-x),
    x^5 - 4*x^3 + 3*x + tan(x),
    exp(x)*sin(x) - x^3 + 2*x,
    tan(x^2) - x*exp(-x),
    x^4 - x^2 + exp(x)*log(x + 2),
    cos(sin(x))
};

% Parameters for the trust_region function
x0 = 1; % Initial point
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-6; % Tolerance for convergence
method = 'tay';
deg = 2; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method

% Loop through each function and apply the trust_region method
for i = 1:length(functions)
    expr = functions{i};
    method = 'tay';
    % Perform the trust region method
    [a_tay, b_tay] = trust_region(expr, x, x0, r0, rf, maxIter, tol, method, deg, eta);
    
    % Display results
    disp(['Function ', char(expr), ':']);
    disp(['Function ', num2str(i), ' with tay method:']);
    %disp('a =');
    %disp(a_tay);
    %disp('b =');
    %disp(b_tay);
    disp('Length of b =');
    disp(length(b_tay));
    
    % Repeat with the 'pad' method
    method = 'pad';
    [a_pad, b_pad] = trust_region(expr, x, x0, r0, rf, maxIter, tol, method, deg, eta);
    
    % Display results for 'pad' method
    disp(['Function ', num2str(i), ' with pad method:']);
    %disp('a =');
    %disp(a_pad);
    %disp('b =');
    %disp(b);
    disp('Length of b =');
    disp(length(b_pad));



    % Define the range for plotting
    x_min = min(min(b_tay)-1, min(b_pad)-1);
    x_max = max(max(b_tay)-1, max(b_pad)+1);
    x_values = linspace(x_min, x_max, 500);
    
     % Evaluate the original function and approximations at x_values
    original_values = double(subs(expr, x, x_values));
    tay_values = double(subs(a_tay, x, x_values));
    pad_values = double(subs(a_pad, x, x_values));
    
    % Plot the results in subplots
    figure;
    
    % Taylor Approximation Plot
    subplot(1, 2, 1);
    plot(x_values, original_values, 'k-', 'LineWidth', 5); hold on;
    %plot(x_values, tay_values, 'b--', 'LineWidth', 2);
    plot(b_tay, double(subs(expr, x, b_tay)), 'b-s', 'LineWidth', 5, 'MarkerSize', 10); % Plot connected points for Taylor
    title(['Taylor Approximation for Function: ', char(expr)]);
    legend('Original Function', 'Taylor Approximation', 'Location', 'Best');
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
    % Pade Approximation Plot
    subplot(1, 2, 2);
    plot(x_values, original_values, 'k-', 'LineWidth', 5); hold on;
    %plot(x_values, pad_values, 'r-.', 'LineWidth', 2);
    plot(b_pad, double(subs(expr, x, b_pad)), 'r-s', 'LineWidth', 5, 'MarkerSize', 10); % Plot connected points for Pade
    title(['Pade Approximation for Function: ', char(expr)]);
    legend('Original Function', 'Pade Approximation', 'Location', 'Best');
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
end