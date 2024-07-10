syms x;

% Define the functions in a cell array
functions = {
    x^2 + 2*x + sin(2*x) + exp(-x),
    x^3 - 3*x + cos(x) + log(x + 1),
    sin(x) + cos(2*x) + exp(x),
    exp(-x^2) + x^4 - 4*x^2,
    x^5 - 5*x^3 + 4*x + sin(3*x),
    tan(x) + x^2 - log(x + 2),
    log(x + 10) + exp(x) - x^2,
    x^2 * sin(x) - cos(x) + exp(-x),
    sin(x),
    x^4 - 2*x^2 + x*log(x + 1),
    exp(sin(x)) + x^3 - x,
    cos(x^2) + x^3 - 2*x,
    log(x + 2) + x*sin(x),
    x^4*exp(-x) - x^2 + cos(x),
    x^2*tan(x) - sin(x),
    exp(x^2 - x) - x^3,
    cos(x) + x*log(x + 3) - exp(-x),
    x^5 - 4*x^3 + 3*x + tan(x),
    exp(x)*sin(x) - x^3 + 2*x,
    tan(x^2) - x*exp(-x),
    x^4 - x^2 + exp(x)*log(x + 2)
};

% Parameters for the trust_region function
x0 = 1; % Initial point (ensuring it is within the valid domain)
r0 = 1; % Initial trust region radius
rf = 4; % Final trust region radius
maxIter = 100; % Maximum number of iterations
tol = 1e-6; % Tolerance for convergence
deg = 4; % Degree of the polynomial approximation
eta = 0.2; % Parameter for the trust region method

% Loop through each function and apply the trust_region method with constraints
for i = 1:length(functions)
    expr = functions{i};
    
    try
        % Perform the trust region method with Taylor approximation
        method = 'tay';
        [a_tay, b_tay] = trust_region_with_constraints(expr, x, x0, r0, rf, maxIter, tol, method, deg, eta);

        % Perform the trust region method with Pade approximation
        method = 'pad';
        [a_pad, b_pad] = trust_region_with_constraints(expr, x, x0, r0, rf, maxIter, tol, method, deg, eta);
    catch ME
        disp(['Error with function ', char(expr), ': ', ME.message]);
        continue;
    end

    % Define the range for plotting around the expansion point
    expansion_point = x0; % Use the initial point as the expansion point
    epsilon = 0.5; % Define a small neighborhood around the expansion point
    x_values = linspace(expansion_point - epsilon, expansion_point + epsilon, 500);

    % Evaluate the original function and approximations at x_values
    original_values = double(subs(expr, x, x_values));
    tay_values = double(subs(a_tay, x, x_values));
    pad_values = double(subs(a_pad, x, x_values));

    % Plot the results in subplots
    figure;

    % Taylor Approximation Plot
    subplot(1, 2, 1);
    plot(x_values, original_values, 'k-', 'LineWidth', 2); hold on;
    plot(x_values, tay_values, 'b--', 'LineWidth', 2);
    plot(b_tay, double(subs(expr, x, b_tay)), 'bs-', 'LineWidth', 2, 'MarkerSize', 10); % Plot connected points for Taylor
    title(['Taylor Approximation for Function: ', char(expr)]);
    legend('Original Function', 'Taylor Approximation', 'Location', 'Best');
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;

    % Pade Approximation Plot
    subplot(1, 2, 2);
    plot(x_values, original_values, 'k-', 'LineWidth', 2); hold on;
    plot(x_values, pad_values, 'r-.', 'LineWidth', 2);
    plot(b_pad, double(subs(expr, x, b_pad)), 'rs-', 'LineWidth', 2, 'MarkerSize', 10); % Plot connected points for Pade
    title(['Pade Approximation for Function: ', char(expr)]);
    legend('Original Function', 'Pade Approximation', 'Location', 'Best');
    xlabel('x');
    ylabel('y');
    grid on;
    hold off;
end

function [a, b] = trust_region_with_constraints(expr, x, x0, r0, rf, maxIter, tol, method, deg, eta)
    b = x0;
    a = [];

    for k = 1:maxIter
        % Ensure b is within the valid domain
        if b <= 0
            error('Value of x is out of bounds: x must be > 0');
        end
        
        % Compute the Taylor or Pade approximation
        if strcmp(method, 'tay')
            a_k = taylor(expr, x, 'Order', deg + 1, 'ExpansionPoint', b);
        elseif strcmp(method, 'pad')
            a_k = pade(expr, 'Order', deg/2, 'ExpansionPoint', b);
        else
            error('Unknown method');
        end
        
        % Calculate gradient and Hessian
        g = gradient(a_k, x);
        H = hessian(a_k, x);

        % Evaluate gradient and Hessian at current point
        g = double(subs(g, x, b));
        H = double(subs(H, x, b));

        % Compute the Cauchy point or the Newton step
        p_b = - H \ g;

        % Trust region update (simple example)
        if norm(p_b) > r0
            p_b = (r0 / norm(p_b)) * p_b;  % Scaling step to stay within the trust region
        end
        
        % Update the trust region radius (optional)
        r0 = min(rf, r0 * 1.1);
        
        % Update b and check for convergence
        b_new = b + p_b;
        if abs(b_new - b) < tol
            break;
        end
        b = b_new;

        % Store the approximation
        a = [a, a_k];
    end
end
