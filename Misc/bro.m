syms x y;

functions = {
    % exp(x+y),                                       % 0. Exponential Function
    % x^2 + y^2;                                      % 1. Quadratic Function
    % (x - 1)^2 + (y - 2)^2 + 1;                      % 2. Quadratic Function with Offset
    sin(x) + cos(y);                                % 3. Trigonometric Function
    % exp(-x^2 - y^2);                                % 4. Gaussian Function
    % x^4 - 16*x^2 + y^4 - 16*y^2 + 64;               % 5. Fourth-Order Function
    % x^2 - 2*x*y + 2*y^2 + 2*x - 4*y + 4;            % 6. Quadratic with Linear Terms
    % (x^2 + y - 11)^2 + (x + y^2 - 7)^2;             % 7. Himmelblau's Function
    % 0.26*(x^2 + y^2) - 0.48*x*y;                    % 8. Beale's Function
    % sin(x) * cos(y);                                % 9. Sine-Cosine Function
    % x^2 * y^2;                                      % 10. Product Function
    % -((1 + cos(12*sqrt(x^2 + y^2)))/(0.5*(x^2 + y^2) + 2)); % 11. Drop Wave Function
    % (x^2 + y^2)/4000 - (cos(x) + 1)*(cos(y)/sqrt(2) + 1 ); % 12. Griewank Function
    % 0.26*(x^2 + y^2) - 0.48*x*y;                    % 13. Matyas Function
    % (1 - x)^2 + 100*(y - x^2)^2;                    % 14. Rosenbrock Function
    % sin(x + y) + (x - y)^2 - 1.5*x + 2.5*y + 1;     % 15. McCormick Function
    % -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 16. Easom Function
    % (x + 2*y - 7)^2 + (2*x + y - 5)^2;              % 17. Booth Function
    %100*sqrt(abs(y - 0.001*x^2)) + 0.01*abs(x + 10); % 18. Bukin Function
    %-0.0001*(abs(sin(x)*sin(y)*exp(abs(100 - (sqrt(x^2 + y^2))/pi) + 1)))^0.1; % 19. Cross-in-Tray Function
    % (1 + (x + y + 1)^2*(19 - 14*x + 3*x^2 - 14*y + 6*x*y + 3*y^2)) * (30 + (2*x - 3*y)^2*(18 - 32*x + 12*x^2 + 48*y - 36*x*y + 27*y^2)) % 20. Goldstein-Price Function
};

Names = {
    % 'Exponential Function'
    % 'Quadratic Function';
    % 'Quadratic Function with Offset';
    'Trigonometric Function';
    % 'Gaussian Function';
    % 'Fourth-Order Function';
    % 'Quadratic with Linear Terms';
    % 'Himmelblau''s Function';
    % 'Beale''s Function';
    % 'Sine-Cosine Function';
    % 'Product Function';
    % 'Drop Wave Function';
    % 'Griewank Function';
    % 'Matyas Function';
    % 'Rosenbrock Function';
    % 'McCormick Function';
    % 'Easom Function';
    % 'Booth Function';
    % 'Goldstein-Price Function'
};



% Parameters for plotting
x_min = -20;
x_max = 20;
y_min = -20;
y_max = 20;

% Define the meshgrid for plotting
[X, Y] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));

% Loop through each function and generate the plots
for i = 1:length(functions)
    expr = functions{i};
    
    % Compute the Taylor and Padé approximations
    taylor_approx = taylor(expr, [x, y] ,'Order', 3, 'ExpansionPoint', [1, 1]);
    
    disp(['Function: ',num2str(i)]);
    disp(expr);
    [coefs, terms] = coeffs(taylor_approx, [x y]);
    simplified_terms = arrayfun(@(term) simplify(subs(term, [(x-1), (y-1)], [x, y])), terms, 'UniformOutput', false);
    
    % Reconstruct the simplified expression
    digits(10);
    simplified_expr = vpa(sum(coefs .* [simplified_terms{:}]));
    disp(simplified_expr);
    pade_approx = two_D_pade_one(x, y, simplified_expr);
    disp(vpa(simplified_expr));
    disp(pade_approx);
    % Evaluate the original function and approximations
    Z_orig = double(subs(expr, {x, y}, {X, Y}));
    Z_taylor = double(subs(taylor_approx, {x, y}, {X, Y}));

    % Evaluate the Padé approximation and its denominator
    [num, den] = numden(pade_approx);
    Z_pade_num = double(subs(num, {x, y}, {X, Y}));
    Z_pade_den = double(subs(den, {x, y}, {X, Y}));
    
    % Create a mask for valid Padé approximation points (denominator non-zero)
    valid_mask = Z_pade_den ~= 0;
    
    % Create a new figure for each function
    figure;
    set(gcf, 'NumberTitle', 'off');
    set(gcf, 'Name', [num2str(i), '. ', Names{i}]);

    % Original function plot
    subplot(1, 3, 1);
    surf(X, Y, Z_orig, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    title(['Function ', num2str(i), ': ', Names(i)]);
    xlabel('x');
    ylabel('y');
    zlabel('f(x, y)');
    grid on;
    
    % Taylor approximation plot
    subplot(1, 3, 2);
    surf(X, Y, Z_taylor, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    title(['Taylor Approximation ',char(vpa(simplified_expr))]);
    xlabel('x');
    ylabel('y');
    zlabel('f(x, y)');
    grid on;
    
    % Padé approximation plot
    subplot(1, 3, 3);
    Z_pade = Z_pade_num ./ Z_pade_den;
    Z_pade(~valid_mask) = NaN; % Ignore invalid points
    surf(X, Y, Z_pade, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
    title(['Padé Approximation ', char(vpa(pade_approx))]);
    xlabel('x');
    ylabel('y');
    zlabel('f(x, y)');
    grid on;
    clear coeffs;
end
