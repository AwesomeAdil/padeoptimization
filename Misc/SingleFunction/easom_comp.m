syms x y;

functions = {
    -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 16. Easom Function
};

Names = {
    'Easom Function';
};


% Loop through each function and generate the plots
for i = 1:length(functions)
    expr = functions{i};
    n_taylor = 3;
    % Compute the Taylor and Padé approximations
    taylor_approx = taylor(expr, [x, y] ,'Order', n_taylor, 'ExpansionPoint', [1, 1]);
    
    disp(['Function: ',num2str(i)]);
    disp(expr);
    

    
    [coefs, terms] = coeffs(taylor_approx, [x y]);
    simplified_terms = arrayfun(@(term) simplify(subs(term, [(x-1), (y-1)], [x, y])), terms, 'UniformOutput', false);
    
    % Reconstruct the simplified expression
    simplified_expr = sum(coefs .* [simplified_terms{:}]);
    disp(char(vpa(simplified_expr)));
    % Regular
    pade_approx = nested_pade(expr, x, y, 1, 1, 2);

    % Naive
    % pade_approx = two_D_pade_one(x,y,taylor_approx);

    disp(char(vpa(pade_approx)));
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
    fig = figure;
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
    digits(5);
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
    %saveas(fig, [num2str(i),'.svg']);
end
