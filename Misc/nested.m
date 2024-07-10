syms x y 
expr =  -cos(x)*cos(y)*exp(-((x - pi)^2 + (y - pi)^2)); % 16. Easom Function

taylor_approx = taylor(expr, [x, y] ,'Order', 3, 'ExpansionPoint', [1, 1]);

disp(expr);
a = vpa(simplify(pade(expr, 'Order', 2, 'ExpansionPoint', 1)));
disp("Nested= ");
disp(a);
[n,d] = numden(a);   % Extract numerator

n = expand(n);
d = expand(d);

[num_coeffs, num_terms] = coeffs(n, x);
[den_coeffs, den_terms] = coeffs(d, x);

num_expr = sym(0);
den_expr = sym(0);


disp("LENNN");
disp(num2str(length(num_coeffs)));
disp(num2str(length(num_terms)));

disp(num2str(length(den_coeffs)));
disp(num2str(length(den_terms)));


for i = 1:length(num_coeffs)
    num_expr = num_expr + num_terms(i) * pade(num_coeffs(i), y, 'Order', 2, 'ExpansionPoint', 1);
end

disp('Numerator= ');
disp(num_terms);

for i = 1:length(den_coeffs)
    den_expr = den_expr + den_terms(i) * pade(den_coeffs(i),y, 'Order', 2, 'ExpansionPoint', 1);
end

disp('Denominator= ');
disp(den_terms);

total_exp = num_expr/den_expr;
disp('Total= ');
disp(total_exp);


% Parameters for plotting
x_min = -8;
x_max = 8;
y_min = -8;
y_max = 8;

[X, Y] = meshgrid(linspace(x_min, x_max, 100), linspace(y_min, y_max, 100));
Z_orig = double(subs(expr, {x, y}, {X, Y}));
Z_taylor = double(subs(taylor_approx, {x, y}, {X, Y}));
Z_pade_num = double(subs(num_expr, {x, y}, {X, Y}));
Z_pade_den = double(subs(den_expr, {x, y}, {X, Y}));
    
% Create a mask for valid Padé approximation points (denominator non-zero)
valid_mask = Z_pade_den ~= 0;

figure;
set(gcf, 'NumberTitle', 'off');
set(gcf, 'Name', "e^(x+y)");

subplot(1, 3, 1);
surf(X, Y, Z_orig, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Function ', ': ', "e^(x+y)"]);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
grid on;

% Taylor approximation plot
subplot(1, 3, 2);
surf(X, Y, Z_taylor, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Taylor Approximation ',char(vpa(taylor_approx))]);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
grid on;

% Padé approximation plot
subplot(1, 3, 3);
Z_pade = Z_pade_num ./ Z_pade_den;
Z_pade(~valid_mask) = NaN; % Ignore invalid points
surf(X, Y, Z_pade, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
title(['Padé Approximation ', char(vpa(total_exp))]);
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
grid on;
clear;
