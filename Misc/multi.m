% Step 1: Define the symbolic variables and the function
syms x y
f = exp(x+y);  % Example multivariable function

% Step 2: Compute the Taylor series expansion around x = 0, y = 0
taylor_series = taylor(f, [x, y], 'Order', 3);  % Expanding up to total degree 3

p = two_D_pade_one(x, y, taylor_series);
disp(p);