syms x;
expr = x^2 + 2*x + sin(2*x) + exp(-x);
[a, b] = trust_region(expr, x, 1, 1, 4, 100, 1e-6, 'tay', 4, 0.2);
a
b
length(b)
%[a, b] = trust_region(expr, x, 1, 1, 4, 100, 1e-6, 'pad', 4, 0.2);
%b
%a
%length(b)
%m = taylor(expr, 'Order', 5, 'ExpansionPoint', x);
%m = pade(expr, 'Order', 2);
