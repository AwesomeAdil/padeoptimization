% Define symbolic variables
syms x y

% Define the unsimplified rational function
f = (x^2 + 2*x*y + y^2) / (x + y) + (3*x^2 - y^2) / (2*x - y);

% Convert to MATLAB function handle for visualization
f_handle = matlabFunction(f, 'Vars', [x, y]);

% Generate grid for visualization
[xGrid, yGrid] = meshgrid(linspace(-10, 10, 100), linspace(-10, 10, 100));
zGrid = f_handle(xGrid, yGrid);

% Plot the function
figure;
surf(xGrid, yGrid, zGrid);
title('Visualization of the Rational Function');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');