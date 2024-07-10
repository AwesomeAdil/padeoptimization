syms x;

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

for i = 1:length(functions)
    expr = functions{i};
    plot(x_values, original_values, 'k-', 'LineWidth', 5); hold on;
    grid on

    fplot(expr)
    fplot(pade(expr, 'Order', 1))
    fplot(taylor(expr, 'Order', 2))
    disp(pade(expr,'Order', 1))
    disp(taylor(expr, 'Order', 2))


    axis([-4 4 -4 4])
    legend('Exp','Pade [2,2]','Taylor',...
                                            'Location','Best')
    title('Difference Between exp(x) and its Pade Approximant')
    ylabel('Error')
    hold off;
end