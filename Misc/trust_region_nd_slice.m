function [result, history] = trust_region_nd_slice(f, xs, x0, delta, max_iter, tol, hat, thresh)
    m = f;
    x = x0;
    history = x0;
    % Generate a grid for plotting the model function
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = vpa(subs(f, xs, x));
    syms t;
    for ind = 0:max_iter
        m = nested_pade(vpa(f), xs, x, 2);
        f_handle = matlabFunction(m, 'Vars', xs);
        initial_guess = x;
        objective = @(v) f_handle(v(1), v(2));
       
        lb = x - delta * ones(length(x));  % Lower bounds
        ub = x + delta * ones(length(x));  % Upper bounds
      
        [x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
        
        num = vpa(subs(f, xs, x)) - vpa(subs(f, xs, x_opt));
        den = old_fval - fval;

        old_fval = fval;

        direction = x_opt - x; 
        direction = direction / norm(direction);

        if abs(num) < 1e-9 && abs(den) < 1e-9
            result = x;
            return;
        end

        if abs(den) < 1e-9
            if num > 0
                ratio = 1;
            else
                ratio = 0;
            end
        else
            ratio = num / den;
        end

        if ratio < 0.25
            delta = delta / 4;
        else
            if ratio > 0.75 && delta == norm(x_opt - x, inf)
                delta = min(2 * delta, hat);
            end
        end

        if norm(x_opt - x, inf) < double(tol)
           result = x;
           return;
        end

        if ratio > thresh
           x = x_opt;
        end

        history(end + 1, :) = x;

        
        % Step 4: Create parameterized functions
        % t will vary along the direction from the point
        param_func_f = subs(f, xs, x + t * direction);
        param_func_m = subs(m, xs, x + t * direction);
        
        % Step 5: Evaluate the parameterized functions for a range of t values
        t_values = linspace(-15, 15, 100);  % Define the range for t
        y_values_f = double(subs(param_func_f, t, t_values));  % Evaluate f at each t
        y_values_m = double(subs(param_func_m, t, t_values));  % Evaluate m at each t
        
        % Step 6: Plot the parameterized functions in subplots
        figure;
        
        % Subplot for function f
        subplot(2, 1, 1);
        plot(t_values, y_values_f, 'LineWidth', 2);
        xlabel('t');
        ylabel('f(point + t * direction)');
        title('2D Plot of Symbolic Function f along a Direction');
        grid on;
        
        % Subplot for function m
        subplot(2, 1, 2);
        plot(t_values, y_values_m, 'LineWidth', 2);  % Set LineWidth to 2
        xlabel('t');
        ylabel('m(point + t * direction)');
        title('2D Plot of Symbolic Function m along a Direction');
        grid on;
        waitfor(gcf);

    end

    result = x; 
end