function [result, history] = trust_region_nd_slicer(f, xs, x0, delta, max_iter, tol, hat, thresh)
    m = f;
    x = x0;
    history = x0;
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = vpa(subs(f, xs, x));
    syms t;
    
    prev_x = x;
    
    for ind = 0:max_iter
        m = nested_pade(vpa(f), xs, x, 2);
        f_handle = matlabFunction(m, 'Vars', xs);
        initial_guess = x;
        objective = @(v) f_handle(v(1), v(2));
       
        lb = x - delta * ones(length(x));  
        ub = x + delta * ones(length(x));  
      
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

        param_func_f = subs(f, xs, prev_x + t * direction);
        param_func_m = subs(m, xs, prev_x + t * direction);
        t_current = norm(x_opt - prev_x) / norm(direction);
        t_values = linspace(-5, 5, 100);  
        y_values_f = double(subs(param_func_f, t, t_values));  
        y_values_m = double(subs(param_func_m, t, t_values));  

        figure;
        
        subplot(2, 1, 1);
        plot(t_values, y_values_f, 'LineWidth', 2);
        
        xlabel('t');
        ylabel('f(point + t * direction)');
        title('2D Plot of Symbolic Function f along a Direction');
        grid on;
        plot(0, vpa(subs(f, xs, prev_x)), 'go', 'MarkerFaceColor', 'g');
        plot(t_current, vpa(subs(f, xs, x_opt)), 'ro', 'MarkerFaceColor', 'r'); 
        subplot(2, 1, 2);
        plot(t_values, y_values_m, 'LineWidth', 2);  
        xlabel('t'); 
        ylabel('m(point + t * direction)');
        title('2D Plot of Symbolic Function m along a Direction');
        grid on;
        
        hold on;
   
        
        plot(t_current, vpa(subs(m, xs, x_opt)), 'bo', 'MarkerFaceColor', 'b'); 
        plot(0, vpa(subs(m, xs, prev_x)), 'mo', 'MarkerFaceColor', 'm'); 
        hold off;

        prev_x = x;
        
        waitfor(gcf);

        history(end + 1, :) = x;
        
    end

    result = x; 
end