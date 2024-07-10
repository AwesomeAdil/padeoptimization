% Steps:
%% Based on trust region radius, method and initial point solve the Subproblem and see if that brings sufficient reduction
%% Modify radius accordingly and see if we terminate
%% Graph slice

function [result, history] = trust_region_nd_slicer_all(f, xs, x0, delta, max_iter, tol, hat, thresh)
    m = f;
    x = x0;
    history = x0;
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = vpa(subs(f, xs, x));
    syms t;
    
    prev_x = x;
    
    for ind = 0:max_iter
        %% Subproblem
        m = nested_pade(vpa(f), num2cell(xs), x, 2);
        f_handle = matlabFunction(m, 'Vars', xs);
        initial_guess = x;
        objective = @(v) f_handle(v(1), v(2));
       
        lb = x - delta * ones(length(x));  
        ub = x + delta * ones(length(x));  

        [x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
        
        direction = x_opt - x; 
        direction = direction / norm(direction);

        %% Ratio of Reductions
        ip = vpa(subs(f, xs, prev_x));
        fp = vpa(subs(f, xs, x_opt));
        im = vpa(subs(m, xs, prev_x));
        fm = vpa(subs(m, xs, x_opt));

        num = ip - fp;
        den = im - fm;

        %old_fval = fval;

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
        
        norm(x_opt - x, inf)
        if norm(x_opt - x, inf) < double(tol)
           result = x;
           return;
        end

        if ratio > thresh
            x = x_opt;
        end


        %% Graphing
        param_func_f = subs(f, xs, prev_x + t * direction);
        param_func_m = subs(m, xs, prev_x + t * direction);
        t_current = norm(x_opt - prev_x) / norm(direction);
        t_values = linspace(-5, 5, 100);  
        y_values_f = double(subs(param_func_f, t, t_values));  
        y_values_m = double(subs(param_func_m, t, t_values));  

        % Adding a constant shift to avoid negative or zero values
        shift_f = abs(min(y_values_f)) + 1;
        shift_m = abs(min(y_values_m)) + 1;
        y_values_f_shifted = y_values_f + shift_f;
        y_values_m_shifted = y_values_m + shift_m;

        figure;
        
        subplot(2, 1, 1);
        semilogy(t_values, y_values_f_shifted, 'LineWidth', 2);  % Use semilogy for logarithmic scale
        xlabel('t');
        ylabel('log(f(point + t * direction)) (shifted)');
        title('2D Plot of Symbolic Function f along a Direction (Log Scale, Shifted)');
        grid on;
        hold on;
     
        initial_point = ip + shift_f;
        
        final_point = vpa(subs(f, xs, x_opt)) + shift_f;
        plot(0, initial_point, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Initial Point');
        text(0, initial_point, sprintf('(%g, %g)', 0, ip), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        plot(t_current, final_point, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Final Point');
        text(t_current, final_point, sprintf('(%g, %g)', t_current, fp), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        % Draw an arrow from the initial point to the final point
        annotation('arrow', [0.25, 0.75], [0.5, 0.5]); % Adjust the positions accordingly
        legend show;
        hold off;

        subplot(2, 1, 2);
        semilogy(t_values, y_values_m_shifted, 'LineWidth', 2);  % Use semilogy for logarithmic scale
        xlabel('t');
        ylabel('log(m(point + t * direction)) (shifted)');
        title('2D Plot of Symbolic Function m along a Direction (Log Scale, Shifted)');
        grid on;
        hold on;
        
        initial_point_m = im + shift_m;
        
        final_point_m = fm + shift_m;
        plot(t_current, final_point_m, 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Final Point');
        text(t_current, final_point_m, sprintf('(%g, %g)', t_current, fm), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        plot(0, initial_point_m, 'mo', 'MarkerFaceColor', 'm', 'DisplayName', 'Initial Point');
        text(0, initial_point_m, sprintf('(%g, %g)', 0, im), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        % Draw an arrow from the initial point to the final point
        annotation('arrow', [0.25, 0.75], [0.5, 0.5]); % Adjust the positions accordingly
        legend show;
        hold off;

        prev_x = x;
        
        waitfor(gcf);

        history(end + 1, :) = x;
        
    end

    result = x; 
end