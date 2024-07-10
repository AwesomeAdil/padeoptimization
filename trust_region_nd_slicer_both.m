function [result, history] = trust_region_nd_slicer_both(f, xs, x0, delta, max_iter, tol, hat, thresh)
    x = x0;
    history = x0;
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = vpa(subs(f, xs, x));
    syms t;
    
    prev_x = x;
    
    for ind = 0:max_iter
        %% Subproblem
        m1 = nested_pade(vpa(f), num2cell(xs), x, 2);
        f_handle = matlabFunction(m1, 'Vars', xs);
        initial_guess = x;
        objective = @(v) f_handle(v(1), v(2));
       
        lb = x - delta * ones(length(x));  
        ub = x + delta * ones(length(x));  

        [x_opt1, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
        
        direction1 = x_opt1 - x; 
        direction1 = direction1 / norm(direction1);

        % Ensure xs is a symbolic vector and x is a numeric vector
        if ~isa(xs, 'sym')
            xs = sym(xs);
        end
        
        % Compute the Taylor expansion at the current point x
        m2 = taylor(vpa(f), xs, 'Order', 3, 'ExpansionPoint', num2cell(x));
        g = vpa(subs(gradient(f, xs), xs, num2cell(x)));
        B = vpa(subs(hessian(f, xs), xs, num2cell(x)));
        tole = 1e-9; % or any other numerical value you prefer
        
        if sqrt(sum(g.^2)) < double(tole)
            result = x;
            return;
        end
        
        step = double(cauchy_point(g, B, delta));
      
        x_opt2 = x + step';
        direction2 = x_opt2 - x; 
        direction2 = direction2 / norm(direction2);

        %% Ratio of Reductions
        ip = vpa(subs(f, xs, num2cell(prev_x)));
        fp1 = vpa(subs(f, xs, num2cell(x_opt1)));
        fp2 = vpa(subs(f, xs, num2cell(x_opt2)));
        im1 = vpa(subs(m1, xs, num2cell(prev_x)));
        fm1 = vpa(subs(m1, xs, num2cell(x_opt1)));
        im2 = vpa(subs(m2, xs, num2cell(prev_x)));
        fm2 = vpa(subs(m2, xs, num2cell(x_opt2)));

        num = ip - fp1;
        den = im1 - fm1;

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
            if ratio > 0.75 && delta == norm(x_opt1 - x, inf)
                delta = min(2 * delta, hat);
            end
        end
        
        if norm(x_opt1 - x, inf) < double(tol)
           result = x;
           return;
        end

        if ratio > thresh
            x = x_opt1;
        end

        %% Graphing
        t_values = linspace(-5, 5, 100);

        % First set of calculations
        param_func_f1 = subs(f, xs, prev_x + t * direction1);
        param_func_m1 = subs(m1, xs, prev_x + t * direction1);
        t_current1 = norm(x_opt1 - prev_x) / norm(direction1);
        
        y_values_f1 = double(subs(param_func_f1, t, t_values));  
        y_values_m1 = double(subs(param_func_m1, t, t_values));  
        
        shift_f1 = abs(min(y_values_f1)) + 1;
        shift_m1 = abs(min(y_values_m1)) + 1;
        y_values_f_shifted1 = y_values_f1 + shift_f1;
        y_values_m_shifted1 = y_values_m1 + shift_m1;
        
        % Second set of calculations
        param_func_f2 = subs(f, xs, prev_x + t * direction2);
        param_func_m2 = subs(m2, xs, prev_x + t * direction2);
        t_current2 = norm(x_opt2 - prev_x) / norm(direction2);
        
        y_values_f2 = double(subs(param_func_f2, t, t_values));  
        y_values_m2 = double(subs(param_func_m2, t, t_values));  
        
        shift_f2 = abs(min(y_values_f2)) + 1;
        shift_m2 = abs(min(y_values_m2)) + 1;
        y_values_f_shifted2 = y_values_f2 + shift_f2;
        y_values_m_shifted2 = y_values_m2 + shift_m2;
        
        figure;
        
        % First function, first direction
        subplot(2, 2, 1);
        semilogy(t_values, y_values_f_shifted1, 'LineWidth', 2);
        xlabel('t');
        ylabel('log(f(point + t * direction1)) (shifted)');
        [ta,~] = title('Function f along Direction Pade', ' ');
        ta.FontSize = 16;
        grid on;
        hold on;
        initial_point_f1 = ip + shift_f1;
        final_point_f1 = vpa(subs(f, xs, num2cell(x_opt1))) + shift_f1;
        plot(0, initial_point_f1, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Initial Point');
        text(0, initial_point_f1, sprintf('(%g, %g)', 0, ip), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        plot(t_current1, final_point_f1, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Final Point');
        text(t_current1, final_point_f1, sprintf('(%g, %g)', t_current1, fp1), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        legend show;
        hold off;
        
        % First model, first direction
        subplot(2, 2, 2);
        semilogy(t_values, y_values_m_shifted1, 'LineWidth', 2);
        xlabel('t');
        ylabel('log(m(point + t * direction1)) (shifted)');
        [ta,~] = title('Model Pade along Direction Pade', ' ');
        ta.FontSize = 16;
        grid on;
        hold on;
        initial_point_m1 = im1 + shift_m1;
        final_point_m1 = fm1 + shift_m1;
        plot(0, initial_point_m1, 'mo', 'MarkerFaceColor', 'm', 'DisplayName', 'Initial Point');
        text(0, initial_point_m1, sprintf('(%g, %g)', 0, im1), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        plot(t_current1, final_point_m1, 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Final Point');
        text(t_current1, final_point_m1, sprintf('(%g, %g)', t_current1, fm1), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        legend show;
        hold off;
        
        % Second function, second direction
        subplot(2, 2, 3);
        semilogy(t_values, y_values_f_shifted2, 'LineWidth', 2);
        xlabel('t');
        ylabel('log(f(point + t * direction2)) (shifted)');
        [ta,~] = title('Function f along Direction Taylor', ' ');
        ta.FontSize = 16;
        grid on;
        hold on;
        initial_point_f2 = ip + shift_f2;
        final_point_f2 = vpa(subs(f, xs, num2cell(x_opt2))) + shift_f2;
        plot(0, initial_point_f2, 'go', 'MarkerFaceColor', 'g', 'DisplayName', 'Initial Point');
        text(0, initial_point_f2, sprintf('(%g, %g)', 0, ip), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        plot(t_current2, final_point_f2, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Final Point');
        text(t_current2, final_point_f2, sprintf('(%g, %g)', t_current2, fp2), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        legend show;
        hold off;
        
        % Second model, second direction
        subplot(2, 2, 4);
        semilogy(t_values, y_values_m_shifted2, 'LineWidth', 2);
        xlabel('t');
        ylabel('log(m(point + t * direction2)) (shifted)');
        [ta,~] = title('Model Taylor along Direction Taylor', ' ');
        ta.FontSize = 16;
        grid on;
        hold on;
        initial_point_m2 = im2 + shift_m2;
        final_point_m2 = fm2 + shift_m2;
        plot(0, initial_point_m2, 'mo', 'MarkerFaceColor', 'm', 'DisplayName', 'Initial Point');
        text(0, initial_point_m2, sprintf('(%g, %g)', 0, im2), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        plot(t_current2, final_point_m2, 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Final Point');
        text(t_current2, final_point_m2, sprintf('(%g, %g)', t_current2, fm2), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
        legend show;
        hold off;
        
        prev_x = x;
        
        waitfor(gcf);
        history(end + 1, :) = x;
    end

    result = x; 
end