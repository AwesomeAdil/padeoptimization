function [result, history] = trust_region_2d_plot(f, x_sym, y_sym, x0, y0, delta, max_iter, tol, hat, thresh)
    m = f;
    x = x0;
    y = y0;
    history = [x0, y0];
    % Generate a grid for plotting the model function
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = vpa(subs(f, {x_sym, y_sym}, {x, y}));

    for ind = 0:max_iter
        m = nested_pade(vpa(f), {x_sym, y_sym}, {x, y}, 2);
        f_handle = matlabFunction(m, 'Vars', [x_sym, y_sym]);
        initial_guess = [x, y];
        objective = @(v) f_handle(v(1), v(2));
       
        lb = [x-delta; y-delta];  % Lower bounds
        ub = [x+delta; y+delta];  % Upper bounds
      
        [x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);

        x_new = x_opt(1);
        y_new = x_opt(2);

        num = vpa(subs(f, {x_sym, y_sym}, {x, y})) - vpa(subs(f, {x_sym, y_sym}, {x_new, y_new}));
        den = old_fval - fval;

        old_fval = fval;

        if abs(num) < 1e-9 && abs(den) < 1e-9
            result = [x, y];
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
            if ratio > 0.75 && delta == max(abs(x - x_new), abs(y - y_new))
                delta = min(2 * delta, hat);
            end
        end

        if max(abs(x - x_new), abs(y - y_new)) < double(tol)
           result = [x, y];
           return;
        end

        if ratio > thresh
            x = x_new;
            y = y_new;
        end

        history(end + 1, :) = [x, y];
    end

    result = [x, y]; 
end