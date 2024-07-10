function [result, history] = trust_region_nd(f, xs, x0, delta, max_iter, tol, hat, thresh)
    m = f;
    x = x0;
    history = x0;
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'none');

    old_fval = vpa(subs(f, xs, x));
    syms t;
    
    prev_x = x;
    
    for ind = 0:max_iter
        m = nested_pade(vpa(f), num2cell(xs), x, 2);
        f_handle = matlabFunction(m, 'Vars', xs);
        initial_guess = x;
        objective = @(v) f_handle(v(1), v(2));
        
        
        lb = x - delta * ones(size(x));  % Ensure the length matches x
        ub = x + delta * ones(size(x));  % Ensure the length matches x 


        [x_opt, fval] = fmincon(objective, initial_guess, [], [], [], [], lb, ub, [], options);
        
        ip = vpa(subs(f, xs, prev_x));
        fp = vpa(subs(f, xs, x_opt));
        im = vpa(subs(m, xs, prev_x));
        fm = vpa(subs(m, xs, x_opt));

        num = ip - fp;
        den = im - fm;

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
        
    end

    result = x; 
end