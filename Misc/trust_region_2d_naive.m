function [result, history] = trust_region_2d_naive(f, x_sym, y_sym, x0, y0, delta, n, max_iter, tol, method, hat, thresh)
    m = f;
    x = x0;
    y = y0;
    history = [x0, y0];

    for ind = 0:max_iter
        m = taylor(f, [x_sym, y_sym] ,'Order', n+1, 'ExpansionPoint', [x, y]);
        if method == 'pad'
            %[coefs, terms] = coeffs(m, [x_sym y_sym]);
            %simplified_terms = arrayfun(@(term) simplify(subs(term, [(x_sym-1), (y_sym-1)], [x_sym, y_sym])), terms, 'UniformOutput', false);
            %digits(10);
            %m = vpa(sum(coefs .* [simplified_terms{:}]));
            m = two_D_pade_one(x_sym, y_sym, m);
        end

        g = vpa(subs(gradient(f, [x_sym, y_sym]), {x_sym, y_sym}, {x, y}));
        B = vpa(subs(hessian(f, [x_sym, y_sym]), {x_sym, y_sym}, {x, y}));
        tole = 1e-9; % or any other numerical value you prefer
        if sqrt(sum(g.^2)) < double(tole)
            result = [x, y];
            return;
        end
        
        step = cauchy_point(g, B, delta);
        x_new = x + step(1);
        y_new = y + step(2);
        
        
        num = vpa(subs(f, {x_sym, y_sym}, {x, y})) - vpa(subs(f, {x_sym, y_sym}, {x_new, y_new}));
        den = vpa(subs(m, {x_sym, y_sym}, {x, y})) - vpa(subs(m, {x_sym, y_sym}, {x_new, y_new}));
        a = vpa(subs(m, {x_sym, y_sym}, {x, y}));
        b = vpa(subs(m, {x_sym, y_sym}, {x_new, y_new}));

        
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
            if ratio > 0.75 && delta == sqrt(sum((step).^2))
                delta = min(2 * delta, hat);
            end
        end

        if sqrt(sum(step.^2)) < double(tol)
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
