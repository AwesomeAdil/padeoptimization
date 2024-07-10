function [result, history] = trust_region(f, x_sym, x0, delta, n, max_iter, tol, method, hat, thresh)
    m = f;
    x = x0;
    history = [x0];
    for ind = 0:max_iter
        if method == 'tay'
            m = taylor(f, 'Order', n+1, 'ExpansionPoint', x);
        elseif method == 'pad'
            m = pade(f, 'Order', n/2, 'ExpansionPoint', x);
        end

        disp(class(m));
        g = vpa(subs(gradient(f, x_sym), x_sym , x));
        B = vpa(subs(hessian(f, x_sym), x_sym, x));
        if sqrt(sum(g.^2)) < 1e-9
            result = x;
            re = "A";
            return;
        end
        x_new = x + cauchy_point(g, B, delta);
        num = vpa(subs(f, x_sym, x)) - vpa(subs(f, x_sym, x_new));
        den = vpa(subs(m, x_sym, x)) - vpa(subs(m, x_sym, x_new));
        
        if abs(num) < 1e-9 && abs(den) < 1e-9
            result = x;
            return;
        end

        if abs(den)<1e-9
            ratio = (0.25+thresh)/2;
        else
            ratio = num/den;
        end

        if ratio < 0.25
            delta = delta/4;
        else
            if ratio > 0.75 && delta == sqrt(sum((x_new - x0).^2))
                delta = min(2*delta, hat);
            end
        end
        
        if sqrt(sum((x_new - x).^2)) < tol
           result = x;
           return;
        end

        if ratio > thresh
            x = x_new;
        end

        history(end+1) = x;
    end            
    result = x; 
end