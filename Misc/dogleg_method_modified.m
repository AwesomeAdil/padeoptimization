function result = dogleg_method_modified(g, H, delta)
        H = make_positive_definite(H);
        
        % Compute the Cauchy point
        p_u = - (g' * g) / (g' * H * g) * g;
        
        % Compute the Newton step
        p_b = - H \ g;
        
        % Dogleg path
        if norm(p_b) <= delta
            p_dl = p_b;
        elseif norm(p_u) >= delta
            p_dl = delta * p_u / norm(p_u);
        else
            if norm(p_u) >= delta
                p_dl = p_u / norm(p_u);
            else
                p_diff = p_b - p_u;
                p_diff_norm_sq = p_diff.dot(p_diff);
                p_w = 2*p_u - p_b;

                a = p_diff_norm_sq;
                b = 2 * p_w;
                c = norm(p_w)^2 - delta^2;
    
                discriminant = b^2 - 4 * a * c;
                t = (-b + sqrt(discriminant))/2*a;

                if t <= 1
                    p_dl = t * p_u;
                else
                    p_dl = p_u + t*p_diff;
                end
            end
        end
        result = p_dl;
end