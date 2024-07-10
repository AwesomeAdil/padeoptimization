function result = nested_pade(expr, vars, xv, n_pade)
    expr = vpa(expr);
    result = pade(expr, vars{1,1}, 'Order', n_pade, 'ExpansionPoint', xv(1));
    
    if isscalar(vars)
        return
    end
    
    [n, d] = numden(result);
    n = expand(n);
    d = expand(d);
    
    [num_coeffs, num_terms] = coeffs(n, vars{1});
    [den_coeffs, den_terms] = coeffs(d, vars{1});
    
    num_expr = sym(0);
    den_expr = sym(0);
    
    for i = 1:length(num_coeffs)
        num_expr = num_expr + num_terms(i) * nested_pade(num_coeffs(i), vars(2:end), xv(2:end), n_pade);      
    end

    % disp("NUM");
    % num_expr = simplify(num_expr);
    % pretty(num_expr);

    for i = 1:length(den_coeffs)
        den_expr = den_expr + den_terms(i) * nested_pade(den_coeffs(i), vars(2:end), xv(2:end), n_pade);
    end

    % disp("DEN");
    % pretty(den_expr);
    
    result = simplify(num_expr / den_expr);
    %disp("RES");
    %pretty(result);
end