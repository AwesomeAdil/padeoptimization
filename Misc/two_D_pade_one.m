function res = two_D_pade_one(x_sym, y_sym, taylor)
    terms = children(taylor);
    myDictionary = containers.Map;

    for i = 1:length(terms)
        term = terms(i);
        [c, t] = coeffs(term, [x_sym, y_sym]); % Get coefficients and corresponding terms
        key_str = char(t); % Convert symbolic term to string
        myDictionary(key_str) = c;
    end
    
    terms = {'1', char(x_sym), char(y_sym), [char(x_sym) '*' char(y_sym)], [char(x_sym) '^2'], [char(y_sym) '^2']};
    
    c = zeros(1, 6);
    for i = 1:length(c)
        if isKey(myDictionary, terms{i})
            c(i) = myDictionary(terms{i});
        else
            c(i) = 0;
        end
    end
    a3 = 0;
    % Calculate b1 and b2
    if c(2) == 0 || c(3) == 0
        res = x_sym^2;
        return;
    end

    b1 = -c(5)/c(2);
    b2 = -c(6)/c(3);

    a3 = c(3) + c(2)*b2 + c(1)*b1;

    % Calculate the Pade approximant
    a0 = c(1);
    a1 = c(2) + b1*c(1);
    a2 = c(3) + b2*c(1);

    digits(5);
    res = (a0 + a1*x_sym + a2*y_sym + a3*x_sym*y_sym) / (1 + b1*x_sym + b2*y_sym);
end
