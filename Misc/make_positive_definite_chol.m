function A_posdef = make_positive_definite_chol(A)
    epsilon = 1e-6; % Small positive value
    while true
        try
            chol(A);
            A_posdef = A;
            return;
        catch
            A = A + epsilon * eye(size(A));
        end
    end
end
