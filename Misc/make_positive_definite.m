function A_posdef = make_positive_definite(A)
    [V, D] = eig(A);
    D(D < 1e-6) = 1e-6; % Set negative eigenvalues to epsilon
    A_posdef = V * D * V';
end