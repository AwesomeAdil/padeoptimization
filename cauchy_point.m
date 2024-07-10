function result = cauchy_point(grad, hess, delta)
    gBg = grad' * hess * grad;
    tau = 1;
    if gBg > 0
        tau = min(1, (sqrt(sum(grad.^2))^3)/(delta*gBg));
    end
    result = -tau * delta * grad / sqrt(sum(grad.^2));
end