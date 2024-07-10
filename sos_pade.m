function [result, history] = sos_pade(f, x_sym, y_sym, x0, y0, delta, max_iter, tol, hat, thresh)
    x = x0;
    y = y0;
    history = [x0, y0];
    options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'final');

    old_fval = double(subs(f, {x_sym, y_sym}, {x, y}));

    for ind = 0:max_iter
        % Define the 2D rational function using the provided formula
        m = (x_sym.^2 + y_sym.^2 - 2*x_sym - 4*y_sym + 4) ./ (x_sym.^2 + y_sym.^2 + 1);

        % Create SOS relaxation problem
        sdpvar xs ys;
        g = m;
        type(g)
        F = [sos(g)];
        options = sdpsettings('solver', 'sedumi', 'verbose', 0);
        diagnostics = solvesos(F, g, options, [xs ys]);

        if diagnostics.problem == 0
            % Extract the optimized variables
            x_opt = value(xs);
            y_opt = value(ys);
            fval = value(g);
        else
            error('SOS optimization failed');
        end

        x_new = x_opt;
        y_new = y_opt;

        num = double(subs(f, {x_sym, y_sym}, {x, y})) - double(subs(f, {x_sym, y_sym}, {x_new, y_new}));
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
            ratio = abs(num / den);
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
            old_x = x; % Store the old x
            old_y = y; % Store the old y
            x = x_new;
            y = y_new;
        end

        history(end + 1, :) = [x, y];

        % Plot the original function
        figure;
        subplot(1, 2, 1);
        hold on;
        grid on;
        [X, Y] = meshgrid(linspace(x-3, x+3, 100), linspace(y-3, y+3, 100));
        Z_orig = double(subs(f, {x_sym, y_sym}, {X, Y}));
        surf(X, Y, Z_orig);
        shading interp;
        colormap jet;
        colorbar;
        alpha 0.5;
        title('Original Function');
        
        % Plot the old and new points on the original function
        z_val_orig_old = double(subs(f, {x_sym, y_sym}, {old_x, old_y}));
        z_val_orig_new = double(subs(f, {x_sym, y_sym}, {x, y}));
        plot3(old_x, old_y, z_val_orig_old, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
        text(x, y, z_val_orig_new, sprintf('(%0.2f, %0.2f, %0.2f)', x, y, z_val_orig_old), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
       
        plot3(x, y, z_val_orig_new, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
        plot3([old_x, x], [old_y, y], [z_val_orig_old, z_val_orig_new], 'k--', 'LineWidth', 1);
        line([x, x], [y, y], [0, z_val_orig_new], 'Color', 'r', 'LineStyle', '--');
        text(x, y, z_val_orig_new, sprintf('(%0.2f, %0.2f, %0.2f)', x, y, z_val_orig_new), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        
        % Plot the rational function approximation
        subplot(1, 2, 2);
        hold on;
        grid on;
        Z_model = double(subs(m, {x_sym, y_sym}, {X, Y}));
        surf(X, Y, Z_model);
        shading interp;
        colormap jet;
        colorbar;
        alpha 0.5;
        title('Rational Function Approximation');
        
        % Plot the old and new points on the rational function approximation
        z_val_model_old = double(subs(m, {x_sym, y_sym}, {old_x, old_y}));
        z_val_model_new = double(subs(m, {x_sym, y_sym}, {x, y}));
        plot3(old_x, old_y, z_val_model_old, 'bo', 'MarkerFaceColor', 'b', 'MarkerSize', 8);
        plot3(x, y, fval, 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 8);
        plot3([old_x, x], [old_y, y], [z_val_model_old, z_val_model_new], 'k--', 'LineWidth', 1);
        line([x, x], [y, y], [0, z_val_model_new], 'Color', 'r', 'LineStyle', '--');
        text(x, y, z_val_model_new, sprintf('(%0.2f, %0.2f, %0.2f)', x, y, z_val_model_new), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        text(old_x, old_y, z_val_model_old, sprintf('(%0.2f, %0.2f, %0.2f)', old_x, old_y, z_val_model_old), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
        
        % Enable 3D rotation
        rotate3d on;
        
        savefig(['Easom/Easom', num2str(ind),'.fig']);
        % Wait for the user to close the figure
        waitfor(gcf);
    end

    result = [x, y]; 
end