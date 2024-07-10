function result = boxplotdata(tay_steps, pad_steps, tay_values, pad_values, tay_times, pad_times, Name, num_trials)    
    mean_tay_steps = mean(tay_steps);
    mean_pad_steps = mean(pad_steps);
    mean_tay_values = mean(tay_values);
    mean_pad_values = mean(pad_values);
    mean_tay_times = mean(tay_times);
    mean_pad_times = mean(pad_times);
    
    % Calculate standard errors
    se_tay_steps = std(tay_steps) / sqrt(num_trials);
    se_pad_steps = std(pad_steps) / sqrt(num_trials);
    se_tay_values = std(tay_values) / sqrt(num_trials);
    se_pad_values = std(pad_values) / sqrt(num_trials);
    se_tay_times = std(tay_times) / sqrt(num_trials);
    se_pad_times = std(pad_times) / sqrt(num_trials);
    
    % Calculate differences
    diff_steps = tay_steps - pad_steps;
    diff_values = tay_values - pad_values;
    diff_times = tay_times - pad_times;
    
    % Create a figure
    fig = figure;
    set(fig, 'Name', Name, 'NumberTitle', 'off', 'Visible', 'on');
    
    % Plot box plots for the number of steps
    subplot(3, 2, 1);
    h = boxplot([tay_steps, pad_steps], 'Labels', {'Taylor Steps', 'Padé Steps'});
    set(h, 'LineWidth', 2);
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0

    title('Box Plot of Number of Steps to Convergence');
    ylabel('Number of Steps');
    
    % Plot box plots for the function values
    subplot(3, 2, 2);
    h = boxplot([tay_values, pad_values], 'Labels', {'Taylor Values', 'Padé Values'});
    set(h, 'LineWidth', 2);
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    title('Box Plot of Function Values at Convergence');
    ylabel('Function Value');
    
    % Plot box plots for the running times
    subplot(3, 2, 3);
    h = boxplot([tay_times, pad_times], 'Labels', {'Taylor Times', 'Padé Times'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Running Times');
    ylabel('Time (s)');
    
    % Plot mean number of steps
    subplot(3, 2, 4);
    errorbar([1, 2], [mean_tay_steps, mean_pad_steps], [se_tay_steps, se_pad_steps], 'o-', 'LineWidth', 2);
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    xticks([1, 2]);
    xticklabels({'Taylor Steps', 'Padé Steps'});
    title('Mean Number of Steps to Convergence');
    ylabel('Mean Number of Steps');
    grid on;
    
    % Plot mean function values
    subplot(3, 2, 5);
    errorbar([1, 2], [mean_tay_values, mean_pad_values], [se_tay_values, se_pad_values], 'o-', 'LineWidth', 2);
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    xticks([1, 2]);
    xticklabels({'Taylor Values', 'Padé Values'});
    title('Mean Function Values at Convergence');
    ylabel('Mean Function Value');
    grid on;
    
    % Plot mean running times
    subplot(3, 2, 6);
    errorbar([1, 2], [mean_tay_times, mean_pad_times], [se_tay_times, se_pad_times], 'o-', 'LineWidth', 2);
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    xticks([1, 2]);
    xticklabels({'Taylor Times', 'Padé Times'});
    title('Mean Running Times');
    ylabel('Mean Time (s)');
    grid on;
    
    % Adjust layout
    sgtitle(['Convergence Analysis Apples: ', Name]);
    savefig(['Apple Comparisons/', Name,'.fig']);

    % Create a figure for differences
    fig_diff = figure;
    set(fig_diff, 'Name', [Name, ' Differences'], 'NumberTitle', 'off', 'Visible', 'on');
    
    % Create a tiled layout with 3 rows and 1 column
    tiledlayout(3, 1, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % Plot box plots for the differences in number of steps
    nexttile;
    h = boxplot(diff_steps, 'Labels', {'Steps Difference'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Differences in Number of Steps');
    ylabel('Steps Difference');
    hold on;
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    hold off;
    
    % Plot box plots for the differences in function values
    nexttile;
    h = boxplot(diff_values, 'Labels', {'Values Difference'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Differences in Function Values');
    ylabel('Values Difference');
    hold on;
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    hold off;
    
    % Plot box plots for the differences in running times
    nexttile;
    h = boxplot(diff_times, 'Labels', {'Times Difference'});
    set(h, 'LineWidth', 2);
    title('Box Plot of Differences in Running Times');
    ylabel('Times Difference');
    hold on;
    yline(0, 'k--', 'LineWidth', 1); % Add a horizontal line at 0
    hold off;
    sgtitle(['Differences Analysis: ',  Name]);
    savefig(['diffs/Diff_', Name,'.fig']);
    end