% Specify the directory containing .fig files
figDir = 'diffs';
outputDir = ['differences_img'];

% Get a list of all .fig files in the directory
figFiles = dir(fullfile(figDir, '*.fig'));

% Loop through each .fig file and save as .jpg
for i = 1:length(figFiles)
    % Construct the full file name and open the .fig file
    figFile = fullfile(figDir, figFiles(i).name);
    openfig(figFile, 'reuse', 'invisible');
    
    % Construct the output file name
    [~, name, ~] = fileparts(figFiles(i).name);
    outputFile = fullfile(outputDir, [name, '.jpg']);
    
    % Save the figure as .jpg
    saveas(gcf, outputFile);
    
    % Close the figure
    close(gcf);
end

disp('Conversion complete.');