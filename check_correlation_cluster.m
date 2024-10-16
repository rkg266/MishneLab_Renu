clc;
clear;
close all;

% Directory containing the FMRI data files
dataDir = 'D:\UCSD_Acads\ProfGal_Research\data32\fMRIData\REST';
% All sessions in the directory
sesList = dir(fullfile(dataDir, 'session-*'));
% Sesion-wise data files paths
ses_paths = struct();
for i = 1:length(sesList)
    ses_name = sesList(i).name;
    ses_name1 = strrep(ses_name, '-', '_');
    ses_paths.(ses_name1) = fullfile(dataDir, ses_name);
end

RUN_ACROSSSESSION = 0;
RUN_EVENODD = 1;

if (RUN_ACROSSSESSION)
    % Directory containing LSSC results
    resDir = 'D:\UCSD_Acads\ProfGal_Research\test_run4\run_fmri_sessions';
    % All files in the directory
    resList = dir(fullfile(resDir, '*.mat'));
    
    corr_results = struct();
    % Run through all LSSC result files
    for i = 1:length(resList)
        fname = resList(i).name;
        fpath = fullfile(resDir, fname);
    
        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
        %run_match = regexp(fname, 'run_(\d+)', 'tokens');
        sub_value = sub_match{1}{1};
        ses_value = ses_match{1}{1};
        %run_value = run_match{1}{1};
        
        pResult = load(fpath);
        THR_ID = 1; % for now there is only one threshold
        labelMatrix = pResult.labels{THR_ID};
    
        % Find unique non-zero labels in the matrix
        uniqueLabels = unique(labelMatrix);
        uniqueLabels(uniqueLabels == 0) = [];  % Remove the zero entry
        
        % Initialize a cell array to store clusters
        clusters = cell(length(uniqueLabels), 1);
        
        % Loop through each unique label
        for j = 1:length(uniqueLabels)
            label = uniqueLabels(j);
            
            % Create a binary mask for the current label
            binaryMask = (labelMatrix == label);
            
            % Find connected components (clusters) in the binary mask
            cc = bwconncomp(binaryMask);
            
            % Store the pixel locations for each cluster
            clusters{j} = cc.PixelIdxList{1};
        end
    
        % Cluster wise temporal correlation
        cur_ses = ['session_' num2str(ses_value)];
        dataFiles = dir(fullfile(ses_paths.(cur_ses), '*.nii.gz'));
        for j1 = 1:length(dataFiles)
            dfname = dataFiles(j1).name;
            if contains(dfname, sub_value) % data file of the current subject found
                dfPath = fullfile(ses_paths.(cur_ses), dfname);
                fprintf('Processing file: %s\n', dfname);
                
                sub_match = regexp(dfname, 'sub-(\w{5})', 'tokens');
                ses_match = regexp(dfname, 'ses-(\d+)', 'tokens');
                run_match = regexp(dfname, 'run-(\d+)', 'tokens');
                sub_value = sub_match{1}{1};
                ses_value = str2double(ses_match{1}{1});
                run_value = str2double(run_match{1}{1});
        
                V = niftiread(dfPath);
                V=squeeze(V);
                [R, C, T] = size(V);
                
                % Compute correlation for each cluster
                clusterwise_corr = zeros(1, length(clusters));
                for j2 = 1:length(clusters)
                    pixelList = clusters{j2};
    
                    % Extract time series for each pixel
                    timeseries = zeros(T, length(pixelList));
                    for j3 = 1:length(pixelList)
                        [x, y] = ind2sub([R, C], pixelList(j3));
                        timeseries(:, j3) = squeeze(V(x, y, :));
                    end
                    
                    % Pairwise correlation matrix
                    correlationMatrix = corr(timeseries);
    
                    % Compute average correlation value
                    upT = triu(correlationMatrix, 1);
                    nzupT = upT(upT ~= 0);
                    clusterwise_corr(j2) = mean(nzupT);
    
                    % Loop over each pixel to create a heatmap for its correlation with all other pixels
                    % for j4 = 1:length(pixelList)
                    %     ImgMat = zeros(R, C);
                    %     [x, y] = ind2sub([R, C], pixelList(j4));
                    %     ImgMat(x, y) = 1;
                    %     for j5 = 1:length(pixelList)
                    %         [x1, y1] = ind2sub([R, C], pixelList(j5));
                    %         ImgMat(x1, y1) = correlationMatrix(j4, j5);
                    %     end
                    % 
                    %     imagesc(ImgMat, [0 1]);
                    %     %caxis([0 1]);
                    %     colorbar;
                    %     colormap('jet');
                    %     drawnow;
                    %     pause(0.1);
                    % end
    
                end
    
                if ~isfield(corr_results, fname(1:end-4))
                    corr_results.(fname(1:end-4)) = {clusterwise_corr};
                else
                    corr_results.(fname(1:end-4)){end+1} = clusterwise_corr;
                end
    
            end
        end
        bh=9;
    end
end



%% Even-Odd case
if (RUN_EVENODD)
    % Directory containing LSSC results
    resDir = 'D:\UCSD_Acads\ProfGal_Research\test_run4\run_fmri_evenodd\lssc_processed';
    % All files in the directory
    resList = dir(fullfile(resDir, '*.mat'));

    FILEID = 1;
    fname = resList(FILEID+2).name; % run 19
    fpath = fullfile(resDir, fname);

    sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
    ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
    run_match = regexp(fname, 'run_(\d+)', 'tokens');
    sub_value = sub_match{1}{1};
    ses_value = ses_match{1}{1};
    run_value = run_match{1}{1};

    pResult = load(fpath);
    THR_ID = 1; % for now there is only one threshold
    labelEven = pResult.even.labels{THR_ID};
    labelOdd = pResult.even.labels{THR_ID};

    % Cluster matching
    pairing = pairComponents(pResult.even.mergedA{1}, pResult.odd.mergedA{1});
    p_1 = pairing.p1;
    p_2 = pairing.p2;
    
    % Re-labeling labels for 2nd one
    for k1=1:size(labelOdd, 1)
        for k2=1:size(labelOdd, 2)
            if (labelOdd(k1, k2) ~= 0)
                tp = find(p_2 == labelOdd(k1, k2));
                if (isempty(tp))
                    labelOdd(k1, k2) = 0;
                else
                    labelOdd(k1, k2) = p_1(tp);
                end
                
            end
        end
    end

    % for now just temporal correlation between even and odd frames
    cur_ses = ['session_' num2str(ses_value)];
    dataFiles = dir(fullfile(ses_paths.(cur_ses), '*.nii.gz'));
    for j1 = 1:length(dataFiles)
        dfname = dataFiles(j1).name;
        dfPath = fullfile(ses_paths.(cur_ses), dfname);
        if (contains(dfname, sub_value) && contains(dfname, ['ses-' ses_value]) && contains(dfname, ['run-' run_value]))
            V = niftiread(dfPath);
            V=squeeze(V);
            %maxV = (max(V(:,:,1,:),[],3));
        
            num_time_samples = size(V, 3);
        
            % Create even and odd time point datasets
            V_even = V(:, :, 2:2:num_time_samples); % Data for even time points
            V_odd = V(:, :, 1:2:num_time_samples);  % Data for odd time points
            
            % Create brain masks for even and odd time points
            maxV_even = max(V_even, [], 3);
            brain_mask_even = maxV_even ~= 0;
            brain_mask_even(29:32,:) = 0;
            maxV_odd = max(V_odd, [], 3);
            brain_mask_odd = maxV_odd ~= 0;
            brain_mask_odd(29:32,:) = 0;
        
            [R,C] = size(brain_mask_even);
        
            % Flatten the spatial dimensions
            V_even_flat = reshape(V_even, [], size(V_even, 3));
            V_odd_flat = reshape(V_odd, [], size(V_odd, 3));
            
            % Extract relevant pixels for even and odd datasets
            allregionspix_even = find(brain_mask_even);
            allregionspix_odd = find(brain_mask_odd);

            numRelevantPixels = length(allregionspix_even);
            dataEven = zeros(numRelevantPixels, size(V_even_flat, 2));
            dataOdd = zeros(numRelevantPixels, size(V_odd_flat, 2));
            
            crossCorrelations = zeros(numRelevantPixels, 1);
            for i = 1:numRelevantPixels
                dataEven(i, :) = V_even_flat(allregionspix_even(i), :);
                dataOdd(i, :) = V_odd_flat(allregionspix_odd(i), :);

                min_len = min(length(dataEven(i, :)), length(dataOdd(i, :)));
                crossCorrelations(i) = corr(dataEven(i, 1:min_len)', dataOdd(i, 1:min_len)');

            end


            break;
        end
    end
end