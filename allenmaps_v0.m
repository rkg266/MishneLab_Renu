clc;
clear; 
close all;

% Maps directory
mapDir = 'D:\UCSD_Acads\ProfGal_Research\allen maps';

% Directory containing the fMRI data
dataDir = 'D:\UCSD_Acads\ProfGal_Research\data32\fMRIData\REST';

% Store results in this directory
resultDir = 'D:\UCSD_Acads\ProfGal_Research\test_run_Allen_fullbrain';

AllenfilePath = fullfile(mapDir, '2D_calcium_atlas.nii');

allen_map = niftiread(AllenfilePath);

% All sessions in the directory
sesList = dir(fullfile(dataDir, 'session-*'));

AllenCorrStorePath = fullfile(resultDir, 'AllenTempCorr');
RUN_ALLEN_CORR_PROCESSING = 0;
RUN_ALLEN_CORR_REPORT = 1;

if (RUN_ALLEN_CORR_PROCESSING)
    for ses = 1:length(sesList) % processing each session
        sesname = sesList(ses).name;
        sesDir = [dataDir, '\', sesname];
        fileList = dir(fullfile(sesDir, '*.nii.gz'));
        
        V = []; % To hold the time series data - 3D matrix
        prev_sub = '';
        prev_ses = '';
        for i = 1:length(fileList) % processing each .nii file
            fname = fileList(i).name;
            filePath = fullfile(sesDir, fname);
        
            sub_match = regexp(fname, 'sub-(\w{5})', 'tokens');
            ses_match = regexp(fname, 'ses-(\d+)', 'tokens');
            run_match = regexp(fname, 'run-(\d+)', 'tokens');
            sub_value = sub_match{1}{1};
            ses_value = str2double(ses_match{1}{1});
            run_value = str2double(run_match{1}{1});
            
            V0 = niftiread(filePath);
            V0 = squeeze(V0);
        
            if ((i == 1 || strcmp(prev_sub, sub_value)) && i ~= length(fileList))    % runs belong to same subject
                if isempty(V)
                    V = V0;
                else
                    V = cat(3, V, V0); % concatenate runs
                end 
            else      
                % Process the concatenated V matrix and at the end initialize it to next subject's data
                if (i == length(fileList)) % if last file
                    prev_sub = sub_value;
                    prev_ses = ses_value;
                end
                % V matrix holds the data of 'prev_sub', 'prev_ses'
                % concatenated over runs

                % Run Allen parcellation for the current data (V)
                num_time_samples = size(V, 3);
        
                % Create brain mask
                maxV = max(V, [], 3);
                brain_mask = maxV ~= 0;
                brain_mask(29:32,:) = 0; 
                [R,C] = size(brain_mask);
            
                % Flatten the spatial dimensions
                V_flat = reshape(V, [], size(V, 3));
                
                % Extract relevant pixels
                allregionspix = find(brain_mask);
                dFoF_masked = V_flat(allregionspix, :);
    
                % Resize the Allen map to the size of the brain image using nearest-neighbor interpolation
                allenMapResized = imresize(allen_map, [size(V, 1), size(V, 2)], 'nearest');
                allenMapResized = allenMapResized.*brain_mask; % apply the brain_mask (removes midline)
                
                % Find temporal correlation for allen clusters
                clusterIDs = unique(allenMapResized(:)); % cluster IDs
                clusterIDs(clusterIDs == 0) = [];
                num_clusters = length(clusterIDs);
                
                %% Within cluster correlations
    
                % To store correlations for each cluster
                clusterwise_within_corr = zeros(num_clusters, 1);
                cluster_centres = zeros(num_clusters, 2);
    
                % Iterate over each cluster ID
                for id = 1:length(clusterIDs)
                    clusterID = clusterIDs(id);
                    
                    % Find the mask for the current cluster
                    clusterMask = (allenMapResized == clusterID);
                    
                    % cluster pixels
                    clus_pix_ids = find(clusterMask);
    
                    % Extract the time series data for the pixels in this cluster
                    clusterTimeSeries = V(repmat(clusterMask, [1, 1, size(V, 3)]));
                    
                    % Reshape the time series data into a 2D matrix (pixels x time)
                    clusterTimeSeries = reshape(clusterTimeSeries, [], size(V, 3));
                    
                    % Calculate the correlation matrix for this cluster
                    correlationMatrix = corr(clusterTimeSeries');
                    
                    % Store the average correlation (excluding the diagonal)
                    clusterwise_within_corr(id) = mean(correlationMatrix(triu(true(size(correlationMatrix)), 1)));
                
                    [p_r, p_c] = ind2sub([R, C], clus_pix_ids);
                    cluster_centres(id, 1) = mean(p_r);
                    cluster_centres(id, 2) = mean(p_c);
                end
                clusterwise_within_corr = clusterwise_within_corr';
                
                %% Cross cluster correlation
                NUM_NEAREST = 2;
                cl_distances = pdist2(cluster_centres, cluster_centres);
                clusterwise_across_corr = zeros(1, num_clusters);
                for cl = 1:num_clusters
                    dists = cl_distances(cl, :);
                    [~, sorted_ids] = sort(dists);
                    nearest_nbrs = sorted_ids(2:NUM_NEAREST+1); % 1st one is always itself (zero dist)
                    
                    cumul_corr = 0;
                    cumul_pix = 0;
                    for nn = 1:NUM_NEAREST
                        NbrClusterMask = (allenMapResized == nearest_nbrs(nn));
                        NbrclusterTimeSeries = V(repmat(NbrClusterMask, [1, 1, size(V, 3)]));
                        NbrclusterTimeSeries = reshape(NbrclusterTimeSeries, [], size(V, 3));
    
                        cross_corr_ = corr(clusterTimeSeries', NbrclusterTimeSeries');
                        cumul_corr = cumul_corr + sum(cross_corr_(:));
                        cumul_pix = cumul_pix + size(cross_corr_, 1)*size(cross_corr_, 2);
                    end
                    clusterwise_across_corr(cl) = cumul_corr / cumul_pix;
                end
    
                save(fullfile(AllenCorrStorePath, ['AllenCorr_sub_' prev_sub 'ses_' num2str(prev_ses),'_out.mat']), 'clusterwise_within_corr',...
                    'clusterwise_across_corr');
                
                % Processing done - initialize V to the next subject's data
                V = V0; 
            end
            prev_sub = sub_value;
            prev_ses = ses_value;
        end
    end
end

%%
if (RUN_ALLEN_CORR_REPORT)
     % All processed files
    pFileList = dir(fullfile(AllenCorrStorePath, '*.mat'));

    % Group the files subject-wise
    pFileGrp = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(AllenCorrStorePath, fname);

        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
        sub_value = sub_match{1}{1};
        ses_value = str2double(ses_match{1}{1});

        % Add the file to the subject group
        ses_key = sprintf('ses_%d', ses_value);
        if ~isfield(pFileGrp, sub_value)
            pFileGrp.(sub_value) = struct();
        end
        pFileGrp.(sub_value).(ses_key) = fpath;
    end

    subNames = fieldnames(pFileGrp);
    NUM_SESS = 3;
    mean_corr_within_subwise = zeros(1, length(subNames));
    mean_corr_across_subwise = zeros(1, length(subNames));
    var_corr_within_subwise = zeros(1, length(subNames));
    var_corr_across_subwise = zeros(1, length(subNames));

    min_corr_within_subwise = zeros(1, length(subNames));
    min_corr_across_subwise = zeros(1, length(subNames));
    max_corr_within_subwise = zeros(1, length(subNames));
    max_corr_across_subwise = zeros(1, length(subNames));
    displayCorr = zeros(length(subNames), 2);

    for s = 1:length(subNames)
        sub = subNames{s};
        sesNames = fieldnames(pFileGrp.(sub));
        corr_within_parcel = [];
        corr_across_parcel = [];
        for s1 = 1:length(sesNames)
            ses = sesNames{s1};
            dat_parcel = load(pFileGrp.(sub).(ses));
            corr_within_parcel = [corr_within_parcel, dat_parcel.clusterwise_within_corr];
            corr_across_parcel = [corr_across_parcel, dat_parcel.clusterwise_across_corr];
        end
        mean_corr_within_subwise(s) = mean(corr_within_parcel);
        mean_corr_across_subwise(s) = mean(corr_across_parcel);
        var_corr_within_subwise(s) = var(corr_within_parcel);
        var_corr_across_subwise(s) = var(corr_across_parcel);
        min_corr_within_subwise(s) = min(corr_within_parcel);
        min_corr_across_subwise(s) = min(corr_across_parcel);
        max_corr_within_subwise(s) = max(corr_within_parcel);
        max_corr_across_subwise(s) = max(corr_across_parcel);
        displayCorr(s, 1) = mean_corr_within_subwise(s);
        displayCorr(s, 2) = mean_corr_across_subwise(s);
    end
    subLabels = {'SLC01', 'SLC03', 'SLC04', 'SLC05', 'SLC06', 'SLC07', 'SLC08', 'SLC09', 'SLC10'};
    %groupLabels = {'Within\_parcel', 'Across\_parcel'};
    % figure;
    % boxplot(displayCorr);
    % %bar(displayCorr);
    % set(gca, 'XTickLabel', subLabels);
    % title('Average Temporal correlation between pixels');
    % xlabel('Subjects');
    % ylabel('Temporal correlation');
    % legend('Within\_parcel', 'Across\_parcel');
    % ylim([0 1]);

    % Create figure
    figure;
    hold on;
    
    % Plotting box-like shapes using min, max, mean, and variance
    for i = 1:length(subLabels)
        % Calculate lower and upper bounds of variance
        lowerBound = mean_corr_within_subwise(i) - sqrt(var_corr_within_subwise(i));
        upperBound = mean_corr_within_subwise(i) + sqrt(var_corr_within_subwise(i));
        
        % Draw the box (representing variance range)
        patch([i-0.2, i+0.2, i+0.2, i-0.2], [lowerBound, lowerBound, upperBound, upperBound], 'b', 'FaceAlpha', 0.3);
    
        % Draw the line from min to max (whiskers)
        plot([i, i], [min_corr_within_subwise(i), max_corr_within_subwise(i)], 'k-', 'LineWidth', 1.5); 
        
        % Draw the mean as a red line inside the box
        plot([i-0.2, i+0.2], [mean_corr_within_subwise(i), mean_corr_within_subwise(i)], 'r-', 'LineWidth', 2);
    end
    
    % Set labels and title
    xlabel('Subjects');
    ylabel('Values');
    title('Temporal correlation within parcel');
    
    % Set x-ticks to match group indices and label them as subjects
    xticks(1:length(subLabels));
    xticklabels({'SLC01', 'SLC03', 'SLC04', 'SLC05', 'SLC06', 'SLC07', 'SLC08', 'SLC09', 'SLC10'});
    ylim([0 1]);
    hold off;

    figure;
    hold on;
    
    % Plotting box plots using min, max, mean, and variance
    for i = 1:length(subLabels)
        % Calculate lower and upper bounds of variance
        lowerBound = mean_corr_across_subwise(i) - sqrt(var_corr_across_subwise(i));
        upperBound = mean_corr_across_subwise(i) + sqrt(var_corr_across_subwise(i));
        
        % Draw the box (representing variance range)
        patch([i-0.2, i+0.2, i+0.2, i-0.2], [lowerBound, lowerBound, upperBound, upperBound], 'b', 'FaceAlpha', 0.3);
    
        % Draw the line from min to max (whiskers)
        plot([i, i], [min_corr_across_subwise(i), max_corr_across_subwise(i)], 'k-', 'LineWidth', 1.5); 
        
        % Draw the mean as a red line inside the box
        plot([i-0.2, i+0.2], [mean_corr_across_subwise(i), mean_corr_across_subwise(i)], 'r-', 'LineWidth', 2);
    end

    save("allenmaps_corr_means.mat", 'mean_corr_within_subwise', 'mean_corr_across_subwise');
    
    % Set labels and title
    xlabel('Subjects');
    ylabel('Values');
    title('Temporal correlation across parcels');
    
    % Set x-ticks to match group indices and label them as subjects
    xticks(1:length(subLabels));
    xticklabels({'SLC01', 'SLC03', 'SLC04', 'SLC05', 'SLC06', 'SLC07', 'SLC08', 'SLC09', 'SLC10'});
    ylim([0 1]);
    hold off;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
