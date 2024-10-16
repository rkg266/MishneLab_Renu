clc;
clear;
close all;

% Directory containing the files
dataDir = 'D:\UCSD_Acads\ProfGal_Research\data32\fMRIData\REST';

% Results diectory
resultDir = 'D:\UCSD_Acads\ProfGal_Research\test_run_KNN25_hemisphere_1';

% All sessions in the directory
sesList = dir(fullfile(dataDir, 'session-*'));

RUN_KNN = 0;
RUN_DICE_SIMILARITY = 0;
RUN_TEMPORAL_CORR = 1;
cfg.thrcluster=[0.9]; % @renu - handle this well 

if (RUN_KNN)
    % Run KNN after concatenating all runs of a session for each subject
    for ses = 1:length(sesList) % processing each session
        sesname = sesList(ses).name;
        sesDir = [dataDir, '\', sesname];
        fileList = dir(fullfile(sesDir, '*.nii.gz'));
        
        V = [];
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
            else      % New subject - perform LSSC to V matrix and initialize to new subject's data
                if (i == length(fileList)) % if last file
                    prev_sub = sub_value;
                    prev_ses = ses_value;
                end
                % Run LSSC for the current data (V)
                num_time_samples = size(V, 3);
                
                MIDLINE1 = 29;
                MIDLINE2 = 32;
                % Separating the brain region into two halves along the
                % midline
                % Extract the top and bottom halves
                V_top = V(1:MIDLINE1-1, :, :);        % Top half
                V_bottom = V(MIDLINE2+1:end, :, :);  % Bottom half
                
                % Flip the bottom half vertically (along the row axis)
                V_bottom_flipped = flip(V_bottom, 1);  % Flip along the 1st dimension (rows)
                
                num_time_frames = size(V_top, 3);
                
                % Pre-allocate a new matrix to hold the interleaved data
                ROW_PAD = 4; % To avoid out of index error in LSSC algorithm
                V_combined_h = zeros(size(V_top, 1)+ROW_PAD, size(V_top, 2), num_time_frames);

                % Interleave the top and bottom halves in time
                % for t = 1:num_time_frames
                %     V_combined_h(1:MIDLINE1-1, :, 2*t-1) = V_top(:, :, t);  % Odd indices: V_top
                %     V_combined_h(1:MIDLINE1-1, :, 2*t) = V_bottom_flipped(:, :, t);  % Even indices: V_bottom_flipped
                % end
  
                % Run for both hemispheres separately
                for hemisp = 1:2
                    for t = 1:num_time_frames
                        if (hemisp == 1)
                            V_combined_h(1:MIDLINE1-1, :, t) = V_top(:, :, t);
                        else
                            V_combined_h(1:MIDLINE1-1, :, t) = V_bottom_flipped(:, :, t);
                        end
                    end


                    % Create brain mask for the hemispheres
                    maxV = max(V_combined_h, [], 3);
                    brain_mask = maxV ~= 0;
                    [R,C] = size(brain_mask);
    
                    % Flatten the spatial dimensions
                    V_flat = reshape(V_combined_h, [], size(V_combined_h, 3));
                    
                    % Extract relevant pixels
                    allregionspix = find(brain_mask);
                    dFoF_masked = V_flat(allregionspix, :);
                    nPIX = size(dFoF_masked, 1); % number of pixels in brain region
                    
                    % KNN Prameters
                    N_KNN_CLUSTERS = 27;
                    cfg.ComputeTemporalCorr = true;
                    
                    % % Apply hierarchial cluistering with correlation as
                    % % distance metric
                    % % Compute the correlation matrix between the pixel time series
                    % corr_matrix = corr(dFoF_masked');  % Correlation between rows (pixels)
                    % 
                    % % Use 1 - correlation as the distance metric
                    % dist_matrix = 1 - corr_matrix;
                    % 
                    % % Perform K-means-like clustering using hierarchical clustering with correlation distance
                    % Z = linkage(dist_matrix, 'average');  % Average linkage hierarchical clustering
                    % idx = cluster(Z, 'maxclust', N_KNN_CLUSTERS);
                    
                    % Apply k-means clustering with Euclidean distance
                    % metric
                    [idx, clust_centers] = kmeans(dFoF_masked, N_KNN_CLUSTERS, 'MaxIter', 1000, 'Replicates', 5);

                    labels = zeros(R, C);
                    labels(allregionspix) = idx;
    
                    clust_idx = unique(labels);
                    clust_idx(clust_idx==0) = [];
                    mergedA = zeros(R*C, length(clust_idx));
                    for i1 = 1:length(clust_idx)
                        t_id = find(labels == clust_idx(i1));
                        mergedA(t_id, clust_idx(i1)) = 1;
                    end
    
                    title_str = ['sub_' prev_sub '_session_' num2str(prev_ses) '_hemisp_' num2str(hemisp)];
                    imagesc(label2rgb(labels));
                    saveas(gcf, fullfile(resultDir,['KNN_Labels_',title_str,'_nkNN_', num2str(N_KNN_CLUSTERS),'.png']));
                    
                    % TemporalCorr
                    clusterwise_within_corr = [];
                    clusterwise_across_corr = [];
                    if cfg.ComputeTemporalCorr
                        % compute temporal correlation between pixels
                        num_clusters = size(mergedA, 2);
                        cluster_centres = zeros(num_clusters, 2);
                        [nR, nC] = size(brain_mask);
        
                        % within parcel 
                        clusterwise_within_corr = zeros(1, num_clusters);
                        %parfor cl = 1:num_clusters  % @renu check the dFoF_masked for parfor 
                        for cl = 1:num_clusters
                            cl_pix_ids = find(mergedA(:, cl)); % pixel ids as per original image
                            [~, cl_pix_pos_mask] = ismember(cl_pix_ids, allregionspix); % position of corresponding pixel time series in maksed data - PixxTime_dff
                            pix_tseries = dFoF_masked(cl_pix_pos_mask, :);
                            corr_within_mat = corr(pix_tseries');            
        
                            % Compute average correlation value
                            upT = triu(corr_within_mat, 1);
                            nzupT = upT(upT ~= 0);
                            clusterwise_within_corr(cl) = mean(nzupT);
        
                            [p_r, p_c] = ind2sub([nR, nC], cl_pix_ids);
                            cluster_centres(cl, 1) = mean(p_r);
                            cluster_centres(cl, 2) = mean(p_c);
                        end
                        
                        % across parcels
                        NUM_NEAREST = 2;
                        cl_distances = pdist2(cluster_centres, cluster_centres);
                        clusterwise_across_corr = zeros(1, num_clusters);
                        for cl = 1:num_clusters
                            dists = cl_distances(cl, :);
                            [~, sorted_ids] = sort(dists);
                            nearest_nbrs = sorted_ids(2:NUM_NEAREST+1); % 1st one is always itself (zero dist)
                            
                            % Time series' of pixels in the current parcel
                            cl_pix_ids = find(mergedA(:, cl)); % pixel ids as per original image
                            [~, cl_pix_pos_mask] = ismember(cl_pix_ids, allregionspix); % position of corresponding pixel time series in maksed data - dFoF_masked
                            pix_tseries = dFoF_masked(cl_pix_pos_mask, :);
                            
                            cumul_corr = 0;
                            cumul_pix = 0;
                            for nn = 1:NUM_NEAREST
                                cl_pix_ids_nn = find(mergedA(:, nearest_nbrs(nn))); % pixel ids as per original image
                                [~, cl_pix_pos_mask_nn] = ismember(cl_pix_ids_nn, allregionspix); % position of corresponding pixel time series in maksed data - dFoF_masked
                                pix_tseries_nn = dFoF_masked(cl_pix_pos_mask_nn, :);
        
                                cross_corr_ = corr(pix_tseries', pix_tseries_nn');
                                cumul_corr = cumul_corr + sum(cross_corr_(:));
                                cumul_pix = cumul_pix + size(cross_corr_, 1)*size(cross_corr_, 2);
                            end
                            clusterwise_across_corr(cl) = cumul_corr / cumul_pix;
                        end
                    end
    
                    % save outputs
                    labels_all = cell(length(cfg.thrcluster), 1); % currently only 1
                    labels_all{1} = labels;
                    mergedA_all = cell(length(cfg.thrcluster), 1);
                    mergedA_all{1} = mergedA;
                    outfname = ['sub_' prev_sub '_ses_' num2str(prev_ses) '_hemisp_' num2str(hemisp) '_knn_out.mat'];
                    storepath = fullfile(resultDir, 'run_knn_sessions');
                    fulloutpath = fullfile(storepath, outfname);
                    results = struct();
                    results.filename = outfname;
                    results.labels = labels_all;
                    results.mergedA = mergedA_all;
                    results.clusterwise_within_corr = clusterwise_within_corr;
                    results.clusterwise_across_corr = clusterwise_across_corr;
                    save(fulloutpath, '-struct', 'results');
                end

                V = V0; % initialize to the new subject
            end
    
            prev_sub = sub_value;
            prev_ses = ses_value;
        end
    end
end
%%
if (RUN_DICE_SIMILARITY)
    processed_dir = fullfile(resultDir, 'run_knn_sessions');
    
    % All processed files
    pFileList = dir(fullfile(processed_dir, '*.mat'));

    % Group the files subject-wise
    pFileGrp = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(processed_dir, fname);

        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
        hem_match = regexp(fname, 'hemisp_(\d+)', 'tokens');
        sub_value = sub_match{1}{1};
        ses_value = str2double(ses_match{1}{1});
        hem_value = str2double(hem_match{1}{1});

        % Add the file to the subject group
        ses_key = sprintf('ses_%d', ses_value);
        hem_key = sprintf('hem_%d', hem_value);
        if ~isfield(pFileGrp, sub_value)
            pFileGrp.(sub_value) = struct();
        end
        pFileGrp.(sub_value).(ses_key).(hem_key) = fpath;
    end

    % Compute Dice similarity between all the session-pairs for each subject
    subNames = fieldnames(pFileGrp);
    NUM_SES_PAIRS = 3;
    displayDice = zeros(length(subNames), NUM_SES_PAIRS);
    for i = 1:length(subNames)
        sub = subNames{i};

        dice_pairwise = cell(1, NUM_SES_PAIRS);
        pairs = cell(1, NUM_SES_PAIRS);
        for j = 1:NUM_SES_PAIRS % 3 session pairs
            r1 = j;
            r2 = rem(j, NUM_SES_PAIRS) + 1;
            ke1 = sprintf('ses_%d', r1);
            ke2 = sprintf('ses_%d', r2);
            h1 = sprintf('hem_%d', 1);
            h2 = sprintf('hem_%d', 2);

            LSSC_out_pair = cell(1, 2);
            
            % Compute dice coefficient for each cluster threshold value
            
            dice_values = zeros(1, length(cfg.thrcluster));
            for thr_id = 1:length(cfg.thrcluster)
                dice_hem_wise = zeros(1, 2);
                for hmsp = 1:2
                    if (hmsp == 1)
                        LSSC_out_pair{1} = load(pFileGrp.(sub).(ke1).(h1));
                        LSSC_out_pair{2} = load(pFileGrp.(sub).(ke2).(h1));
                    elseif (hmsp == 2)
                        LSSC_out_pair{1} = load(pFileGrp.(sub).(ke1).(h2));
                        LSSC_out_pair{2} = load(pFileGrp.(sub).(ke2).(h2));
                    elseif (hmsp == 3)
                        LSSC_out_pair{1} = load(pFileGrp.(sub).(ke1).(h1));
                        LSSC_out_pair{2} = load(pFileGrp.(sub).(ke2).(h2));
                    else
                        LSSC_out_pair{1} = load(pFileGrp.(sub).(ke1).(h2));
                        LSSC_out_pair{2} = load(pFileGrp.(sub).(ke2).(h1));
                    end

                    % Pairing similar cluster labels 
                    pairing = pairComponents(LSSC_out_pair{1}.mergedA{thr_id}, LSSC_out_pair{2}.mergedA{thr_id});
                    p_1 = pairing.p1;
                    p_2 = pairing.p2;
                
                    labels_1 = LSSC_out_pair{1}.labels{thr_id};
                    labels_2 = LSSC_out_pair{2}.labels{thr_id};
                
                    % Re-labeling labels for 2nd one
                    for k1=1:size(labels_2, 1)
                        for k2=1:size(labels_2, 2)
                            if (labels_2(k1, k2) ~= 0)
                                tp = find(p_2 == labels_2(k1, k2));
                                if (isempty(tp))
                                    labels_2(k1, k2) = 0;
                                else
                                    labels_2(k1, k2) = p_1(tp);
                                end
                                
                            end
                        end
                    end
                
                    dice_similarity = multiclass_dice_coefficient(labels_1, labels_2);
                    dice_hem_wise(hmsp) = dice_similarity;
                end
                dice_values(thr_id) = mean(dice_hem_wise);
            end
            dice_pairwise{j} = dice_values;
            pairs{j} = sprintf('ses_%d-%d', r1, r2);
            displayDice(i, j) = dice_values;
        end
        pFileGrp.(sub).dice = dice_pairwise;
        pFileGrp.(sub).pairs = pairs;
    end
    
    data_dice = displayDice;
    %data_dice(:, 4) = mean(displayDice, 2);
    subLabels = {'SLC01', 'SLC03', 'SLC04', 'SLC05', 'SLC06', 'SLC07', 'SLC08', 'SLC09', 'SLC10'};
    groupLabels = {'Session 1-2', 'Session 2-3', 'Session 3-1'};
    figure;
    %bar(data_dice);
    bar(displayDice);
    set(gca, 'XTickLabel', subLabels);
    title('Comparison between sessions');
    xlabel('Subjects');
    ylabel('Dice value');
    legend('Session 1-2', 'Session 2-3', 'Session 3-1');
    ylim([0 1]);

    figure;
    bar(mean(displayDice, 2));
    set(gca, 'XTickLabel', subLabels);
    title('Comparison between sessions');
    xlabel('Subjects');
    ylabel('Dice value');
    legend('Average');
    ylim([0 1]);
end
%%
if (RUN_TEMPORAL_CORR)
     % All processed files
     processed_dir = fullfile(resultDir, 'run_knn_sessions');
    pFileList = dir(fullfile(processed_dir, '*.mat'));

    % Group the files subject-wise
    pFileGrp = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(processed_dir, fname);

        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
        hem_match = regexp(fname, 'hemisp_(\d+)', 'tokens');
        sub_value = sub_match{1}{1};
        ses_value = str2double(ses_match{1}{1});
        hem_value = str2double(hem_match{1}{1});

        % Add the file to the subject group
        ses_key = sprintf('ses_%d', ses_value);
        hem_key = sprintf('hem_%d', hem_value);
        if ~isfield(pFileGrp, sub_value)
            pFileGrp.(sub_value) = struct();
        end
        pFileGrp.(sub_value).(ses_key).(hem_key) = fpath;
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
        N_HEMISP = 2;
        for s1 = 1:length(sesNames)
            ses = sesNames{s1};
            for hmsp = 1:N_HEMISP
                hem_key = sprintf('hem_%d', hmsp);
                dat_parcel = load(pFileGrp.(sub).(ses).(hem_key));
                corr_within_parcel = [corr_within_parcel, dat_parcel.clusterwise_within_corr];
                corr_across_parcel = [corr_across_parcel, dat_parcel.clusterwise_across_corr];
            end
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
    
    % Plotting box-like shapes using min, max, mean, and variance
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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%