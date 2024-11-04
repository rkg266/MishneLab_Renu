clc;
clear;
close all;

% Directory containing the files
dataDir = 'D:\UCSD_Acads\ProfGal_Research\data32\fMRIData\REST';

% Results diectory
resultDir = 'D:\UCSD_Acads\ProfGal_Research\test_run_norm1_pca0_kNN16_sftune4';

% All sessions in the directory
sesList = dir(fullfile(dataDir, 'session-*'));

RUN_LSSC = 0;
RUN_DICE_SIMILARITY = 0;
RUN_TEMPORAL_CORR = 1;
cfg.thrcluster=[0.9]; % @renu - handle this well 

if (RUN_LSSC)
    % Run LSSC after concatenating all runs of a session for each subject
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
            else                
                %% Process the concatenated V matrix and at the end initialize it to next subject's data
                if (i == length(fileList)) % if last file
                    prev_sub = sub_value;
                    prev_ses = ses_value;
                end
                % Run LSSC for the current data (V)
                num_time_samples = size(V, 3);
        
                % Create brain mask
                %maxV1 = (max(V(:,:,1,:),[],3)); % @Gal doubt
                maxV = max(V, [], 3);
                brain_mask = maxV ~= 0;
                brain_mask(29:32,:) = 0; 
                [R,C] = size(brain_mask);
            
                % Flatten the spatial dimensions
                V_flat = reshape(V, [], size(V, 3));
                
                % Extract relevant pixels
                allregionspix = find(brain_mask);
                dFoF_masked = V_flat(allregionspix, :);
                
                % Configuring segmentation
                cfg.preProcess=false;
                cfg.N_TRIALS=1;
                cfg.n_clust = [100 ];
                cfg.makePlots = false;
                %cfg.thrcluster=[0.9:0.03:0.99];
                cfg.thrcluster=[0.9];
                cfg.NROWS = R;
                cfg.NCOLS = C;
                cfg.isoverlap = false;
                cfg.min_filled_area = 0.98;
                cfg.title_str = ['sub_' prev_sub 'session_' num2str(prev_ses) '_preproc_0_normal_1_pca_0_neig_51_nclust_100'];
                cfg.outputfilepath = resultDir;
                %@renu
                cfg.pca = 0;
                cfg.normalize = 1;
                cfg.ComputeTemporalCorr = true;
               
                % Run segmentation          
                [labels_all, mergedA_all] = runROI_meso_nlm_new_v1(cfg, dFoF_masked, allregionspix, brain_mask);  
    
                % save outputs
                outfname = ['sub_' prev_sub 'ses_' num2str(prev_ses) '_lssc_out.mat'];
                storepath = fullfile(resultDir, 'run_fmri_sessions');
                fulloutpath = fullfile(storepath, outfname);
                results = struct();
                results.filename = outfname;
                results.labels = labels_all;
                results.mergedA = mergedA_all;
                save(fulloutpath, '-struct', 'results');
    
                % Processing done - initialize V to the next subject's data
                V = V0; 
            end
    
            prev_sub = sub_value;
            prev_ses = ses_value;
        end
    end
end
%%
if (RUN_DICE_SIMILARITY)
    processed_dir = fullfile(resultDir, 'run_fmri_sessions');
    
    % All processed files
    pFileList = dir(fullfile(processed_dir, '*.mat'));

    % Group the files subject-wise
    pFileGrp = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(processed_dir, fname);

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
            k1 = sprintf('ses_%d', r1);
            k2 = sprintf('ses_%d', r2);

            LSSC_out_pair = cell(1, 2);
            LSSC_out_pair{1} = load(pFileGrp.(sub).(k1));
            LSSC_out_pair{2} = load(pFileGrp.(sub).(k2));
            
            % Compute dice coefficient for each cluster threshold value
            
            dice_values = zeros(1, length(cfg.thrcluster));
            for thr_id = 1:length(cfg.thrcluster)
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
                dice_values(thr_id) = dice_similarity;
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
    pFileList = dir(fullfile(resultDir, '*.mat'));

    % Group the files subject-wise
    pFileGrp = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(resultDir, fname);

        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'session_(\d+)', 'tokens');
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
