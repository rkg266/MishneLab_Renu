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
RUN_DICE_SIMILARITY = 1;
cfg.thrcluster=[0.9]; % @renu - handle this well 

if (RUN_LSSC)
    % Split each run into even-odd and run LSSC
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
            
            V = niftiread(filePath);
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
            
            dFoF_masked_even = V_even_flat(allregionspix_even, :);
            dFoF_masked_odd = V_odd_flat(allregionspix_odd, :);
            
            
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
            %cfg.title_str = ['sub_' sub_value 'session_' num2str(ses_value) '_preproc_0_normal_1_pca_1_neig_51_nclust_100'];
            cfg.outputfilepath = fullfile(resultDir, 'run_fmri_evenodd');

            %@renu
            cfg.pca = 0;
            cfg.normalize = 1;
            cfg.ComputeTemporalCorr = true;
            
            % Run segmentation for even and odd time points
            cfg.title_str = ['sub_' sub_value 'even_' 'session_' num2str(ses_value) 'run_' num2str(run_value) '_preproc_0_normal_1_pca_1_neig_51_nclust_100'];
            [labels_all_even, mergedA_all_even] = runROI_meso_nlm_new_v1(cfg, dFoF_masked_even, allregionspix_even, brain_mask_even);
        
            cfg.title_str = ['sub' sub_value 'odd_' 'session_' num2str(ses_value) 'run_' num2str(run_value) '_preproc_0_normal_1_pca_1_neig_51_nclust_100'];
            [labels_all_odd, mergedA_all_odd] = runROI_meso_nlm_new_v1(cfg, dFoF_masked_odd, allregionspix_odd, brain_mask_odd);

            results = struct();
            results.odd = struct();
            results.even = struct();
            results.odd.labels = labels_all_odd;
            results.odd.mergedA = mergedA_all_odd;
            results.even.labels = labels_all_even;
            results.even.mergedA = mergedA_all_even;

            % save outputs
            outfname = ['sub_' sub_value 'ses_' num2str(ses_value) 'run_' num2str(run_value) '_lssc_out.mat'];
            storepath = fullfile(resultDir, 'run_fmri_evenodd\lssc_processed');
            fulloutpath = fullfile(storepath, outfname);
            results.filename = outfname;
            save(fulloutpath, '-struct', 'results');
        end
    end
end

if (RUN_DICE_SIMILARITY)
    processed_dir = fullfile(resultDir, 'run_fmri_evenodd\lssc_processed');
    
    % All processed files
    pFileList = dir(fullfile(processed_dir, '*.mat'));
    diceResults = struct();
    for i = 1:length(pFileList)
        fname = pFileList(i).name;
        fpath = fullfile(processed_dir, fname);

        sub_match = regexp(fname, 'sub_(\w{5})', 'tokens');
        ses_match = regexp(fname, 'ses_(\d+)', 'tokens');
        run_match = regexp(fname, 'run_(\d+)', 'tokens');
        sub_value = sub_match{1}{1};
        ses_value = ses_match{1}{1};
        run_value = run_match{1}{1};
        
        pResult = load(fpath);
        pResultEven = pResult.even;
        pResultOdd = pResult.odd;

        % Compute dice coefficient for the even-odd pair
        dice_values = zeros(1, length(cfg.thrcluster));
        for thr_id = 1:length(cfg.thrcluster)
            % Pairing similar cluster labels 
            pairing = pairComponents(pResultEven.mergedA{thr_id}, pResultOdd.mergedA{thr_id});
            p_1 = pairing.p1;
            p_2 = pairing.p2;
        
            labels_1 = pResultEven.labels{thr_id};
            labels_2 = pResultOdd.labels{thr_id};
        
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
        ses_name = ['ses_' ses_value];
        run_name = ['run_' run_value];
        if ~isfield(diceResults, sub_value)
            diceResults.(sub_value) = struct();
            diceResults.(sub_value).(ses_name) = struct();
        end
        diceResults.(sub_value).(ses_name).(run_name) = dice_values;
    end
    
    % Plot Dice results
    num_subs = 9;
    num_sess = 3;
    data_dice = zeros(num_subs, num_sess);
    subLabels = {'SLC01', 'SLC03', 'SLC04', 'SLC05', 'SLC06', 'SLC07', 'SLC08', 'SLC09', 'SLC10'};
    sessKeys = {'ses_1', 'ses_2', 'ses_3'};
    for sub = 1:num_subs
        cur_dice = diceResults.(subLabels{sub});
        for s = 1:num_sess
            ses_dice = cur_dice.(sessKeys{s});
            % Extract the field names (runs) of the struct
            fldNames = fieldnames(ses_dice);
            % Extract the values from each field (run)
            d_vals = cellfun(@(f) ses_dice.(f), fldNames);
            data_dice(sub, s) = mean(d_vals);
        end
    end

    figure;
    bar(data_dice);
    set(gca, 'XTickLabel', subLabels);
    title('Comparison between even and odd');
    xlabel('Subjects');
    ylabel('Dice value');
    legend('Session 1', 'Session 2', 'Session 3');
    ylim([0 1]);

    figure;
    bar(mean(data_dice, 2));
    set(gca, 'XTickLabel', subLabels);
    title('Comparison between even and odd');
    xlabel('Subjects');
    ylabel('Dice value');
    legend('Average');
    ylim([0 1]);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%