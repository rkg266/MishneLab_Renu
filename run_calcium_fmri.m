clc;
clear; 
close all;

calFastDir = 'D:\UCSD_Acads\ProfGal_Research\data32\CalciumData\REST\session-1\FastBand';
fMRIDir = 'D:\UCSD_Acads\ProfGal_Research\data32\fMRIData\REST\session-1';
calSlowDir = 'D:\UCSD_Acads\ProfGal_Research\data32\CalciumData\REST\session-1\SlowBand';

% Results diectory
resultDir = 'D:\UCSD_Acads\ProfGal_Research\test_run_CalFMRI_1';

calFastFile = fullfile(calFastDir, 'animal01_ses-1_run3_regress_dff_masked_flipped_fMRI_lengthtrimmed2common_cleaned_fast.nii.gz');
fMRIFile = fullfile(fMRIDir, 'sub-SLC01_ses-1_task-rest_acq-EPI_run-11_bold_RAS_combined_cleaned2D.nii.gz');
calSlowFile = fullfile(calSlowDir, 'animal01_ses-1_run3_regress_dff_masked_flipped_fMRI_lengthtrimmed2common_cleanedslow.nii.gz');



VCalFast = niftiread(calFastFile);
%VCalFast = niftiread(calSlowFile);
VfMRI = niftiread(fMRIFile);

VCalFast = squeeze(VCalFast);
VfMRI = squeeze(VfMRI);

% Downsample Calcium data
%VCalFast = imresize(VCalFast, [size(VfMRI, 1), size(VfMRI, 2)]);

% Video Cal data
% fig1 = figure;
% fig2 = figure;
% for j4 = 1:size(VCalFast, 3)
%     figure(fig1);
%     imagesc(VCalFast(:, :, j4));
%     %caxis([0 1]);
%     colorbar;
%     colormap('jet');
% 
%     figure(fig2);
%     imagesc(VCalFast_down(:, :, j4));
%     %caxis([0 1]);
%     colorbar;
%     colormap('jet');
%     drawnow;
%     pause(0.1);
% end

ntsamp_Ca = size(VCalFast, 3);
ntsamp_fmri = size(VfMRI, 3);
        
% Create brain mask
%maxV1 = (max(V(:,:,1,:),[],3));
maxV_Ca = max(VCalFast, [], 3);
maxV_fmri = max(VfMRI, [], 3);

brain_mask_Ca = maxV_Ca ~= 0;
brain_mask_Ca(58:64,:) = 0; 
brain_mask_fmri = maxV_fmri ~= 0;
brain_mask_fmri(29:32,:) = 0; 
[R,C] = size(brain_mask_Ca);

% Flatten the spatial dimensions
V_Ca_flat = reshape(VCalFast, [], size(VCalFast, 3));
V_fmri_flat = reshape(VfMRI, [], size(VfMRI, 3));

% Extract relevant pixels
allregionspix_Ca = find(brain_mask_Ca);
dFoF_masked_Ca = V_Ca_flat(allregionspix_Ca, :);
allregionspix_fmri = find(brain_mask_fmri);
dFoF_masked_fmri = V_fmri_flat(allregionspix_fmri, :);

% Configuring segmentation
cfg.preProcess=false;
cfg.N_TRIALS=1;
cfg.n_clust = [100 ];
cfg.makePlots = true;
%cfg.thrcluster=[0.9:0.03:0.99];
cfg.thrcluster=[0.9];
cfg.NROWS = R;
cfg.NCOLS = C;
cfg.isoverlap = false;
cfg.min_filled_area = 0.98;

 %@renu
cfg.ComputeTemporalCorr = false;
cfg.pca=1; 

cfg.outputfilepath = resultDir;

% Run segmentation Ca 
cfg.title_str = ['CalFast_' 'session_'  '_preproc_0_normal_1_pca_1_neig_51_nclust_100'];
[labels_all_Ca, mergedA_all_Ca] = runROI_meso_nlm_new_v1(cfg, dFoF_masked_Ca, allregionspix_Ca, brain_mask_Ca);  

% % Run segmentation fMRI
% cfg.title_str = ['fMRI_' 'session_' '_preproc_0_normal_1_pca_1_neig_51_nclust_100'];
% [labels_all_fmri, mergedA_all_fmri] = runROI_meso_nlm_new_v1(cfg, dFoF_masked_fmri, allregionspix_fmri, brain_mask_fmri);  