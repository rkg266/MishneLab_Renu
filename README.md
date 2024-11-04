# Validation of LSSC algorithm for multimodal imaging data - Calcium and fMRI 

**fMRI raw data:** raw data\fMRIData\REST

## LSSC - parcellation and temporal correlation (within and across parcel)
**Run code:** 
* Full brain: run_LSSC_fmri_sessions.m
* Hemisphere: run_LSSC_fmri_sessions_hemisphere.m 
  
**Files/folders required:** 
* Results folder:
    - Full brain: lssc results\test_run_norm1_pca0_kNN16_sftune4
    - Hemisphere: lssc results\test_run_norm1_pca0_kNN16_sftune4_hemisphere_1

**How to run?:** Update the directories as per your local and set the following paramaters in the code -
* RUN_LSSC (1/0): Enable to run the LSSC parcellation. Set "Line 88: cfg.ComputeTemporalCorr = true" to enable correlation calculation.
* RUN_DICE_SIMILARITY (1/0): Enable to compute Dice similarity and generate plots.
* RUN_TEMPORAL_CORR (1/0): Enable to consolidate and generate temporal correlation plots. 

## Allen map - parcellation and temporal correlation (within and across parcel)
**Run code:** allenmaps_v0.m <br/>
**Files/folders required:** 
* Allen map: allen maps\2D_calcium_atlas.nii
* Results folder: allen maps\test_run_Allen_fullbrain\AllenTempCorr

**How to run?:** Update the directories as per your local and set the following paramaters in the code -
* RUN_ALLEN_CORR_PROCESSING (1/0): Enable to run the Allen parcellation and correlation calculation. Outputs (.mat) get stored in the results folder.
* RUN_ALLEN_CORR_REPORT (1/0): Displays the correlation plots. Plots are not saved automatically. 

@Gal: Debug Calcium-LSSC code. <br/>
**Tip:** Search "@renu" in the code files to see my additions/edits. <br/>
**Run:** run_calcium_fmri.m 
