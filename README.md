# Validation of LSSC algorithm for multimodal imaging data - Calcium and fMRI 

## Allen map - parcellation and temopral correlation (within and across parcel)
**Run code:** allenmaps_v0.m <br/>
**Files/folders required:** 
* Allen map: allen maps\2D_calcium_atlas.nii
* Results folder: allen maps\test_run_Allen_fullbrain\AllenTempCorr <br/>
**How to run?:** Update the directories as per your local and set the following paramaters in the code -
* RUN_ALLEN_CORR_PROCESSING (1/0): Enable to run the Allen parcellation and correlation calculation. Outputs (.mat) get stored in the results folder.
* RUN_ALLEN_CORR_REPORT (1/0): Displays the correlation plots. Plots are not saved automatically. 

@Gal: Debug Calcium-LSSC code. <br/>
**Tip:** Search "@renu" in the code files to see my additions/edits. <br/>
**Run:** run_calcium_fmri.m 
