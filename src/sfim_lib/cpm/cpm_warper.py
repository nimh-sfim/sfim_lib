# This file contains code copied from Emily Finn's github account 
# https://github.com/esfinn/movie_cpm/blob/master/code/cpm.py

import pandas as pd
import os.path as osp
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sfim_lib.io.afni import load_netcc
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import random
from cpm import  *

# ==========================================
#      End of confound removal functions
# ==========================================
# Deprecated function, we now run code as separate programs
def cpm_wrapper(fc_data, prediction_targets_df, prediction_target, k=10, **cpm_kwargs):
    """This function will run the whole CPM algorithm given a set of connectivity data, a target behavior to predict and a few hyper-parameters.
    
    INPUTS
    ======
    fc_data: a pd.DataFrame with rows denoting scans and columns denoting connections (#scans, #unique connections).
    
    prediction_targets_df: a pd.DataFrame with rows denoting scans and columns denoting behaviors (#scans, #behaviors).
    
    prediction_target: column identifier to select a behavior to be predicted among those presented as columns in prediction_targets_df
    
    k: number of k-folds for the cross-validation procedure. [default = 10]
    
    cpm_kwargs: additional hyper-parameters for the CPM algoritm in the form of a dictionary.
                * r_threh: edge-based threshold for the feature selection step.
                * corr_type: correlation type for the edge selection step (pearson or spearman).
                * verbose: show additional information.
    
    OUTPUTS
    =======
    behav_obs_pred: 
    
    all_masks:
    """
    # Check input requirements
    # ========================
    assert isinstance(fc_data,pd.DataFrame), "++ ERROR [cpm_wrapper]: fc_data is not an instance of pd.DataFrame"
    assert isinstance(prediction_targets_df,pd.DataFrame), "++ ERROR [cpm_wrapper]:fc_data is not an instance of pd.DataFrame"
    assert prediction_target in prediction_targets_df.columns, "++ ERROR [cpm_wrapper]:behavior not present in prediction_targets_df"
    assert fc_data.index.equals(prediction_targets_df.index), "++ ERROR [cpm_wrapper]:Index in FC dataFrame and behavior dataframe do not match"
    assert (('r_thresh' in cpm_kwargs) & ('p_thresh' not in cpm_kwargs)) | (('r_thresh' not in cpm_kwargs) & ('p_thresh' in cpm_kwargs)), "++ ERROR [cpm_wrapper]: Provided edge-level threshold as both p-value and r-value. Remove one"
    
    if ('r_thresh' not in cpm_kwargs):
        cpm_kwargs['r_thresh'] = None
    if ('p_thresh' not in cpm_kwargs):
        cpm_kwargs['p_thresh'] = None
    print(cpm_kwargs)
    
    # Extract list of scan identifiers from the fc_data.index
    # =======================================================
    scan_list = fc_data.index

    # Split the same into k equally sized (as much as possible) folds
    # ===============================================================
    indices = mk_kfold_test_indices(scan_list, k=k)
    
    # Verbose
    # =======    
    if cpm_kwargs['verbose']:
        print('++ INFO [cpm_wrapper]: Number of scans                      = %d' % len(scan_list))
        print('++ INFO [cpm_wrapper]: Number K-folds                       = %d' % k)
        print('++ INFO [cpm_wrapper]: Correlation mode                     = %s' % cpm_kwargs['corr_type'])
        if not (cpm_kwargs['r_thresh'] is None):
            print('++ INFO [cpm_wrapper]: Edge Selection Threshold (r)         = %.3f' % cpm_kwargs['r_thresh'])
        if not (cpm_kwargs['p_thresh'] is None):
            print('++ INFO [cpm_wrapper]: Edge Selection Threshold (p-val)     = %.3f' % cpm_kwargs['p_thresh'])
        print('++ INFO [cpm_wrapper]: Target Beahvior Label                = %s' % prediction_target)
        print('++ INFO [cpm_wrapper]: Edge Summarization Method            = %s' % cpm_kwargs['edge_summary_method'])
        
    # Initialize df for storing observed and predicted behavior for the three models
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(prediction_target + " predicted (" + tail + ")")
    col_list.append(prediction_target + " observed")
    behav_obs_pred = pd.DataFrame(index=scan_list, columns = col_list)

    # Initialize array for storing feature masks to all zeros
    n_edges = fc_data.shape[1]
    all_masks = {}
    all_masks["pos"] = np.zeros((k, n_edges))
    all_masks["neg"] = np.zeros((k, n_edges))

    # For each cross-validation fold
    for fold in range(k):
        print(" + doing fold {}".format(fold), end=' --> ')
        # Gather testing and training data for this particular fold
        # =========================================================
        train_subs, test_subs              = split_train_test(scan_list, indices, test_fold=fold)
        tr_vcts, tr_pred_target, tt_vcts = get_train_test_data(fc_data, train_subs, test_subs, prediction_targets_df, prediction_target=prediction_target)
        tr_vcts  = tr_vcts.infer_objects()
        tr_pred_target = tr_pred_target.infer_objects()
        tt_vcts   = tt_vcts.infer_objects()
        # Find edges that correlate above threshold with behavior in this fold
        # ====================================================================
        mask_dict   = select_features(tr_vcts, tr_pred_target, **cpm_kwargs)
        # Gather the edges found to be significant in this fold in the all_masks dictionary
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
        # Build model and predict behavior
        # ================================
        model_dict = build_model(tr_vcts, mask_dict, tr_pred_target)
        behav_pred = apply_model(tt_vcts, mask_dict, model_dict)
    
        # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
        # ========================================================================================================
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, prediction_target + " predicted (" + tail + ")"] = predictions

    # Add observed behavior to the returned dataframe behav_obs_pred
    # ==============================================================
    behav_obs_pred.loc[scan_list, prediction_target + " observed"] = prediction_targets_df[prediction_target]
    
    return behav_obs_pred, all_masks
