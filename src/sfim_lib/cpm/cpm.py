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
from sklearn.utils import shuffle

# ==========================================
# === Cross-validation related functions ===
# ==========================================
def mk_kfold_test_indices(scan_list, random_seed=43, k = 10, verb=False):
    """
    This function will return a numpy array, where each entry in the array
    tells us in which of the k-folds a given scan will belong to the test set.
    
    INPUTS
    ======
    scan_list: list of scan identifiers

    k: number of folds
 
    random_seed: random seed. Provide a number if you want to set it [default=43]
  
    OUTPUTS
    =======
    indices: np.array with one value per scan indicating k-fold in which the scan
             belongs to the test set. 
    """
    print('++ INFO [mk_kfold_test_indices]: Assigning scans to test set for each individual fold') 
    n_scans = len(scan_list)
    n_scans_per_fold = n_scans//k # floor integer for n_scans_per_fold
    if verb:
       print("++ INFO [mk_kfold_test_indices]: Min number of scans per fold = %d" % n_scans_per_fold) 
    indices = [[fold_no]*n_scans_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_scans % k # figure out how many subs are left over
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]    
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_scans, "Length of indices list does not equal number of sscans, something went wrong"

    indices = shuffle(indices, random_state=random_seed)
    if verb:
       print('++ INFO [mk_kfold_test_indices]: Final number of scans in test fold for each fold')
       print(pd.DataFrame(indices, columns=['Fold']).value_counts())
    return np.array(indices)

def mk_kfold_test_indices_subject_aware(scan_list, sub_list=None,random_seed=43, k=10,verb=False):
    """
    Split data into k-folds based on subject ID (as opposed to scan_id). This is done to ensure data from the same
    subject does not end on the test and training set for any fold. This is only useful for cases when there are
    multiple scans per subject.
    
    This function will return a numpy array, where each entry in the array tells us in which of the k-folds 
    a given scan will belong to the test set.
    
    INPUTS
    ======
    scan_list: list of scan identifiers. Each entry expected to be a tuple (sbj_id, run_id)

    sub_list: list of unique subjects. If not provided, it will be inferred from scan_list.

    k: number of folds
 
    random_seed: random seed. Provide a number if you want to set it [default=43]
  
    OUTPUTS
    =======
    indices: np.array with one value per scan indicating k-fold in which the scan
    """
    if sub_list is None:
        #Get list of subjects from scan list:
        print('++ INFO [mk_kfold_test_indices_subject_aware]: Extracting subject list from scan_list.')
        aux_sub_list = [sbj for sbj,_ in scan_list]
        aux_used     = set()
        sub_list     = [x for x in aux_sub_list if x not in aux_used and (aux_used.add(x) or True)]
        # LEADS TO RANDOM RESULTS BECUASE OF SET
        #sub_list = []
        #for (sbj,scan) in  scan_list:
        #    sub_list.append(sbj)
        #sub_list = list(set(sub_list))

    #Number of Subjects:
    n_subs = len(sub_list)
    print('++ INFO [mk_kfold_test_indices_subject_aware]: Number of subjects = %d' % n_subs)
    #Floor integer for n_subs_per_fold
    n_subs_per_fold = n_subs//k
    print('++ INFO [mk_kfold_test_indices_subject_aware]: Minimum Number of subjects per test fold = %d' % n_subs_per_fold)

    #Figure out how many subs are left over
    remainder = n_subs % k
    remainder_inds = list(range(remainder))

    #Generate repmat list of indices
    indices = [[fold_no]*n_subs_per_fold for fold_no in range(k)]
    indices = [item for sublist in indices for item in sublist]

    #Add indices for remainder subs
    [indices.append(ind) for ind in remainder_inds]

    assert len(indices)==n_subs, "Length of indices list does not equal number of subjects, something went wrong"

    #Shuffles in place, Random(i) sets the seed for each iteration
    indices = shuffle(indices,random_state=random_seed)

    #Convert scan_list to df so that it can be indexed
    scan_df = scan_list.to_frame(index=False)
    scan_df = scan_df.set_index('Subject')

    #Add the respective fold to each scan
    for sbj,idx in zip(sub_list,indices):
        scan_df.loc[sbj,'indices']=idx

    #Create df of just the folds
    scan_indices = scan_df['indices'].astype(int).values
    if verb:
       print('++ INFO [mk_kfold_test_indices]: Final number of scans in test fold for each fold')
       print(pd.DataFrame(scan_indices, columns=['Fold']).value_counts())

    return scan_indices

def _mk_kfold_indices_subject_aware_incorrect(scan_list, random_seed=43, k = 10):
    """
    Split scans into folds taking into account subject identity. This function is WRONG becuase it does not
    take into account the way in which Emily's original code encodes k-fold membership. 

    Will be removed in next version. Here only for checking against original implementations.
    
    INPUTS
    ======
    scan_list: list of scan identifiers (sbj,scan)
    
    k: number of folds
    
    random_seed: random seed. Provide a number if you want to set it [default=43]
  
    OUTPUTS
    =======
    indices: np.array with one value per scan indicating k-fold in which the scan
             belongs to the test set. 
    """
    # Count the number of scans
    n_scans                        = len(scan_list)
    # Shuffle scans to randomize the folds across iterations
    groups    = [sbj for (sbj,scan) in  scan_list]
    # Create GroupKFold object for k splits
    grp_cv  = GroupShuffleSplit(n_splits=k, random_state=random_seed) #Isabel setting random state 4/30/24
    indices = np.zeros(n_scans)
    for fold, (_,ix_test) in enumerate(grp_cv.split(scan_list,groups=groups)):
        indices[ix_test]=fold
    indices = indices.astype(int)
    return indices

def split_train_test(subj_list, indices, test_fold):
    """
    Given a list of scans, k-fold indices, and fold number, this function returns lists of tr_scan_ids and tt_scan_ids
    This function is to be used in combination with the mk_fold functions, as indices is expected to contain
    the k-fold in which a given scan will be in the test set. Very different expectations to scikit-learn.

    INPUTS
    ======
    subj_list: list of scan identifiers (sbj, scan)

    k: number of folds

    test_fold: current fold

    OUTPUTS
    =======
    tr_scan_ids: list of scans in training set (sbj,scan)

    tt_scan_ids: list of scans in test set (sbj,scan)
    """

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    tr_scan_ids = []
    for sub in subj_list[train_inds]:
        tr_scan_ids.append(sub)

    tt_scan_ids = []
    for sub in subj_list[test_inds]:
        tt_scan_ids.append(sub)

    return (tr_scan_ids, tt_scan_ids)

def get_train_test_data(fc_data, tr_scan_ids, tt_scan_ids, prediction_targets_df, prediction_target):
    """
    Extracts requested FC and prediction target for a given list of tr_scan_ids and tt_scan_ids
    
    INPUTS
    ======
    fc_data: Dataframe with FC per scan in vectorized form.
                 index = Multiindex (sbj, scan)
                 columns = connections by number

    tr_scan_ids: list of training scans (sbj,scan)

    tt_scan_ids: list of testing scans (sbj, scan)

    prediction_targets_df: Dataframe with prediction targets
                index = Multiindex (sbj, scan)
                columns = potential prediction targets

    prediction_target: target to predict

    OUTPUTS
    =======
    tr_vects: Dataframe with FC for training scans.    
                 index = Multiindex (sbj, scan)
                 columns = connections by number
   
    tr_target: Dataframe with target prediction for training scans. 
                index = Multiindex (sbj, scan)
                columns = prediction target

    tt_vects:  Dataframe with FC for test scans.    
                 index = Multiindex (sbj, scan)
                 columns = connections by number
    """
    tr_vcts = fc_data.loc[tr_scan_ids, :]
    tt_vcts = fc_data.loc[tt_scan_ids, :]

    tr_target = prediction_targets_df.loc[tr_scan_ids, prediction_target]

    return (tr_vcts, tr_target, tt_vcts)
   
# =====================================================
# ===                  CPM Functions                ===
# =====================================================
def select_features(tr_vcts, tr_pred_target, r_thresh=None, p_thresh=None, corr_type='pearson', verbose=False, **other_options):
    """
    For a given set of training scans and a prediction target, this function will find the brain edges that correlate above a given threshold with
    the prediction target.
    
    INPUTS
    ======
    tr_vcts: Dataframe with vectorized FC matrices for training scans

    tr_target: Dataframe with prediction target for training scans

    r_thresh: edge selection threshold as a Pearson's correlation value

    p_thresh: edge selection threshold as a P-val (associated with a Pearson's correlation)

    corr_type: metric used to asses the relationship between FC and prediction targets. 
               possible values: 'pearson', 'spearman'
               default value: 'pearson'

    verb: be verbose. defaulf=False 

    OUTPUTS
    =======
    mask_dict: dictionary with edges found to be strongly associated with the prediction target.
               'pos': edges that positively correlate with the prediction target.
               'neg': edges that negatively correlate with the prediction target.
    """
    assert not((p_thresh is not None) and (r_thresh is not None)), "++ERROR [select_features]: Threshold provided in two different ways. Do not know how to continue."
    assert corr_type in ['pearson','spearman'], "++ERROR [select_features]: Unknown correlation type."
    
    # Compute correlations between each edge and behavior
    n_edges = tr_vcts.shape[1]
    r = pd.Series(index=range(n_edges),name='r', dtype=float)
    p = pd.Series(index=range(n_edges),name='p', dtype=float)
    for edge in range(n_edges):
        if corr_type == 'pearson':
            r[edge],p[edge] = pearsonr(tr_vcts.loc[:,edge], tr_pred_target)
        if corr_type == 'spearman':
            r[edge],p[edge] = spearmanr(tr_vcts.loc[:,edge], tr_pred_target)
    # Select edges according to thresholding criteria
    mask_dict = {}
    if p_thresh is not None:
        print('++ INFO [select_features]: Threshold based on p_value [p<%f]' % p_thresh, end=' ')
        mask_dict["pos"] = (r > 0) & (p<p_thresh)
        mask_dict["neg"] = (r < 0) & (p<p_thresh)
    if r_thresh is not None:
        print('++ INFO [select_features]: Threshold based on R value [R>%f]' % r_thresh, end=' ')
        mask_dict["pos"] = r > r_thresh
        mask_dict["neg"] = r < -r_thresh
    if verbose:
        print("[{} pos/ {} neg] edges correlated with prediction target".format(mask_dict["pos"].sum(), mask_dict["neg"].sum()), end='') # for debugging
    return mask_dict

def build_model(tr_vcts, mask_dict, tr_pred_target, edge_summary_method='sum'):
    """
    Builds a CPM model:
    - takes a feature mask, sums all edges in the mask for each subject, and uses simple linear regression to relate summed network strength to behavior
    
    INPUTS
    ======
    train_vects: np.array(#scan,#connections) with the FC training data
    
    mask_dict: dictionary with two keys ('pos' and 'neg') with information about which edges were selected as meaningful during the edge selection step
    
    tr_pred_target: np.array(#scans,) with the behavior to be predicted
    
    edge_summary_method: whether to add or average edge strengths across all edges selected for a model. Initial version of CPM uses sum, but a variant
                         with mean was reported by Jangraw et al. 2018. This variant should in principle be less senstive to the number of edges entering
                         the model.
    
    OUTPUTS
    =======
    model_dict: contains one entry per model (pos, neg and glm). For each of the three models contains a tuple with two values, first the slope and then
                the intercept.
    
    NOTE: This function has been updated relative to the code in the cpm tutorial so that it does not fail for null models. When null models are present
          the the model is filled with np.nans, which in the other functions should be interpreted as a non-existing model.
    """
    
    assert tr_vcts.index.equals(tr_pred_target.index), "Row indices of FC vcts and behavior don't match!"
    assert edge_summary_method in ['mean','sum'], "Edge summary method not recognized"
    model_dict = {}
    X_glm      = None
    # FOR BOTH THE POSITIVE AND NEGATIVE MODEL
    for t, (tail, mask) in enumerate(mask_dict.items()):
        if mask.sum()>0:       # At least one edge entered the model
            if edge_summary_method == 'sum':
                X                  = tr_vcts.values[:, mask].sum(axis=1) # Pick the values for the edges in each subject and sum them.
            elif edge_summary_method == 'mean':
                X                  = tr_vcts.values[:, mask].mean(axis=1) # Pick the values for the edges in each subject and average them.
            y                  = tr_pred_target
            (slope, intercept) = np.polyfit(X, y, 1)
            model_dict[tail]   = (slope, intercept)
            if X_glm is None:
                X_glm = np.reshape(X,(X.shape[0],1))
            else:
                X_glm = np.c_[X_glm,X]
        else:
            print("++ WARNING [build_model_new,%s,%d]: No edges entered the model --> Setting slope and intercept to np.nan" % (tail,mask.sum()))
            model_dict[tail] = (np.nan, np.nan)
    # CONSTRUCT THE FULL MODEL WITH POSITIVE AND NEGATIVE TOGETHER
    if X_glm is None:
        print("++ WARNING [build_model_new,glm]: No edges entered the model --> Setting slope and intercept to np.nan")
        model_dict["glm"] = (np.nan, np.nan)
    else:
        X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
        model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])
    return model_dict
   
def apply_model(tt_vcts, mask_dict, model_dict, edge_summary_method='sum'):
    """
    Applies a previously trained linear regression model to a test set to generate predictions of behavior.
    
    INPUTS
    ======
    tt_vcts: np.array(#scan,#connections) with the FC training data
    
    mask_dict: dictionary with two keys ('pos' and 'neg') with information about which edges were 
               selected as meaningful during the edge selection step.
    
    
    model_dict: dictionary with three keys ('pos', 'neg' and 'glm'). For each models, the dictionary
                object contains a tuple with two values (or three for 'glm'), first the slope/s and 
                then the intercept.
    
    edge_summary_method: whether to add or average edge strengths across all edges selected for a model. 
                         Initial version of CPM uses sum, but a variant with mean was reported by Jangraw
                         et al. 2018. This variant should in principle be less senstive to the number of 
                         edges entering the model.
    OUTPUT
    ======
    pred_values: dictionary with one key per model. For each model it contains a pd.Series with the
                predicted behaviors. Unless model was empty.
    """
    
    assert edge_summary_method in ['mean','sum'], "Edge summary method not recognized"
    pred_values = {}
    X_glm = None

    for t, (tail, mask) in enumerate(mask_dict.items()):
        if mask.sum()>0: # At least one edge entered the model
            if edge_summary_method == 'sum':
                X                = tt_vcts.loc[:, mask].sum(axis=1).values
            elif edge_summary_method == 'mean':
                X                = tt_vcts.loc[:, mask].mean(axis=1).values
            slope, intercept = model_dict[tail]
            pred_values[tail] = pd.Series(slope*X + intercept).set_axis(tt_vcts.index)
            if X_glm is None:
                X_glm = np.reshape(X,(X.shape[0],1))
            else:
                X_glm = np.c_[X_glm,X] 
        else:
            pred_values[tail] = pd.Series(np.ones(tt_vcts.shape[0])*np.nan).set_axis(tt_vcts.index)
    
    if X_glm is None:
       pred_values["glm"] = pd.Series(np.ones(tt_vcts.shape[0])*np.nan).set_axis(tt_vcts.index)
    else:
       X_glm             = np.c_[X_glm, np.ones(X_glm.shape[0])]
       pred_values["glm"] = pd.Series(np.dot(X_glm, model_dict["glm"])).set_axis(tt_vcts.index)
    return pred_values

# =====================================
#      Ridge Regression Functions
# =====================================
def build_ridge_model(tr_vcts, sel_edges, tr_pred_target, alpha=0.5):
    """
    Build final model using Ridge approach (multiple connections).

    INPUTS
    ======
    tr_vcts: Dataframe with vectorized FC for training scans

    sel_edges: numpy array of length equal to the number of available edges.
               i-entry = 0 --> edge not selected
               i-entry = 1 --> selected edge

    tr_pred_target: prediction target for training scans 
    
    alpha: alpha value for the Ridge fit

    OUTPUTS
    =======
    ridge_obj: ridge model generated using the selected edges.
    """
    sel_edges_id = np.where(sel_edges==1)[0]
    X            = tr_vcts.copy().loc[:,sel_edges_id]
    ridge_obj  = Ridge(alpha=alpha)
    ridge_obj.fit(X,tr_pred_target)
    return ridge_obj

def apply_ridge_model(tt_vcts, sel_edges, model):
    """
    Apply previously computed Ridge model

    tt_vcts: Dataframe with vectorized FC for the test set

    sel_edges: numpy array of length equal to the number of available edges.
               i-entry = 0 --> edge not selected
               i-entry = 1 --> selected edge

    
    model: previously generated ridge_model obj

    OUTPUT
    ======
    pred_values = predicted values for target based on the application of the ridge model
    """
    sel_edges_id = np.where(sel_edges==1)[0]
    X            = tt_vcts.copy().loc[:,sel_edges_id]
    pred_values   = pd.Series(model.predict(X)).set_axis(tt_vcts.index)
    return pred_values
# ======================================
# End of Ridge functions
# ======================================


# ======================================
#     Confound Removal Functions
# ======================================
def get_confounds(tr_scans, tt_scans, confound_data, prediction_targets_df=None, motion=True, vigilance=False):
    """
    Remove confounds from data from target to be predicted
    
    INPUTS
    ======
    tr_scans: list of training scans. Expected to be a tutle (sbj_id, run_id)
   
    tt_scans: list of test scans. Expected to be a tuple (sbj_id, run_id)

    confound_data: dataframe with confound data

    prediction_targets_df: Dataframe with vigilance information

    motion: include motion as a confound. default = True

    vigilance: include vigilance as a confound. default = False

    OUTPUTS
    =======
    tr_confounds_df: dataframe with final confound info for the training set
    
    tt_confounds_df: dataframe with final confound info for the test set
    """
    assert all(isinstance(item, tuple) for item in tr_scans),"++ ERROR [get_confounds]: tr_scans is not list of tuples."
    assert all(isinstance(item, tuple) for item in tt_scans),"++ ERROR [get_confounds]: tt_scans is not list of tuples."
    
    tr_index = pd.MultiIndex.from_tuples(tr_scans, names=['Subject','Run'])
    tt_index  = pd.MultiIndex.from_tuples(tt_scans, names=['Subject','Run'])

    assert (tr_index).isin(confound_data.index).all(), "++ ERROR [get_confounds]: Confound info not available for all training scans"
    assert (tt_index).isin(confound_data.index).all(),  "++ ERROR [get_confounds]: Confound info not available for all test scans"
    
    assert (vigilance == False) or ((vigilance == True) & (prediction_targets_df is not None)), "++ ERROR [get_confounds]: vigilance = True, but no vigilance information provided"
    # Create column list for final confounds dataframe based on the motion and vigilance parameters
    cols = []
    if motion:
        cols.append('Motion')
    if vigilance:
        cols.append('Vigilance')
    # Create separate confound datasets for the test and training set
    tr_confounds_df = pd.DataFrame(index=tr_index, columns=cols)
    tt_confounds_df  = pd.DataFrame(index=tt_index, columns=cols)

    # Get motion confounds
    if motion is True:
        tr_confounds_df['Motion'] = confound_data.loc[tr_index].values
        tt_confounds_df['Motion']  = confound_data.loc[tt_index].values

    # Get vigilance confounds
    if vigilance is True:
        tr_confounds_df['Vigilance'] = prediction_targets_df.loc[tr_index,'Vigilance'].values
        tt_confounds_df['Vigilance']  = prediction_targets_df.loc[tt_index,'Vigilance'].values
    # Return confound dataframes for test and training datasets
    return tr_confounds_df, tt_confounds_df

def residualize(y,confounds):
    """
    Residualize the training set and returns a model to be used on the test set
    
    INPUTS
    ======
    y: original prediction targets
   
    confounds: confounds to be regressed

    OUTPUTS
    =======
    y_resid: prediction targets with confounds removed
 
    lm: model used for the removal of confounds. To be used on the test set later on.
    """
    for confound in confounds.columns:
        print("R_before = {:.3f}".format(pearsonr(y, confounds[confound])[0]), end=', ')

    lm = LinearRegression().fit(confounds, y)
    y_resid = y - lm.predict(confounds)

    for confound in confounds.columns:
        print("R_after = {:.3f}".format(pearsonr(y_resid, confounds[confound])[0]), end=' ')

    return y_resid, lm
# ==========================================
#      End of confound removal functions
# ==========================================
