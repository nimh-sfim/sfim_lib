import os.path as osp
import pandas as pd

def correct_ranked_atlas(path_to_order_file, path_to_centroids_file, path_to_rank_file, verbose=True):
    # Create path to output files
    path_to_new_order_file    = osp.splitext(path_to_order_file)[0]+'.ranked.txt'
    path_to_new_centroids_file = osp.splitext(path_to_centroids_file)[0]+'.ranked.txt'
    
    # Load original order_file
    orig_order = pd.read_csv(path_to_order_file, header=None, delimiter='\t')
    orig_order.columns=['Number','ID','R','G','B','Size']
    orig_order = orig_order.set_index('Number')
    print("++ INFO [correct_ranked_shaefer_atlas] Original Order File in memory [%s]" % str(orig_order.shape))
    
    # Load original centroids into memory
    orig_centroids = pd.read_csv(path_to_centroids_file)
    orig_centroids = orig_centroids.set_index('ROI Label')
    print("++ INFO [correct_ranked_shaefer_atlas] Original Centroids File in memory [%s]" % str(orig_centroids.shape))
    
    # Load rank information file
    ranking = pd.read_csv(path_to_rank_file, comment='#', header=None, delimiter='\s+')
    ranking.columns = ['New','Old']
    print("++ INFO [correct_ranked_shaefer_atlas] Rank information in memory    [%s]" % str(ranking.shape))
    
    # Create new order file in memory
    new_order = pd.DataFrame(columns=orig_order.columns)
    for i,row in ranking.iterrows():
        if i == 0:
            continue # Ignore entry for ROI ID = 0 in Rank File
        new_order = new_order.append({'Number':row['New'],
                      'ID':  orig_order.loc[row['Old']]['ID'],
                      'R':   orig_order.loc[row['Old']]['R'],
                      'G':   orig_order.loc[row['Old']]['G'],
                      'B':   orig_order.loc[row['Old']]['B'],
                      'Size':orig_order.loc[row['Old']]['Size']},ignore_index=True)
    
    # Write new order file
    new_order.to_csv(path_to_new_order_file,header=False, index=False, sep='\t')
    print("++ INFO [correct_ranked_shaefer_atlas] New order file written to disk: %s" % path_to_new_order_file)
    
    # Create new centroids file in memory
    new_centroids  = pd.DataFrame(columns = orig_centroids.columns)
    for i,row in ranking.iterrows():
        if i == 0:
            continue
        new_centroids = new_centroids.append({'ROI Label':row['New'],
                                          'ROI Name':orig_centroids.loc[row['Old']]['ROI Name'],
                                          'R'       :orig_centroids.loc[row['Old']]['R'],
                                          'A'       :orig_centroids.loc[row['Old']]['A'],
                                          'S'       :orig_centroids.loc[row['Old']]['S']}, ignore_index=True)
    # Write new centroids to disk
    new_centroids.to_csv(path_to_new_centroids_file, index=False)
    print("++ INFO [correct_ranked_shaefer_atlas] New centroids file written to disk: %s" % path_to_new_centroids_file)
    if verbose:
        print("======================================")
        print(" + Original Number of ROIs = %d" % ranking['Old'].max())
        print(" + New      Number of ROIs = %d" % ranking['New'].max())
        print("======================================")
        print(" + Last entry in original Order File:")
        print(str(orig_order.iloc[-1]))
        print(" + Last entry in new Order File:")
        print(str(new_order.iloc[-1]))
        print("======================================")
        print(" + Last entry in original Centroids File:")
        print(str(orig_centroids.iloc[-1]))
        print(" + Last entry in new Centroids File:")
        print(str(new_centroids.iloc[-1]))
    return

# THIS FUNCTION IS NOW DEPRECATED. WILL BE REMOVED IN THE NEXT VERSION.
# It fails when ROIs do not start at 1 in the original atlas
def correct_ranked_shaefer_atlas(path_to_order_file, path_to_centroids_file, path_to_rank_file, verbose=True):
    # Create path to output files
    path_to_new_order_file    = osp.splitext(path_to_order_file)[0]+'.ranked.txt'
    path_to_new_centroids_file = osp.splitext(path_to_centroids_file)[0]+'.ranked.txt'
    
    # Load original order_file
    orig_order = pd.read_csv(path_to_order_file, header=None, delimiter='\t')
    orig_order.columns=['Number','ID','R','G','B','Size']
    print("++ INFO [correct_ranked_shaefer_atlas] Original Order File in memory [%s]" % str(orig_order.shape))
    
    # Load original centroids into memory
    orig_centroids = pd.read_csv(path_to_centroids_file)
    print("++ INFO [correct_ranked_shaefer_atlas] Original Centroids File in memory [%s]" % str(orig_centroids.shape))
    
    # Load rank information file
    ranking = pd.read_csv(path_to_rank_file, comment='#', header=None, delimiter='\s+')
    ranking.columns = ['New','Old']
    print("++ INFO [correct_ranked_shaefer_atlas] Rank information in memory    [%s]" % str(ranking.shape))
    
    # Create new order file in memory
    new_order = pd.DataFrame(columns=orig_order.columns)
    for i,row in ranking.iterrows():
        if i == 0:
            continue # Ignore entry for ROI ID = 0 in Rank File
        new_order = new_order.append({'Number':row['New'],
                      'ID':  orig_order.iloc[row['Old']-1]['ID'],
                      'R':   orig_order.iloc[row['Old']-1]['R'],
                      'G':   orig_order.iloc[row['Old']-1]['G'],
                      'B':   orig_order.iloc[row['Old']-1]['B'],
                      'Size':orig_order.iloc[row['Old']-1]['Size']},ignore_index=True)
    
    # Write new order file
    new_order.to_csv(path_to_new_order_file,header=False, index=False, sep='\t')
    print("++ INFO [correct_ranked_shaefer_atlas] New order file written to disk: %s" % path_to_new_order_file)
    
    # Create new centroids file in memory
    new_centroids  = pd.DataFrame(columns = orig_centroids.columns)
    for i,row in ranking.iterrows():
        if i == 0:
            continue
        new_centroids = new_centroids.append({'ROI Label':row['New'],
                                          'ROI Name':orig_centroids.iloc[row['Old']-1]['ROI Name'],
                                          'R':orig_centroids.iloc[row['Old']-1]['R'],
                                          'A':orig_centroids.iloc[row['Old']-1]['A'],
                                          'S':orig_centroids.iloc[row['Old']-1]['S']}, ignore_index=True)
    # Write new centroids to disk
    new_centroids.to_csv(path_to_new_centroids_file, index=False)
    print("++ INFO [correct_ranked_shaefer_atlas] New centroids file written to disk: %s" % path_to_new_centroids_file)
    if verbose:
        print("======================================")
        print(" + Original Number of ROIs = %d" % ranking['Old'].max())
        print(" + New      Number of ROIs = %d" % ranking['New'].max())
        print("======================================")
        print(" + Last entry in original Order File:")
        print(str(orig_order.iloc[-1]))
        print(" + Last entry in new Order File:")
        print(str(new_order.iloc[-1]))
        print("======================================")
        print(" + Last entry in original Centroids File:")
        print(str(orig_centroids.iloc[-1]))
        print(" + Last entry in new Centroids File:")
        print(str(new_centroids.iloc[-1]))
    return
