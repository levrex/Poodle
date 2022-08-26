import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial.distance import cosine# cosine minkowski
from sklearn.metrics.pairwise import cosine_similarity
import time
from math import exp
from pickle import load
import xgboost as xgb

def t_test(x,y,alternative='both-sided'):
    """
    Description:
        Perform a t-test, either one- or two-sided
    """
    stats, double_p = ttest_ind(x,y,equal_var = False)

    if alternative == 'both-sided':
        pval = double_p
    elif alternative == 'greater':
        if np.mean(x) > np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    elif alternative == 'less':
        if np.mean(x) < np.mean(y):
            pval = double_p/2.
        else:
            pval = 1.0 - double_p/2.
    return stats, pval

def getSimilarityWithin(sim_matrix, cluster_indices):
    """
    Description:
        Acquire the within cluster similarity distribution (all pairwise distances of said cluster).

    Input: 
        sim_matrix = distance/similarity matrix 
        
    Return:
        cluster_scores = all pairwise distances of patients within the same cluster
    """

    # Within cluster
    sim_matrix = np.array(pd.DataFrame(sim_matrix).loc[cluster_indices, cluster_indices])
    
    # Collect scores of all unique pairwise distances within cluster
    mask = np.triu(np.ones_like(sim_matrix, dtype=np.bool),1) # 1 below diagonal
    cluster_scores =  np.array(sim_matrix[mask])

    return cluster_scores

def projectSampleMAUI(maui_model, z_space, d_input, sample): # sample
    """
    Input:
        maui_model = Loaded autoencoder object with the learned product space 
        z_space = product space (based on original set)
    
        d_input = dictionary featuring columns in the original space for each modality
        sample = features of 1 patient from the replication set
        
    Description: 
    Project new sample onto the product space by employing previously trained MAUI model.
    We also make sure to look at the same latent factors as before!
    """
    
    sample_cat = sample[d_input['cat']].copy()
    sample_num = sample[d_input['num']].copy()   
    new_pat = sample.name
    
    df_cat = pd.DataFrame(columns=list(sample_cat.index))
    df_cat.loc[0] = sample_cat
    
    df_num = pd.DataFrame(columns=list(sample_num.index))
    df_num.loc[0] = sample_num
    
    # Project new sample in product space
    z_patient = maui_model.transform({ 'Categorical': df_cat.T, 'Lab_numerical': df_num.T})
    
    # Add Merged factors
    l_merged = [col for col in z_space.columns if '_' in col]
    for col in l_merged: 
        i, j = col.split('_')[0], col.split('_')[1]
        z_patient[col] =  z_patient[['LF%s' % i, 'LF%s' % j]].mean(axis=1)
    
    # Only select relevant latent factors
    z_patient = z_patient[z_space.columns]
    
    # Add new patient to product space
    z_space = z_space.append(z_patient, ignore_index = True)
    return z_space

def get_digital_twins(df_meta, new_pat, sim_matrix, indices, idx=None):
    """
    Description:
    Find the digital twins of the newly projected patient.
    
    Input:
        df_meta = pandas dataframe with coordinates of latent space + meta data, such as cluster information
        new_pat = sample identifier for the patient that is projected onto the latent space
        sim_matrix = similarity_matrix
        indices = list of indices
        
    Code inspired by: https://www.kaggle.com/a7madmostafa/imdb-movies-recommendation-system
    """
    idx = indices[new_pat]
        
    # Quantify similarity across patients
    sim_scores = list(enumerate(sim_matrix[idx]))

    # Sort patients according to similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar patients
    sim_scores = sim_scores[1:11]

    # Get the indices of these patients
    patient_indices = [i[0] for i in sim_scores]

    # Return the neighbours (a.k.a the top 10 most similar patients)
    df_neighbours = df_meta[['pseudoId', 'PhenoGraph_clusters']].iloc[patient_indices].copy()
    df_neighbours = df_neighbours.reset_index(drop=True)
    return df_neighbours.join(pd.Series([i[1] for i in sim_scores], name='Similarity'))

def find_neighbours(df_meta, z_filtered, new_pat, idx=None):
    """
    Input:
        df_cluster = dataframe with cluster information
        z_filtered = product space created by MAUI
        new_pat = pseudoId of the projected patient
        l_pseudoId = list of pseudoIds; this should contain the ids of the development set 
            plus 1 extra id (of projected patient)
    
    Description:
        Find the most similar patients 
    """
    df_meta = pd.concat([df_meta, z_filtered], axis=1)
    
    # Compute the cosine similarity matrix
    sim_matrix = cosine_similarity(z_filtered.astype(np.float32))

    #Construct a reverse map of indices and cluster info
    indices = pd.Series(df_meta.index, index=df_meta['pseudoId']).drop_duplicates()

    return get_digital_twins(df_meta, new_pat, sim_matrix, indices, idx=idx)
    
def getMetaDataPatient(df_cluster, l_pseudoId, new_pat):
    """
    Description:
        Construct metadata for the original data and append the newly projected patient
    """
    # make list of pseudoIds
    l_pseudoId_replication = l_pseudoId.copy()
    l_pseudoId_replication.append(str(new_pat))
    
    # Construct metadata
    d = {'pseudoId' : l_pseudoId_replication}
    df_meta = pd.DataFrame(data=d)
    d_phenograph = dict(zip(df_cluster['pseudoId'], df_cluster['PhenoGraph_clusters']))
    df_meta['PhenoGraph_clusters'] = df_meta['pseudoId'].apply(lambda x : d_phenograph[x] if x in d_phenograph.keys() else -1)
    return df_meta

def similarityToCluster(df, sim_matrix, cluster_ix=0, cluster_label='PhenoGraph_clusters', output_dist=False):
    """
    Description:
        Calculate the distances of a new patient to a specific cluster (cluster_ix), and 
        compare this distribution (distance of new patient vs all patients of cluster) to 
        the within cluster similarity (all pairwise distances of said cluster).

    Input:
        df = pandas dataframe containing the metadata
        sim_matrix = distance/similarity matrix 
        new_pat = patient of interest
        cluster_ix = the cluster of interest
        cluster_label = name of cluster columns
        output_dist = set to True if you want to get the raw cluster/ patient distances
            
    Output: 
        list of predictors expressing the relationship between the sample and the cluster
    
    """
    # Get the indices of patients from specific cluster
    cluster_indices = list(df[df[cluster_label]==cluster_ix].index)

    # Add the new patient
    patient_indices = cluster_indices.copy()
    patient_indices.append(len(sim_matrix)-1)

    # Subset similarity matrix on said cluster
    sim_matrix_cluster = pd.DataFrame(sim_matrix.copy()).loc[patient_indices, patient_indices]

    # Patient vs cluster
    patient_scores = list(sim_matrix_cluster[len(sim_matrix)-1])
    patient_scores = sorted(patient_scores, reverse=True)
    patient_scores = patient_scores[1:] # remove comparison with self

    # Within cluster
    sim_matrix_cluster = np.array(pd.DataFrame(sim_matrix_cluster).loc[cluster_indices, cluster_indices])
    # Keep scores of all unique pairwise distances within cluster
    mask = np.triu(np.ones_like(sim_matrix_cluster, dtype=np.bool),1) # 1 below diagonal
    cluster_scores =  np.array(sim_matrix_cluster[mask])

    # Perform one tailed t-test
    tstat, pval = t_test(patient_scores, cluster_scores, alternative='less')

    # Define predictors (Be aware: we assume a normal distribution)
    pred_0 = max(0.000001, pval) #pval # H0 = patient is similar to cluster, Ha = patient is not similar
    pred_1 = np.mean(cluster_scores) # average cluster probability
    pred_2 = np.std(cluster_scores) # stability of the cluster
    pred_3 = np.mean(patient_scores) # average patient probability
    pred_4 = np.std(patient_scores) # stability of patient probability
    
    if output_dist == False: 
        # Return predictors for model
        return [pred_0, pred_1, pred_2, pred_3, pred_4]
    else : 
        # Return predictors for model
        return patient_scores, cluster_scores

def getOrientation(maui_model, df_meta, z_existent, d_input, sample, sim_matrix=None, cluster_label='PhenoGraph_clusters'):
    """
    Description: 
    Discover the orientation of the sample on the learned embedding and quantify its similarity to each cluster
    
    Output: 
        l_orientation = list that features predictors expressing the relationship 
            between the sample and each cluster 
        df_meta = metadata of original sample population + newly projected sample
    
    """
    # Bookmark all orientation info
    l_orientation = []
    z_updated = projectSampleMAUI(maui_model, z_existent.copy(),  d_input, sample)
    
    # We only need to calculate the pairwise similarities of the initial space 1 time
    if type(sim_matrix) == type(None) :
        # Construct similarity matrix
        sim_matrix = cosine_similarity(z_updated[:-1].astype(np.float32))

    # Create duplicate
    sim_matrix_child = sim_matrix.copy()
    sim_matrix_child = pd.DataFrame(sim_matrix_child)

    # Calculate distance of projected patient to other patients
    l_dist = [1-cosine(z_updated.iloc[-1].values, z_updated.loc[i].values) for i in range(len(z_updated))]

    # Add to distance matrix
    sim_matrix_child.loc[len(sim_matrix_child)] = l_dist[:-1]
    sim_matrix_child[len(sim_matrix_child)-1] = l_dist
    
    n_clusters = len(df_meta[cluster_label][:-1].unique()) # ignore final row (because it is the projected patient)
    
    for cluster_ix in range(n_clusters): 
        # Calculate similarity to each cluster
        sim_scores = similarityToCluster(df_meta, sim_matrix_child, cluster_ix, cluster_label=cluster_label)
        l_orientation.extend(sim_scores)
    return l_orientation

def classifyPatient(X, path='../example_data/model/labeler/'):
    """
    Description:
        Employ a previously trained Poodle model to assign cluster labels to
        the new samples
    
    Input:
        X = input data
        path = path to Poodle model tasked with labeling the patients
    """
    
    # load model
    loaded_bst = xgb.Booster()
    loaded_bst.load_model('%s' % path + 'xgb_model.json' )

    # load the scaler
    scaler = load(open('%s' % path + 'scaler.pkl' , 'rb'))
    
    # Apply Z-score normalization on the data
    X = scaler.transform(X)

    dmat_blind = xgb.DMatrix(X)
    return loaded_bst.predict(dmat_blind) 