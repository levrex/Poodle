import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial.distance import cosine# cosine minkowski
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import time
from math import exp
from pickle import load
import xgboost as xgb # ToDo: Prevent that you need to load one of these -> 
from sklearn.metrics.pairwise import pairwise_distances

def t_test(x,y,alternative='both-sided'):
    """
    Description:
        Perform a t-test, either one- or two-sided
    """
    stats, double_p = ttest_ind(list(x),list(y),equal_var = False)

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
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool),1) # 1 below diagonal
    cluster_scores =  np.array(sim_matrix[mask])

    return cluster_scores

def projectSample(model, z_space, d_input, sample, prefix_lf='LF'): # sample
    """
    Input:
        model = Loaded autoencoder object with the learned product space (Tensorflow model)
        z_space = product space (based on original set)
        d_input = dictionary featuring columns in the original space for each modality
        sample = features of 1 patient from the replication set
        prefix_lf =  prefix to recognize latent factor coordinates (default: LF)
        
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
    if hasattr(model, "transform"):
        z_patient = model.transform({ 'Categorical': df_cat.T, 'Lab_numerical': df_num.T})
    elif hasattr(model, "encoder"):
        z_patient = model.encoder.predict([np.array(df_cat.values), np.array(df_num.values)], batch_size=256)
        z_patient = pd.DataFrame(z_patient)
    
        z_patient.columns = [prefix_lf + '%s' % col for col in z_patient.columns]
                             
    
    # Add Merged factors
    l_merged = [col for col in z_space.columns if '_' in str(col)]
    for col in l_merged: 
        i, j = col.split('_')[0], col.split('_')[1]
        z_patient[col] =  z_patient[[prefix_lf + '%s' % i, prefix_lf + '%s' % j]].mean(axis=1)
    
    # Only select relevant latent factors
    z_patient = z_patient[z_space.columns]
    
    # Add new patient to product space
    z_space = pd.concat([z_space, z_patient], ignore_index=True)
    return z_space

def getOrientation(model, df_meta, z_existent, d_input, sample, sim_matrix=None, cluster_label='PhenoGraph_clusters', prefix_lf='LF'):
    """
    Description: 
    Discover the orientation of the sample on the learned embedding and quantify its similarity to each cluster
    
    In other words: Compare each novel instance to all other 'old' instances

    Input: 
        prefix_lf =  prefix to recognize latent factor coordinates (default: LF)
    
    Output: 
        l_orientation = list that features predictors expressing the relationship 
            between the sample and each cluster 
        df_meta = metadata of original sample population + newly projected sample
    
    """
    # Debug: check if there are identifiers -> distance calculation will break for that that 
    if len(list(z_existent[[i.sum()==0 for i in z_existent.values]].index)) > 0:
        print('(Negative control failed) Samples w/ only zeroes in latent space: ', list(z_existent[[i.sum()==0 for i in z_existent.values]].index))
    
    
    # Bookmark all orientation info
    l_orientation = []
    z_updated = projectSample(model, z_existent.copy(),  d_input, sample, prefix_lf)
    
    
    
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

def find_neighbours(df_meta, z_filtered, new_pat, idx=None, metric='cosine'):
    """
    Input:
        df_cluster = dataframe with cluster information
        z_filtered = product space created by MAUI
        new_pat = pseudoId of the projected patient
        l_pseudoId = list of pseudoIds; this should contain the ids of the development set 
            plus 1 extra id (of projected patient)
        metric = distance metric (default = cosine)
    
    Description:
        Find the most similar patients 
    """
    df_meta = pd.concat([df_meta, z_filtered], axis=1)
    
    # Compute the cosine similarity matrix
    sim_matrix = pairwise_distances(z_filtered.astype(np.float32), metric=metric)

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
    mask = np.triu(np.ones_like(sim_matrix_cluster, dtype=bool),1) # 1 below diagonal
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

def getOrientation(model, df_meta, z_existent, d_input, sample, sim_matrix=None, cluster_label='PhenoGraph_clusters', prefix_lf='LF'):
    """
    Description: 
    Discover the orientation of the sample on the learned embedding and quantify its similarity to each cluster
    
    In other words: Compare each novel instance to all other 'old' instances

    Input: 
        prefix_lf =  prefix to recognize latent factor coordinates (default: LF)
    
    Output: 
        l_orientation = list that features predictors expressing the relationship 
            between the sample and each cluster 
        df_meta = metadata of original sample population + newly projected sample
    
    """
    # Debug: check if there are identifiers -> distance calculation will break for that that 
    if len(list(z_existent[[i.sum()==0 for i in z_existent.values]].index)) > 0:
        print('(Negative control failed) Samples w/ only zeroes in latent space: ', list(z_existent[[i.sum()==0 for i in z_existent.values]].index))
    
    # Bookmark all orientation info
    l_orientation = []
    z_updated = projectSample(model, z_existent.copy(),  d_input, sample, prefix_lf)
    
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

def quantifySimilarity(df_cluster, sim_matrix, CLUSTER_LABEL = 'PhenoGraph_clusters'):
    """
    Calculate similarity characteristics between samples in the learned space 
    """
    
    N_CLUSTERS = len(np.unique(df_cluster[CLUSTER_LABEL]))
    
    archetype_columns = ['weight_pval', 'weight_mean', 'weight_sd', 'cluster_mean_pat', 'cluster_sd_pat'] # + latent factors?
    l_col = ['pseudoId', CLUSTER_LABEL]
    for i in range(N_CLUSTERS):
        l_col.extend(['%s_%s' % (col, i) for col in archetype_columns ])
    
    df_characteristics = pd.DataFrame(columns=l_col)
    #df_clustering
    #w_1, w_2, w_3, w_4 = l_weights[0], l_weights[1], l_weights[2], l_weights[3]


    N_CLUSTERS = len(df_cluster[CLUSTER_LABEL].unique())

    # Create trainingsset
    for idx, pat in enumerate(df_cluster['pseudoId']):
        l_row = [df_cluster.iloc[idx]['pseudoId'], df_cluster.iloc[idx][CLUSTER_LABEL]]
        l_prob = []
        l_p = []
        cluster = -1

        #print(N_CLUSTERS)
        for cluster_ix in range(N_CLUSTERS):
            #print('Cluster %s' % cluster_ix)
            # Get the indices of patients from specific cluster
            patient_indices = list(df_cluster[df_cluster[CLUSTER_LABEL]==cluster_ix].index)

            # Add the new patient if not already in cluster
            if idx not in patient_indices: 
                patient_indices.append(idx)


            # Keep seperate list (where we will exclude the patient of interest)
            cluster_indices = patient_indices.copy()
            # Remove patient that you want to predict
            #if idx in cluster_indices: 
            cluster_indices.remove(idx)

            # Create copy of product space
            # Subset similarity matrix on said cluster
            sim_matrix_child = pd.DataFrame(sim_matrix.copy()).loc[patient_indices, patient_indices]

            # Patient vs cluster
            patient_scores = list(sim_matrix_child[idx])
            patient_scores = sorted(patient_scores, reverse=True)
            patient_scores = patient_scores[1:] # remove comparison with self

            # Within cluster
            sim_matrix_child = np.array(pd.DataFrame(sim_matrix_child).loc[cluster_indices, cluster_indices])

            # Keep scores of all unique pairwise distances within cluster
            mask = np.triu(np.ones_like(sim_matrix_child, dtype=bool),1) # 1 below diagonal
            cluster_scores = np.array(sim_matrix_child[mask])
            tstat, pval = t_test(patient_scores, cluster_scores, alternative='less')

            # Define predictors (Be aware: we assume a normal distribution)
            pred_0 = pval #pval # H0 = patient is similar to cluster, Ha = patient is not similar
            pred_1 = np.mean(cluster_scores) # average cluster probability
            pred_2 = np.std(cluster_scores) # stability of the cluster
            pred_3 = np.mean(patient_scores) # average patient probability
            pred_4 = np.std(patient_scores) # stability of patient probability

            l_row.extend([pred_0, pred_1, pred_2, pred_3, pred_4])


        df_characteristics.loc[len(df_characteristics)] = l_row
    return df_characteristics

def identifyOutliers(df_orient, std_factor=3, cluster_label='PhenoGraph_clusters', prefix_lf='LF', id_label='pseudoId', repl_label='Replication'): 
    """
    Description: 
         Identify outliers within each cluster (according to the standard deviation).
         We only calculate centroid based on original cluster data.
    
    Input: 
        df_orient = dataframe with id, clus+ter information and latent factors
        std_factor= standard deviation cut-off used to define outliers 
        cluster_label = name of colum with cluster label
        prefix_lf =  prefix to recognize latent factor coordinates (default: LF)
        id_label = name of colum with id label (default: pseudoId)
        repl_label = name of colum with replication label (default: Replication)
    
    Output: 
        outliers_n = list of sample ids of the outliers
    """
    lf_col = [col for col in df_orient.columns if prefix_lf in col]
    
    outliers_n = []
    for c in df_orient[cluster_label].unique():
        # Get cluster samples and centroids
        centroid = np.mean(df_orient[((df_orient[cluster_label] == c) & (df_orient[repl_label] == 0))][lf_col].values, axis=0)
        cluster_samples = df_orient[df_orient[cluster_label] == c][lf_col].values

        # calculate intra cluster distance 
        intra_cluster_distances = [np.linalg.norm(sample - centroid) for sample in cluster_samples]
        mean_distance = np.mean(intra_cluster_distances)
        std_distance = np.std(intra_cluster_distances)
        
        # Identify deviants by predefined cut_off
        outliers = []
        outliers.extend([ix for ix, sample, dist in zip(range(len(cluster_samples)), cluster_samples, intra_cluster_distances)
            if dist > mean_distance + std_factor * std_distance])
        
        outliers_n.extend([df_orient[df_orient[cluster_label] == c][id_label].iloc[i] for i in outliers])
    return outliers_n