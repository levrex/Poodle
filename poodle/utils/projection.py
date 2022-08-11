import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy.spatial.distance import cosine# cosine minkowski
from sklearn.metrics.pairwise import cosine_similarity
import time

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

def projectSample(maui_model, z_space, d_input, sample): # sample
    """
    Input:
        maui_model = Loaded autoencoder object with the learned product space 
        z_space = product space (based on original set)
    
        d_input = dictionary featuring columns in the original space for each modality
        sample = features of 1 patient from the replication set
        
    Description: 
    Project new sample onto the product space by employing previously trained MAUI mdoel
    """
    
    sample_cat = sample[d_input['cat']].copy()
    sample_num = sample[d_input['num']].copy()   
    new_pat = sample.name
    
    df_cat = pd.DataFrame(columns=list(sample_cat.index))
    df_cat.loc[0] = sample_cat
    
    df_num = pd.DataFrame(columns=list(sample_num.index))
    df_num.loc[0] = sample_num

    # Project new sample in product space
    t0 = time.time()
    # Combined analysis using MUSE
    #z = maui_model.transform({ 'Categorical': categoric_Full.T, 'Lab_numerical': numeric_Full.T}) # 'Mannequin_Counts': df_counts.T,
    z_patient = maui_model.transform({ 'Categorical': df_cat.T, 'Lab_numerical': df_num.T}) # 'Mannequin_Counts': df_counts.T,
    
    # Only select relevant latent factors
    z_patient = z_patient[z_space.columns]
    
    # Add new patient to product space
    z_space = z_space.append(z_patient, ignore_index = True)

    # add projection to product space
    t1 = time.time()
    #print('Time to project %s into product space: %s' % (new_pat, str(t1-t0))) # .as_matrix()
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

def mapping_to_cluster(df, sim_matrix, new_pat, cluster_ix=0, cluster_label='PhenoGraph_clusters', id_label='pseudoId'):
    """
    Description:
        Calculate the distances of a new patient to a specific cluster (cluster_ix), and 
        compare this distribution (distance of new patient vs all patients of cluster) to 
        the within cluster similarity (all pairwise distances of said cluster).
        
    
    Input: 
        df = pandas dataframe containing the metadata
        sim_matrix = distance/similarity matrix 
        new_pat = patient of interest
        l_pseudoId_replication = the pseudo identifiers used
        cluster_ix = the cluster of interest
        cluster_label = name of cluster columns
        id_label = name of patient columns
        
    Return:
        patient_scores = distance between the new patient and every patient in the specified cluster
        cluster_scores = all pairwise distances of patients within the same cluster
    
    ToDo: Fix hardcoded column names (such as pseudoId & PhenoGraph_clusters)
    """
    idx = None

    #Construct a reverse map of indices and movie titles
    indices = pd.Series(df.index, index=df[id_label]).drop_duplicates()

    if idx == None:
        # Get the index of current patient
        idx = indices[new_pat]
    else :
        idx = idx
    
    # Get the indices of patients from specific cluster
    cluster_indices = list(df[df[cluster_label]==cluster_ix].index)
    
    # Add the new patient
    patient_indices = cluster_indices.copy()
    patient_indices.append(idx)
    
    # Subset similarity matrix on said cluster
    sim_matrix = pd.DataFrame(sim_matrix).loc[patient_indices, patient_indices]
    
    # Patient vs cluster
    patient_scores = list(sim_matrix[idx])
    patient_scores = sorted(patient_scores, reverse=True)
    patient_scores = patient_scores[1:] # remove comparison with self

    # Within cluster
    sim_matrix = np.array(pd.DataFrame(sim_matrix).loc[cluster_indices, cluster_indices])
    # Keep scores of all unique pairwise distances within cluster
    mask = np.triu(np.ones_like(sim_matrix, dtype=np.bool),1) # 1 below diagonal
    cluster_scores =  np.array(sim_matrix[mask])

    return patient_scores, cluster_scores


def calculate_proba_pval(l_p, n_clusters, l_t=None, l_m=None, l_mp=None):
    """
    Calculate probabilities of patients belonging to each cluster
    according to the pairwise distributions
    
    l_p = list with pvalues
    n_clusters = The total number of clusters you have
    l_t = list with t statistic, we will weight these, unless it isn't provided
    l_m = list with cluster mean
    l_mp = list with patient mean
    """
    l_prob = []
    l_p = [0.99 if i == 1 else i for i in l_p ]
    
    if l_mp == None and l_m != None:
        l_mp = [1 for i in range(len(l_m))]
    
    # Perform inverse log transformation (to get probability)
    if l_t == None:
        l_prob = [1./np.log10(l_p[i]) for i in range(len(l_p))] 
    else : # weight t-statistic
        
        l_prob = [(1./np.log10(l_p[i]))*max(l_t[i], 1.) for i in range(len(l_p))] 
        
    if l_m == None:
        l_prob = [1./np.log10(l_p[i]) for i in range(len(l_p))] 
    else : # we value more pronounced representations more!
        l_prob = [(1./(np.log10(l_p[i]*l_m[i])))*l_mp[i] for i in range(len(l_p))]  # *l_mp[i]
    
    # Change any values close to zero 
    l_prob = [l_prob[i]/sum(l_prob) for i in range(len(l_prob))] 
    #print(l_prob)
    return l_prob

def calculate_proba_top10(df, n_clusters):
    """
    Calculate probabilities of patients belonging to each cluster 
    according to the top 10 neighbours
    
    n_clusters = The total number of clusters you have
    top_n = how many neighbours do you look at
    """
    top_n = 10
    l_prob = []
    for clus in range(n_clusters):
        l_prob.append(sum(df[df['PhenoGraph_clusters']==clus]['Similarity'])/top_n)
    return l_prob

def classifyPatient(new_pat, df, sim_matrix, N_CLUSTERS = 4):
    l_p = []
    l_t = []
    l_mean = []
    l_mean_pat = []

    for i in range(N_CLUSTERS):
        lut = {0: 'b', 1: 'y', 2: 'g', 3: 'r'}

        patient_scores, cluster_scores = mapping_to_cluster(df, sim_matrix, new_pat, cluster_ix=i)
        tstat, pval = t_test(patient_scores, cluster_scores, alternative='less')
        l_mean.append(np.mean(cluster_scores)) 
        l_mean_pat.append(np.mean(patient_scores)) 
        l_p.append(pval)
        l_t.append(tstat)
    return calculate_proba_pval(l_p, N_CLUSTERS, l_m=l_mean , l_mp=l_mean_pat) # , l_m=l_mean # , l_mp=l_mean_pat

def getClusterLabel(row):
    l_proba = [col for col in row.index if 'Proba_cluster' in col]
    l_val = row[l_proba].values
    return np.argmax(l_val) + 1
    
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

def predictPatientCluster(maui_model, df_meta, z_existent, d_input, sample, sim_matrix=None):
    new_pat = sample.name
    
    z_updated = projectSample(maui_model, z_existent.copy(),  d_input, sample) # df_categoric[l_cat], df_numeric
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
    
    # Classify patients
    l_prob = classifyPatient(new_pat, df_meta, sim_matrix_child)
    
    return l_prob, z_updated
