import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from poodle import utils as pup
from sklearn.metrics.pairwise import cosine_similarity


def plotQualityControl(df_old, df_new, z_old, z_new):
    """
    Description: 
        Compare within variance of old clusters to within variance of new clusters
        
    
    ToDo: maybe use this to optimize?
    ToDo: ensure this function works with more clusters
    
    """
    sim_old = cosine_similarity(z_old.astype(np.float32))
    sim_new = cosine_similarity(z_new.astype(np.float32))

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle('Within similarity in clusters vs replicate clusters', fontsize=16)

    N_CLUSTERS = 4

    l_p = []
    
    for i in range(N_CLUSTERS):
        lut = {0: 'b', 1: 'y', 2: 'g', 3: 'r'}
        
        
        # Get the indices of patients from specific cluster
        cluster_indices_old = list(df_old[df_old['PhenoGraph_clusters']==i].index)
        
        # Get the indices of patients from specific cluster
        cluster_indices_new = list(df_new[df_new['PhenoGraph_clusters']==i].index)
        
        # Maybe create standalone function at some point
        cluster_scores_old = pup.getSimilarityWithin(sim_old, cluster_indices_old)
        cluster_scores_new = pup.getSimilarityWithin(sim_new, cluster_indices_new)
        tstat, pval = pup.t_test(cluster_scores_new, cluster_scores_old, alternative='less') # both-sided
        l_p.append(pval)

        if i == 0:
            axs[int((i /2) >= 1), int((i % 2) != 0)].hist(cluster_scores_new, color='gray', range=[0, 1], 
                                                          alpha=.99, bins=30, label='Replicate', density=True)
        else :
            axs[int((i /2) >= 1), int((i % 2) != 0)].hist(cluster_scores_new, color='gray', range=[0, 1], 
                                                          alpha=.99, bins=30, density=True)

        axs[int((i /2) >= 1), int((i % 2) != 0)].hist(cluster_scores_old, color=lut[i], range=[0, 1], 
                                                      alpha=.6, bins=30, label='Within cluster %s' % str(i+1), density=True) 
        axs[int((i /2) >= 1), int((i % 2) != 0)].set_title('Cluster %s' % str(i+1))

        # Add anchored text
        anchored_text = AnchoredText("P-val:  %s\nT-stat: %.1f" % (f"{pval:.03g}", tstat), loc=3)
        axs[int((i /2) >= 1), int((i % 2) != 0)].add_artist(anchored_text)
    
    # Create legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(i, []) for i in zip(*lines_labels)]
    
    fig.legend(lines, labels)

def plotClusterMapping(df, z_filtered, new_pat):
    """
    Description: 
        Visualize the distribution of a patient (distances of a new patient to a specific cluster (cluster_ix))
        and compare this distribution to the within cluster similarity (all pairwise distances of said cluster).
    
    Input: 
        df = pandas dataframe containing the metadata
        z_filtered = shared product space (also containing the projected patient (new_pat))
        new_pat = identifier of newly projected patient
    
    ToDo: 
        adjust the function so you can easily map for more than 4 clusters (currently hard limit)
    """
    sim_matrix = cosine_similarity(z_filtered.astype(np.float32))

    fig, axs = plt.subplots(2, 2, figsize=(10,10))
    fig.suptitle('Cluster mapping patient %s' % (new_pat), fontsize=16)

    N_CLUSTERS = 4

    l_p = []
    
    for i in range(N_CLUSTERS):

        lut = {0: 'b', 1: 'y', 2: 'g', 3: 'r'}
        
        patient_scores, cluster_scores = pup.similarityToCluster(df, sim_matrix, cluster_ix=i, output_dist=True)
        tstat, pval = pup.t_test(patient_scores, cluster_scores, alternative='less')
        l_p.append(pval)

        if i == 0:
            axs[int((i /2) >= 1), int((i % 2) != 0)].hist(patient_scores, color='gray', range=[0, 1], 
                                                          alpha=.99, bins=30, label='Projected patient', density=True)
        else :
            axs[int((i /2) >= 1), int((i % 2) != 0)].hist(patient_scores, color='gray', range=[0, 1], 
                                                          alpha=.99, bins=30, density=True)

        axs[int((i /2) >= 1), int((i % 2) != 0)].hist(cluster_scores, color=lut[i], range=[0, 1], 
                                                      alpha=.6, bins=30, label='Within cluster %s' % str(i+1), density=True) 
        axs[int((i /2) >= 1), int((i % 2) != 0)].set_title('Cluster %s' % str(i+1))

        # Add anchored text
        anchored_text = AnchoredText("P-val:  %s\nT-stat: %.1f" % (f"{pval:.03g}", tstat), loc=3)
        axs[int((i /2) >= 1), int((i % 2) != 0)].add_artist(anchored_text)
    
    # Create legend
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(i, []) for i in zip(*lines_labels)]
    
    fig.legend(lines, labels)
    return 

def plot_neighbours(df_neighbours, new_pat):
    """
    Description: 
        Plot the top 10 closest neighbours with respect to the specified patient (new_pat)
    
    Input: 
        df_neighbours = dataframe containing the top 10 neighbours
        new_pat = identifier of newly projected patient
         
    ToDo:
        add more colors
    """
    
    # D_col specifies color per cluster. Add extra if you have more clusters (e.g. 4:'m')
    d_col = {0: 'b', 1: 'y', 2: 'g', 3: 'r'} 
    
    height = df_neighbours['Similarity']
    bars = df_neighbours['pseudoId']
    c = df_neighbours['PhenoGraph_clusters'].apply(lambda x : d_col[x])
    y_pos = np.arange(len(bars))

    # Show top 10 neighbours
    plt.bar(y_pos, height, color = c)
    plt.title("Patient %s's neighbours in latent space (top 10)" % (new_pat))
    plt.ylim(0, 1)
    plt.xticks(y_pos, bars, rotation = 90)
    
    # make legend
    l_patches = []
    for ix in range(len(d_col)):
        l_patches.append(mpatches.Patch(color=d_col[ix], label='Cluster %s' % str(ix+1)))
    plt.legend(handles=l_patches)
    
    plt.show()
    
def plotSpatialVariation(l_new, l_old, title='Compare spatial variance Exploratory vs Replication set'):
    """ 
    Description:
        Plot the cluster proportions per set
    Input:
        l_new = list featuring the proportions of each cluster in the replication set
        l_old = list featuring the proportions of each cluster in the exploratory set
    """
    x = np.arange(1, len(l_new)+1)  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, l_old, width, label='Exploratory set')
    rects2 = ax.bar(x + width/2, l_new, width, label='Replication set')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Quantity')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend()

    fig.tight_layout()

    plt.show()
