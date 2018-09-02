from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
import numpy as np

def create_coassociation_matrix(labels):
    rows = []
    cols = []
    unique_labels = set(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        for index1 in indices:
            for index2 in indices:
                rows.append(index1)
                cols.append(index2)
    data = np.ones((len(rows),))
    return csr_matrix((data, (rows, cols)), dtype='float')




def ensemble(all_clusters, cut_threshold=1): 
    C = sum((create_coassociation_matrix(all_clusters[i])
                 for i in range(all_clusters.shape[0])))
    C = C/all_clusters.shape[0]
    mst = minimum_spanning_tree(-C)
    print("MST thresholds", np.unique(mst.data, return_counts=True))
    mst.data[mst.data > -cut_threshold] = 0
    mst.eliminate_zeros()
    return connected_components(mst)