import DataLoader
import numpy as np
import pickle
from collections import defaultdict
import networkx as nx
import hdbscan
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def create_documents_for_clustering(documents, labels=None, mask=None):

    clustered_documents = []
    for idx, include in enumerate(mask):
        if include:
            doc = documents[idx]
            clustered_documents.append(doc)
    for idx, label in enumerate(labels):
        clustered_documents[idx].cluster = label

    with open("clustered_documents.pkl", "wb") as tagged_pickle:
        pickle.dump(clustered_documents, tagged_pickle)
    return clustered_documents


# Creates reference adjacency matrix of the clusters found by HDBSCAN
def create_adjacency_matrix_for_clusters(mask, labels, adjacency_references_all_documents):

    np.fill_diagonal(adjacency_references_all_documents, 0)
    number_of_clusters = len((set(labels)))
    edges_dict = defaultdict(lambda: defaultdict(int))
    clustered = adjacency_references_all_documents[mask]
    clustered = np.transpose(np.transpose(clustered)[mask])
    for idx_from, label_from in enumerate(labels):
        for idx_to, label_to in enumerate(labels):
            if clustered[idx_from][idx_to] != 0:
                edges_dict[labels[idx_from]][labels[idx_to]] += 1
    print("Created edges_dict")
    clusters_adjacency = np.zeros((number_of_clusters, number_of_clusters))

    for cluster_from, val in edges_dict.items():
        for cluster_to, edges_count in val.items():
            clusters_adjacency[cluster_from][cluster_to] = edges_count

    np.save('cluster_adjacency.npy', clusters_adjacency)
    return clusters_adjacency
    # for idx, entry in enumerate(mask):


def extract_clustered_documents(documents, labels_full):
    indeces_of_clustered = np.where(labels_full != -1)[0]
    mask = np.where(labels_full != -1, True, False)
    labels_subset = labels_full[mask]
    subset_data = []
    for idx in indeces_of_clustered:
        subset_data.append(documents[idx])
    for idx, label in enumerate(labels_subset):
        subset_data[idx].cluster = label

    return mask, labels_subset, subset_data


def make_adjacency_matrix_undirected(adjacency_matrix, k=3):
    shape = adjacency_matrix.shape[0]

    for outer_idx in range(shape):
        for inner_idx in range(shape):
            a = adjacency_matrix[outer_idx][inner_idx]
            b = adjacency_matrix[inner_idx][outer_idx]
            total = a + b
            adjacency_matrix[outer_idx][inner_idx] = total
            adjacency_matrix[inner_idx][outer_idx] = total
    diag = np.diag(adjacency_matrix).copy()
    np.fill_diagonal(adjacency_matrix, 0)
    max_value = adjacency_matrix.max()
    adjacency_matrix = adjacency_matrix / max_value
    for outer_idx in range(shape):
        largest_values = []
        for inner_idx in range(shape):

            a = adjacency_matrix[outer_idx][inner_idx]
            if a > 0:
                largest_values.append((a, inner_idx))
        largest_sorted = sorted(largest_values, reverse=True)
        largest_sorted = largest_sorted[:k]
        adjacency_matrix[outer_idx, :] = 0
        for val, index in largest_sorted:
            adjacency_matrix[outer_idx][index] = val
    for index in range(shape):
        diag_val = diag[index]
        adjacency_matrix[index][index] = diag_val
    return  adjacency_matrix




# Runs hdbscan with default parameters
def run_hdbscan(model, min_cluster_size = 4, min_samples = 4):

    clusterer = hdbscan.HDBSCAN(metric='cosine', algorithm='generic', min_cluster_size=min_cluster_size, min_samples=min_samples)
    clusterer.fit(model.docvecs.vectors_docs.astype('double'))
    labels = clusterer.labels_
    with open('labels.npy', 'wb') as f:
        np.save(f, labels)
    return labels


def create_graph(tagged_data, model, labels):
    graph = nx.Graph()
    nodes_in_cluster = []
    #Add nodes
    for idx, doc in enumerate(tagged_data):
        if labels[idx] != -1:
            graph.add_node(doc.tags[0])
            nodes_in_cluster.append(doc.tags[0])

    #Add edges
    for idx, doc in enumerate(tagged_data):
        if labels[idx] != -1:
            current_document = doc.tags[0]
            ten_nearest_documents = model.docvecs.most_similar(doc.tags)
            for (neighbour_tag, neighbour_similarity) in ten_nearest_documents:
                if neighbour_tag in nodes_in_cluster:
                    graph.add_edge(current_document, neighbour_tag, weight=neighbour_similarity)
            #print(model.docvecs.most_similar(doc.tags))
    print("done")
    return graph


# Creates a json representation that can be used to create the graph with amcharts4
def clusters_to_json():
    adjacency = np.load('adjacency_matrix_normalised_k_nearest.npy')
    np.fill_diagonal(adjacency, 0)
    documents = pickle.load(open("clustered_documents.pkl", "rb"))
    clusters = defaultdict(list)
    clusters_itemid_pagerank = defaultdict()
    index_to_document_id_cluster = {}
    for doc in documents:
        clusters[doc.cluster].append(doc)
    for key, values in clusters.items():
        sorted_by_pagerank = sorted(values, key= lambda x: x.pagerank, reverse=True)
        doc_determing_cluster = sorted_by_pagerank[0]
        clusters_itemid_pagerank[doc_determing_cluster.document_id] = sorted_by_pagerank
        index_to_document_id_cluster[key] = doc_determing_cluster.document_id
    cluster_doc_ids = list(index_to_document_id_cluster.values())
    list_of_dicts = []
    for idx, (key, val) in enumerate(clusters_itemid_pagerank.items()):

        current_dict = {"name": key, "value": len(val), "children": [], "linkWith": []}
        for adj_idx ,element in enumerate(adjacency[idx]):
            if element !=0:
                current_dict["linkWith"].append(cluster_doc_ids[adj_idx])
        for doc in val[1:]:
            child_dict = {"name": doc.document_id, "value":1}
            current_dict["children"].append(child_dict)
        list_of_dicts.append(current_dict)

    with open("graph.json", "w") as write_file:
        json.dump(list_of_dicts, write_file)


# Given TaggedDocuments i.e the documents and their labels doc2vec is trained.
def train_doc2vec(tagged_data, epochs_to_run=100, embedding_size=100):
    max_epochs = epochs_to_run
    vec_size = embedding_size
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    dm=1)
    print("doc2vec_building_vocabulary")
    model.build_vocab(tagged_data)
    print(f'doc2vec starting training for {max_epochs} epochs')
    for epoch in range(max_epochs):
        print('Epoch {0}'.format(epoch))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    print('Training complete')
    model.save("d2v.model")
    return model
