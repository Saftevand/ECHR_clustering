import Clustering
import DataLoader
import visualization


def visualize_only():
    documents, labels, adj_matrix = DataLoader.load_data_for_visualize()
    visualization.doc_to_vec_visualize(documents=documents, labels=labels, adj_matrix=adj_matrix)

def run_pipeline():
    # Loads documents and TaggedDocuments (used for training)
    tagged_documents = DataLoader.load_tagged()
    all_documents = DataLoader.load_all_documents()
    adjacency_matrix_references_all_documents = DataLoader.load_adjacency_matrix_all_documents()

    # Trains doc2vec
    model = Clustering.train_doc2vec(tagged_data=tagged_documents, epochs_to_run=100, embedding_size=100)
    # Runs HDBSCAN, returns a list of labels (a label for each documents. -1 == outlier)
    labels = Clustering.run_hdbscan(model=model, min_cluster_size=4, min_samples=4)

    # Extracts the documents which have been clustered such that we have no outliers
    # Mask denotes the ones to include and exclude. Labels of the clustered documents and the clustered documents
    mask, labels_subset, clustered_documents = Clustering.extract_clustered_documents(all_documents, labels)

    # Creates the adjacency matrix for references between clusters
    cluster_references_adjacency = Clustering.create_adjacency_matrix_for_clusters(mask=mask, labels=labels_subset,
                                                                                   adjacency_references_all_documents=
                                                                                   adjacency_matrix_references_all_documents)

    # k-nearest undirected adjacency
    cluster_references_adjacency = Clustering.make_adjacency_matrix_undirected(cluster_references_adjacency, k=3)
    DataLoader.save_data(cluster_references_adjacency, clustered_documents, labels_subset)

    # Creates the graph and sets up the interactive webpage showing the graph
    visualization.doc_to_vec_visualize(documents=clustered_documents, adj_matrix=cluster_references_adjacency, labels=labels_subset)

def run_pipeline_with_pretrained_doc2vec():
    all_documents = DataLoader.load_all_documents()
    adjacency_matrix_references_all_documents = DataLoader.load_adjacency_matrix_all_documents()
    # Load model
    model = DataLoader.load_model()
    # Runs HDBSCAN, returns a list of labels (a label for each documents. -1 == outlier)
    labels = Clustering.run_hdbscan(model=model, min_cluster_size=4, min_samples=4)

    # Extracts the documents which have been clustered such that we have no outliers
    # Mask denotes the ones to include and exclude. Labels of the clustered documents and the clustered documents
    mask, labels_subset, clustered_documents = Clustering.extract_clustered_documents(all_documents, labels)

    # Creates the adjacency matrix for references between clusters
    cluster_references_adjacency = Clustering.create_adjacency_matrix_for_clusters(mask=mask, labels=labels_subset,
                                                                                   adjacency_references_all_documents=
                                                                                   adjacency_matrix_references_all_documents)

    # k-nearest undirected adjacency
    cluster_references_adjacency = Clustering.make_adjacency_matrix_undirected(cluster_references_adjacency, k=3)
    DataLoader.save_data(cluster_references_adjacency, clustered_documents, labels_subset)

    # Creates the graph and sets up the interactive webpage showing the graph
    visualization.doc_to_vec_visualize(documents=clustered_documents, adj_matrix=cluster_references_adjacency,
                                       labels=labels_subset)

def main():
    # If train is True, the full pipeline will be run, where a clustering model is trained. This will take quite some time
    train = False
    # Runs pipeline with a loaded model. Set this to True and Train to False if you want to test the parameters of HDBSCAN
    HDBSCAN = False

    if train:
        # Run training
        run_pipeline()
    else:
        if HDBSCAN:
            run_pipeline_with_pretrained_doc2vec()
        else:
            # Visualize only - Loads all the pretrained and preprosseced files needed to show the graph
            visualize_only()


if __name__ == '__main__':
    main()