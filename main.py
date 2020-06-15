import DataLoader
import Clustering
import pickle
from sklearn.cluster import AffinityPropagation, KMeans
from sklearn import preprocessing
import Document

def save_all_files(cluster_labels, cluster_centers):
    #with open("reference_matrix.pkl", "wb") as handle:
    #    pickle.dump(reference_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("affinity_prop_labels.pkl", "wb") as handle:
        pickle.dump(cluster_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("affinity_prop_cluster_centers.pkl", "wb") as handle:
        pickle.dump(cluster_centers, handle, protocol=pickle.HIGHEST_PROTOCOL)

def main():

    docs = DataLoader.load_dataset()

    ''' fix
    with open("affinity_prop_labels.pkl", "rb") as labels_pickle:
        labels = pickle.load(labels_pickle)

    print(labels)


    with open("documents.pkl", "rb") as documents_pickle:
        documents = pickle.load(documents_pickle)
        print("1")
        DataLoader.fix_references(documents)
        print("2")
        DataLoader.find_bad_refs(documents)
        print("3")
        documents = DataLoader.remove_docs_no_in_out_refs(documents)
        with open("documents.pkl", "wb") as documents_pickle:
            pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    

    '''
    # this is text to disable advanced code assistance
    with open("documents.pkl", "rb") as documents_pickle:

        print("Loading documents")
        documents = pickle.load(documents_pickle)

    print("Creating sparse tfidf")
    sparse_tfidf = DataLoader.get_sparse_tfidf_matrix(documents)

    #sparse_norm = preprocessing.normalize(sparse_tfidf)

    #kmeans_res = KMeans(n_clusters=10, random_state=0).fit(sparse_norm)

    pred, cluster_centers, X = Clustering.cluster_LCA_kmeans(sparse_tfidf)




    save_all_files(pred, cluster_centers)

    #print("Calculating cosine-similarities")
    #cosine_similarities = Clustering.calc_cosine_similarity(sparse_tfidf)

    print("Complete")
    '''


    big_dict = {}
    for doc in documents:
        big_dict[doc.application_id] = doc

    all_references = set()
    for doc in documents:
        for key in doc.references.keys():
            all_references.add(key)
            other_doc = big_dict[key]
            other_doc.references[key] = 1

    for doc in documents:
        if doc.application_id not in all_references:
            print("problem")

    DataLoader.find_bad_refs(documents)

    smaller_docs = DataLoader.remove_docs_no_in_out_refs(documents)
    print(smaller_docs.shape)


    print("Calculating reference matrix")
    ref_matrix = DataLoader.get_sparse_reference_matrix(documents)
    print(ref_matrix.shape)





    
    print("running affinity prop")
    clusters = AffinityPropagation().fit(ref_matrix)
    labels = clusters.labels_
    cluster_centers = clusters.cluster_centers_
    save_all_files(reference_matrix=ref_matrix, cluster_labels=labels, cluster_centers=cluster_centers)
'''




if __name__ == "__main__":
    main()

