import DataLoader
import Clustering
import pickle
import numpy
import Document
from sklearn.cluster import KMeans
import visualization


def main():

    #documents = DataLoader.load_dataset()
    #with open("documents.pkl", "wb") as documents_pickle:
    #    pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    #DataLoader.fix_references(documents)
    #DataLoader.find_bad_refs(documents)
    #with open("documents.pkl", "wb") as documents_pickle:
    #    pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)



    # this is text to disable advanced code assistance
    documents = []
    ref_matrix = None
    with open("documents.pkl", "rb") as documents_pickle:
        print("Loading documents")
        documents = pickle.load(documents_pickle)
    with open("sparse_tfidf.pkl", "rb") as documents_pickle:
        print("Loading tfidf")
        sparse_tfidf = pickle.load(documents_pickle)
    with open("cosine_sim.pkl", "rb") as documents_pickle:
        print("Loading cosine similarities")
        cosine_similarities = pickle.load(documents_pickle)
    #with open("clusters10.pkl", "rb") as documents_pickle:
    #    print("Loading km")
    #    km = pickle.load(documents_pickle)
    #with open("adj_matrix.pkl", "rb") as documents_pickle:
    #    print("Loading adj_matrix")
    #    adj_matrix = pickle.load(documents_pickle)




    k = 10

    km = KMeans(n_clusters=k)

    km.fit(sparse_tfidf)

    clusters = km.labels_.tolist()

    with open("clusters10.pkl", "wb") as documents_pickle:
       pickle.dump(km, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)





    def create_adj_matrix():
        m = len(documents)
        matrix = numpy.full((m, m), 0)
        for x in range(m):
            for y in range(m):
                if cosine_similarities[x][y] > 0.5:
                    matrix[x][y] = 1

        return matrix

    adj_matrix = create_adj_matrix()

    with open("adj_matrix.pkl", "wb") as documents_pickle:
       pickle.dump(adj_matrix, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    visualization.gui_graph_run(adj_matrix)


    #with open("reference_matrix.pkl", "rb") as references_pickle:
    #    print("Loading reference_matrix")
    #    ref_matrix = pickle.load(references_pickle)

    #results = Clustering.eigenDecomposition(ref_matrix)
    #print(results)

    #print("Calculating reference matrix")
    #ref_matrix = DataLoader.get_sparse_reference_matrix(documents)

    #print("Creating sparse tfidf")
    #sparse_tfidf = DataLoader.get_sparse_tfidf_matrix(documents)

    #with open("sparse_tfidf.pkl", "wb") as documents_pickle:
    #    pickle.dump(sparse_tfidf, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    #print("Calculating cosine-similarities")
    #cosine_similarities = Clustering.calc_cosine_similarity(sparse_tfidf)

    #with open("cosine_sim.pkl", "wb") as documents_pickle:
    #    pickle.dump(cosine_similarities, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Complete")

    '''
    test1 = Document.Document(application_id=1, document_id=1, title='one',
                                 bag_of_words=None, tf_idf=None, references={'4': '4', '5': '5', '3':'3'},
                                 related_appnos=['4', '5'], multiple_appnos=True)
    test2 = Document.Document(application_id=1, document_id=1, title='two',
                              bag_of_words=None, tf_idf=None, references={'4': '4', '5': '5', '6':'6'},
                              related_appnos=[], multiple_appnos=False)
    test = [test1, test2]
    print("stop")
    DataLoader.fix_references(test)
    print("stop")
    '''

if __name__ == "__main__":
    main()

