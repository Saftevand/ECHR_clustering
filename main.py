import DataLoader
import Clustering
import pickle


def main():

    #documents = DataLoader.load_dataset()
    #with open("documents.pkl", "wb") as documents_pickle:
    #    pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("documents.pkl", "rb") as documents_pickle:
        print("Loading documents")
        documents = pickle.load(documents_pickle)
        print("Creating sparse tfidf")
        sparse_tfidf = DataLoader.get_sparse_tfidf_matrix(documents)
        print("Calculating cosine-similarities")
        cosine_similarities = Clustering.calc_cosine_similarity(sparse_tfidf)
        print("Complete")




if __name__ == "__main__":
    main()

