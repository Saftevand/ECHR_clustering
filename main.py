import DataLoader
import Clustering
import pickle
import Document


def main():
    '''
    documents = DataLoader.load_dataset()
    with open("documents.pkl", "wb") as documents_pickle:
        pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    DataLoader.fix_references(documents)
    DataLoader.find_bad_refs(documents)
    with open("documents.pkl", "wb") as documents_pickle:
        pickle.dump(documents, documents_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    '''


    # this is text to disable advanced code assistance
    with open("documents.pkl", "rb") as documents_pickle:

        print("Loading documents")
        documents = pickle.load(documents_pickle)

        print("Calculating reference matrix")
        ref_matrix = DataLoader.get_sparse_reference_matrix(documents)

        print("Creating sparse tfidf")
        sparse_tfidf = DataLoader.get_sparse_tfidf_matrix(documents)

        print("Calculating cosine-similarities")
        cosine_similarities = Clustering.calc_cosine_similarity(sparse_tfidf)



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

