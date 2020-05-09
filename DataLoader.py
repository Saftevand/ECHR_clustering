import os
import json
import Document
from sklearn.feature_extraction import DictVectorizer
dataset_path = 'D:\datasets\ECHR_DATASET'
tfidf_folder_path = os.path.join(dataset_path, 'tfidf')
bow_folder_path = os.path.join(dataset_path, 'bow')
references_path = os.path.join(dataset_path, 'matrice_appnos.json')
cases_path = os.path.join(dataset_path, 'cases.json')



def load_dataset():
    reference_multikey_count = 0
    documents = []
    reference_dict = {}
    with open(references_path) as reference_file:
        reference_dict = json.load(reference_file)

    with open(cases_path) as json_file:
        cases_json = json.load(json_file)

        for index, case in enumerate(cases_json):
            application_number = case['appno']
            document_name = case['docname']
            document_id = case['itemid']
            with open(os.path.join(bow_folder_path, document_id + '_bow.txt')) as bow_file:
                bow_dict = {}
                for line in bow_file:
                    entries_arr = line.split()
                    for entry in entries_arr:
                        (key, value) = entry.split(':')
                        bow_dict[int(key)] = int(value)
            with open(os.path.join(tfidf_folder_path, document_id + '_tfidf.txt')) as tfidf_file:
                tfidf_dict = {}
                for line in tfidf_file:
                    entries_arr = line.split()
                    for entry in entries_arr:
                        (key, value) = entry.split(':')
                        tfidf_dict[int(key)] = float(value)
            reference_list = []
            try:
                references_for_document = reference_dict[application_number]
            except:
                matching_key = [key for key in reference_dict.keys() if application_number in key]
                references_for_document = reference_dict[matching_key[0]]
                reference_multikey_count += 1
            for key in references_for_document.keys():
                if key != application_number:
                    reference_list.append(key)

            document = Document.Document(application_id=application_number, document_id=document_id, title=document_name, bag_of_words=bow_dict, tf_idf=tfidf_dict, references=reference_list)
            documents.append(document)
            print(f'Loaded document: {index}')
    print(f'Reference_multikey_count: {reference_multikey_count}')
    print('Succesfully loaded dataset')
    return documents

def get_sparse_tfidf_matrix(documents):
    tfidf_dicts = []
    for document in documents:
        tfidf_dicts.append(document.tf_idf)
    vectorizer = DictVectorizer(sparse=True)
    sparse_tfidf = vectorizer.fit_transform(tfidf_dicts)
    feature_names = vectorizer.get_feature_names()
    return sparse_tfidf