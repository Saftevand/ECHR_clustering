import os
import json
import Document
import pickle
import numpy as np
import json
import glob
import networkx as nx
from sklearn.feature_extraction import DictVectorizer
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize


json_path = 'D:\\datasets\\ECHR-OD_process-develop\\build\\echr_database\\preprocessed_documents'
dataset_raw_documents_path = 'D:\\datasets\\ECHR-OD_process-develop\\build\echr_database\\raw_documents\\test'
full_txt_dir = 'D:\\datasets\\ECHR-OD_process-develop\\build\\echr_database\\preprocessed_documents\\full_txt'
tokenized_txt_dir = 'D:\\datasets\\ECHR-OD_process-develop\\build\\echr_database\\preprocessed_documents\\tokenized'


#dataset_path = 'E:\Job AAU\ECHR-OD_process-develop\\build\echr_database\structured' # 'D:\AI-for-the-people\structured'
dataset_path = 'D:\datasets\ECHR_DATASET'
references_path = os.path.join(dataset_path, 'matrice_appnos.json')
cases_path = os.path.join(dataset_path, 'cases.json')

# Creates documents from ECHR dataset
def load_dataset():
    documents = []

    with open(references_path) as reference_file:
        reference_dict = json.load(reference_file)

    with open(cases_path) as json_file:
        cases_json = json.load(json_file)

        for index, case in enumerate(cases_json):

            application_number = case['appno']
            document_name = case['docname']
            document_id = case['itemid']

            multiple_appnos = False

            # Alle keys hvor application number indgår
            matching_key = [key for key in reference_dict.keys() if application_number in key.split(';')]
            multi_refs = []
            temp_internal_appnos = []
            for k in matching_key:
                multi_refs.extend(reference_dict[k].keys())

                split = k.split(';')
                temp_internal_appnos.extend(split)

            # alle outgoing refs fra "matching keys"
            multi_refs = remove_dups(multi_refs)

            # alle appnos i hver "matching key"
            temp_internal_appnos = remove_dups(temp_internal_appnos)

            document = Document.Document(application_id=application_number, document_id=document_id,
                                         title=document_name,
                                         references_appno=multi_refs,
                                         internal_appnos=temp_internal_appnos,
                                         multiple_appnos=multiple_appnos)  # related_appnos=remaining_appnos
            documents.append(document)
            print(f'Loaded document: {index}')
    for count, doc in enumerate(documents):
        print(f'reffing doc {count}')
        # tjekker alle refs (internal og outgoing)
        for outgoing in doc.all_refs:
            for other_doc in documents:
                # hvis appno er i andet docs internal appno --> ref fra doc til doc
                if outgoing in other_doc.internal_appnos:
                    doc.outgoing_refs[other_doc.document_id] = 1
                    break
    documents = add_articles_and_conclusion_to_documents(documents)
    adjacency_matrix = create_adjacency_matrix_from_references(documents)

    assign_pagerank_to_documents(documents, adjacency_matrix)

    return documents


def save_data(adjacency_matrix, clustered_documents, labels):
    np.save('adjacency_matrix_normalised_k_nearest.npy', adjacency_matrix)
    np.save('labels_clustered.npy', labels)
    with open('clustered_documents.pkl', "wb") as file:
        pickle.dump(clustered_documents, file)

def assign_pagerank_to_documents(documents, adjacency_matrix):
    np.fill_diagonal(adjacency_matrix, 0)

    graph = nx.from_numpy_array(adjacency_matrix)
    pagerank_dict = nx.pagerank(graph)

    for idx, doc in enumerate(documents):
        doc.pagerank = pagerank_dict[idx]


def load_all_documents():
    documents = pickle.load(open('documents.pkl', "rb"))
    return documents


def load_adjacency_matrix_all_documents():
    adjacency = pickle.load(open("adjacency_references_all_documents.pkl", "rb"))
    return adjacency.toarray()

def add_articles_and_conclusion_to_documents(documents):
    documents_corrected = []
    for index, doc in enumerate(documents):
        with open(os.path.join(json_path, doc.document_id + '_parsed.json'), "r", encoding='utf-8') as read_file:
            data = json.load(read_file)
            if index % 1000 == 0:
                print(index)
        articles = data['__articles']
        conclusion = data['__conclusion']
        if not articles:
            for dict in data['conclusion']:
                for key, val in dict.items():
                    if key == 'article':
                        articles += val + ' '

        doc.articles = articles
        doc.conclusion = conclusion
        documents_corrected.append(doc)

    return documents_corrected


# Creates the txt files used for training doc2vec.
def documents_to_txt():
    json_documents = glob.glob(os.path.join(json_path, '*.json'))
    for index, json_document in enumerate(json_documents):
        with open(os.path.join(json_path, json_document), "r", encoding='utf-8') as read_file:
            data = json.load(read_file)
            if index % 1000 == 0:
                print(index)

        articles = 'Articles: '
        conclusion = 'Conclusion: '
        articles += data['__articles'] + ' '
        conclusion += data['__conclusion'] + ' '

        itemid = data['itemid']
        corrected_content = []
        corrected_content.append(articles)
        corrected_content.append(conclusion)
        content = data['content'][itemid + '.docx']

        corrected_content = lookup(content, corrected_content)
        final_string = ' '.join(corrected_content)
        with open(os.path.join(full_txt_dir, itemid + '.txt'), "w", encoding='utf-8') as output_file:
            output_file.write(final_string)


def tokenize_txt_documents():
    docs = []
    txt_documents = glob.glob(os.path.join(full_txt_dir, '*.txt'))
    for index, txt_document in enumerate(txt_documents):
        if index % 1000 == 0:
            print(index)
        tag = txt_document.split('.txt')[0].split('\\')[-1]
        with open(os.path.join(json_path, txt_document), "r", encoding='utf-8') as read_file:
            docs.append((word_tokenize((read_file.read().lower())), tag))
    tagged_data = [TaggedDocument(words=doc, tags=[tag]) for (doc, tag) in docs]
    with open(os.path.join(tokenized_txt_dir, "tagged_documents.pickle"), "wb") as tagged_pickle:
        pickle.dump(tagged_data, tagged_pickle)
    return tagged_data


# Recursive method for extracting paragraphs
def lookup(content, string_list=[]):
    for sub_dict in content:
        for key, val in sub_dict.items():
            if isinstance(val, str):
                string_list.append(val)
            if isinstance(val, list) and val:
                lookup(val, string_list)
    return string_list


def load_tagged(path="tagged_documents.pickle"):
    tagged_data = pickle.load(open(path, "rb"))
    print("Loaded tagged_data")
    return tagged_data

def load_data_for_visualize():

    with open("clustered_documents.pkl", "rb") as documents_pickle:
        print("Loading documents")
        documents = pickle.load(documents_pickle)

    adj_matrix = np.load('adjacency_matrix_normalised_k_nearest.npy')
    labels = np.load('labels_clustered.npy')
    return documents, labels, adj_matrix



def create_adjacency_matrix_from_references(documents):
    reference_dicts = []
    for doc in documents:
        reference_dicts.append(doc.outgoing_refs)
    vectorizer = DictVectorizer(sparse=True)
    sparse_refs = vectorizer.fit_transform(reference_dicts)
    return sparse_refs.toarray()


def remove_dups(some_list):
    temp = set(some_list)
    return list(temp)


def get_sparse_tfidf_matrix(documents):
    tfidf_dicts = []
    for document in documents:
        tfidf_dicts.append(document.tf_idf)
    vectorizer = DictVectorizer(sparse=True)
    sparse_tfidf = vectorizer.fit_transform(tfidf_dicts)
    feature_names = vectorizer.get_feature_names()
    return sparse_tfidf


def fix_references(documents):
    # fixing cases with extra appnos pr document
    for current_doc in documents:
        if current_doc.multiple_appnos is True:
            # assumption based on random samples: if some doc refs one related_appnos in a document they also ref the rest
            first_appno = current_doc.related_appnos[0]

            # delete references in other documents
            for other_doc in documents:
                if first_appno in other_doc.references:
                    for app_num in current_doc.related_appnos:
                        if app_num in other_doc.references.keys():
                            del other_doc.references[app_num]
                        else:
                            print(f'ref missing: {app_num} in {other_doc.application_id}')


def find_bad_refs(documents):
    set_appnos = set()
    for doc in documents:
        set_appnos.add(doc.application_id)
    for doc in documents:
        remove_list = []
        for ref in doc.references.keys():
            if ref not in set_appnos:
                remove_list.append(ref)
        for remove in remove_list:
            del doc.references[remove]

