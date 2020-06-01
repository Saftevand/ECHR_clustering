import os
import json
import Document
from sklearn.feature_extraction import DictVectorizer

dataset_path = 'D:\AI-for-the-people\structured' #'D:\datasets\ECHR_DATASET'
tfidf_folder_path = os.path.join(dataset_path, 'tfidf')
bow_folder_path = os.path.join(dataset_path, 'bow')
references_path = os.path.join(dataset_path, 'matrice_appnos.json')
cases_path = os.path.join(dataset_path, 'cases.json')


def load_dataset():
    reference_multikey_count = 0
    documents = []

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
            multiple_appnos = False

            try:
                references_for_document = reference_dict[application_number]
                remaining_appnos = []
            except:
                # a key can contain multiple appnos --> references to one of these appnos references them all.
                # 1 appnos = one document --> part 1, 2, 3 etc.
                multiple_appnos = True

                # TODO problem her. application number må ikke søges efter sådan. når manleder efter fx 223 vil case 4223 også blive accepteret. kan ikke reddes med f';{application_number};' desværre
                matching_key = [key for key in reference_dict.keys() if application_number in key]
                # find 7714/06 --> 27714/06;3213/12,   3848;7714/06
                if len(matching_key) > 1:
                    # this, hopefully does not happen
                    for i, key in enumerate(matching_key):
                        app_split = key.split(';')
                        for part in app_split:
                            if part == application_number:
                                correct = i
                                break

                    matching_key = [matching_key[correct]]


                    print(f'Found more than one dict entry containing same appno: {matching_key}')
                    if len(matching_key) == 1:
                        print("fixed it tho")

                split_keys = matching_key[0].split(';')

                references_for_document = reference_dict[matching_key[0]]

                # remove reference to self
                #del references_for_document[application_number]

                # remaining appnos will be contained in document class for future display.. possibly
                remaining_appnos = split_keys[1:]
                reference_multikey_count += 1

            document = Document.Document(application_id=application_number, document_id=document_id, title=document_name, bag_of_words=bow_dict, tf_idf=tfidf_dict, references=references_for_document, related_appnos=remaining_appnos, multiple_appnos=multiple_appnos)
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

def get_sparse_reference_matrix(documents):
    reference_dicts = []
    for doc in documents:
        reference_dicts.append(doc.references)
    vectorizer = DictVectorizer(sparse=True)
    sparse_refs = vectorizer.fit_transform(reference_dicts)
    return sparse_refs

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
