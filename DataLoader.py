import os
import json
import Document
from sklearn.feature_extraction import DictVectorizer

dataset_path = 'D:\AI-for-the-people\structured' #'D:\datasets\ECHR_DATASET'
tfidf_folder_path = os.path.join(dataset_path, 'tfidf')
bow_folder_path = os.path.join(dataset_path, 'bow')
references_path = os.path.join(dataset_path, 'matrice_appnos.json')
cases_path = os.path.join(dataset_path, 'cases.json')

def validate_json_load():
    with open(references_path) as reference_file:
        reference_dict = json.load(reference_file)
    print(len(reference_dict.keys()))



def load_dataset():
    reference_multikey_count = 0
    documents = []
    debug_count = 0

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

            # all_oppnos_for_document = application_number.split(';')

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
                                         title=document_name, bag_of_words=bow_dict, tf_idf=tfidf_dict,
                                         references_appno=multi_refs,
                                         internal_appnos=temp_internal_appnos,
                                         multiple_appnos=multiple_appnos)  # related_appnos=remaining_appnos
            documents.append(document)
            print(f'Loaded document: {index}')
            '''
            debug_count += 1
            if debug_count == 1000:
                docCount = 1
                for doc in documents:
                    print(f'reffing doc {docCount}')
                    docCount += 1
                    for outgoing in doc.all_refs:
                        for other_doc in documents:
                            if outgoing in other_doc.internal_appnos:
                                doc.outgoing_refs[other_doc.document_id] = 1
                print("stopo")
            '''

    for doc in documents:
        # tjekker alle refs (internal og outgoing)
        for outgoing in doc.all_refs:
            for other_doc in documents:
                # hvis appno er i andet docs internal appno --> ref fra doc til doc
                if outgoing in other_doc.internal_appnos:
                    doc.outgoing_refs[other_doc.document_id] = 1
    print("test")



    '''
            try:
                references_for_document = reference_dict[application_number]
                remaining_appnos = []
            except:
                # a key can contain multiple appnos --> references to one of these appnos references them all.
                # 1 appnos = one document --> part 1, 2, 3 etc.
                multiple_appnos = True

                # TODO problem her. application number må ikke søges efter sådan. når manleder efter fx 223 vil case 4223 også blive accepteret. kan ikke reddes med f';{application_number};' desværre
                matching_key = [key for key in reference_dict.keys() if application_number in key.split(';')]
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
            

            document = Document.Document(application_id=application_number, document_id=document_id, title=document_name, bag_of_words=bow_dict, tf_idf=tfidf_dict, references_appno=references_for_document, internal_appnos=all_oppnos_for_document, multiple_appnos=multiple_appnos) # related_appnos=remaining_appnos
            documents.append(document)
            print(f'Loaded document: {index}')
            '''

    print(f'Reference_multikey_count: {reference_multikey_count}')
    print('Succesfully loaded dataset')

    for doc in documents:
        refs = doc.all_refs
        for ref in refs:
            for other_doc in documents:
                # Hvis other doc indeholder appno vi leder efter sætter vi en ref
                if ref in other_doc.internal_appnos:
                    doc.final_ref[other_doc.document_id] = 1
                    break



    return documents

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
            print(f'Entry {remove} has bad out ref')
            del doc.references[remove]


def remove_docs_no_in_out_refs(documents):
    no_outgoing = []
    for doc in documents:
        l = len(doc.references.keys())
        if l == 0:
            no_outgoing.append(doc.application_id)

    print(len(no_outgoing))

    for doc in documents:
        for key in doc.references.keys():
            if key in no_outgoing:
                no_outgoing.remove(key)
                if len(no_outgoing) == 0:
                    break

    if len(no_outgoing) == 0:
        print("no_outgoing empty before return")
        return documents
    else:
        print("working")
        return [x for x in documents if x.application_id not in no_outgoing]


