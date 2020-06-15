class Document:

    def __init__(self, title, document_id, application_id, tf_idf, bag_of_words, multiple_appnos, references_appno, internal_appnos, related_appnos=None, references=None):
        allAppnos = internal_appnos.copy()
        allAppnos.extend(references_appno)
        self.all_refs = list(set(allAppnos))

        self.references_appno = references_appno
        self.internal_appnos = internal_appnos
        self.title = title
        self.document_id = document_id
        self.application_id = application_id
        self.tf_idf = tf_idf
        self.bag_of_words = bag_of_words
        # self.references = references
        # self.related_appnos = related_appnos
        self.multiple_appnos = multiple_appnos

        self.final_ref = {}

        self.outgoing_refs = {}




