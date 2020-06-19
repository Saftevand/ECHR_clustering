class Document:

    def __init__(self, title, document_id, application_id, multiple_appnos, references_appno,
                 internal_appnos):
        allAppnos = internal_appnos.copy()
        allAppnos.extend(references_appno)
        self.all_refs = list(set(allAppnos))

        self.references_appno = references_appno
        self.internal_appnos = internal_appnos
        self.title = title
        self.document_id = document_id
        self.application_id = application_id
        self.articles = None
        self.conclusion = None
        self.multiple_appnos = multiple_appnos
        self.cluster = None
        self.pagerank = None
        self.final_ref = {}

        self.outgoing_refs = {self.document_id: 1}