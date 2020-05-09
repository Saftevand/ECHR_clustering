class Document:

    def __init__(self, title, document_id, application_id, tf_idf, bag_of_words, references):
        self.title = title
        self.document_id = document_id
        self.application_id = application_id
        self.tf_idf = tf_idf
        self.bag_of_words = bag_of_words
        self.references = references

