import pickle

from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

from nlp_final_project.models.qapipelinerextr import in_memory_retriever

if __name__ == "__main__":
    document_store, retriever = in_memory_retriever(load=False)
    document_store.write_documents([Document(content="CONTENT")])

    #print(document_store.count_documents())
    #dictionary = document_store.to_dict()
    #print(dictionary)
    #document_store2 = InMemoryDocumentStore.from_dict(dictionary)
    #print(document_store2.count_documents())
    #print(document_store2.to_dict())


    print(document_store.count_documents())

    with open("emb.pkl", "wb") as pickle_file:
        pickle.dump(document_store, pickle_file)

    with open("emb.pkl", "rb") as f:
        document_store = pickle.load(f)
        # document_store = InMemoryDocumentStore.from_dict(obj)

    print(document_store.count_documents())