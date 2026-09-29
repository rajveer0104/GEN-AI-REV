from langchain_community.document_loaders import TextLoader

data = TextLoader('RAG/document loaders/deep.txt')

docs = data.load()

print(docs)