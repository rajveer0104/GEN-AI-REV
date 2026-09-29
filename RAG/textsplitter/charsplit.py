from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,TokenTextSplitter,RecursiveCharacterTextSplitter
data=TextLoader('RAG/document loaders/deep.txt')
#splitter=CharacterTextSplitter(
#    separator="",
#    chunk_size=10,
#    chunk_overlap=1
#)
'''sp=TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)'''
spl=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)
docs=data.load()
#chunks=splitter.split_documents(docs)
#chunks=sp.split_documents(docs)
chunks=spl.split_documents(docs)

print(len(chunks))
