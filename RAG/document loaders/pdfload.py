from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader('RAG/document loaders/13030823080_Rajveer_PEC-AIML601B_CA2.pdf')
docs=data.load()
print(len(docs))