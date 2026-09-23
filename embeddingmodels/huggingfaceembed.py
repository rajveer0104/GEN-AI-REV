from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

sentence = "I am learning Generative AI and LangChain."

vector = embeddings.embed_query(sentence)

print(vector)
print("Embedding dimensions:", len(vector))