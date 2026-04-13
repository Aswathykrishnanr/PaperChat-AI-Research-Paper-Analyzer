import chromadb
from embeddings import get_embeddings
# Create ChromaDB client
client = chromadb.Client()
def create_collection(collection_name):
    collection= client.get_or_create_collection(name=collection_name)
    return collection

def store_chunks(collection, chunks_with_source):
    texts = [c["text"] for c in chunks_with_source]
    sources = [c["source"] for c in chunks_with_source]
    embeddings = get_embeddings(texts)
    
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"source": s} for s in sources],
                ids=[f"chunk_{i}" for i in range(len(texts))])

def search_chunks(collection, query, top_k=5):
    query_embedding = get_embeddings([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )
    # Return both chunks and sources
    chunks = results['documents'][0]
    sources = [m["source"] for m in results['metadatas'][0]]
    return chunks,sources

#  Testing Code
# if __name__ == "__main__":
    
#     # Sample chunks
#     sample_chunks = [
#         "Brain tumor detection using deep learning methods",
#         "MRI scans are used for tumor classification",
#         "CNN models achieve high accuracy in medical imaging",
#         "U-Net architecture is used for image segmentation",
#         "Random forest is a machine learning algorithm"
#     ]
    
#     # Create collection
#     print("Creating ChromaDB collection...")
#     collection = create_collection("test_papers")
    
#     # Store chunks
#     print("Storing chunks...")
#     store_chunks(collection, sample_chunks)
    
#     # Search
#     query = "What is used for tumor detection?"
#     print(f"\nSearching for: '{query}'")
#     results = search_chunks(collection, query)
    
#     print("\nTop relevant chunks found:")
#     for i, chunk in enumerate(results):
#         print(f"\n{i+1}. {chunk}")