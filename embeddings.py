# Import sentence transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
def get_embeddings(chunks):
    # Convert list of chunks to embeddings
    embeddings = model.encode(chunks)
    return embeddings

# Testing
# if __name__ == "__main__":
    
#     # Sample chunks to test
#     sample_chunks = [
#         "Brain tumor detection using deep learning",
#         "MRI scans are used for tumor classification",
#         "CNN models achieve high accuracy"
#     ]
    
#     print("Converting chunks to embeddings...")
#     embeddings = get_embeddings(sample_chunks)
    
#     print(f"Number of embeddings: {len(embeddings)}")
#     print(f"Each embedding size: {len(embeddings[0])}")
#     print(f"First embedding (first 5 numbers):")
#     print(embeddings[0][:5])