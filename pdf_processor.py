import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    full_text = ""            
    for page in doc:
        # Extract text from each page
        full_text+= page.get_text()
    return full_text
def split_text_into_chunks(text, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = splitter.split_text(text)
    # Add filename to each chunk
    chunks_with_source = [{"text": chunk, "source": filename}
        for chunk in chunks]
    return chunks_with_source


# #Testing our functions
# if __name__ == "__main__":
    
#     #Extract text from PDF
#     print("Reading PDF...")
#     text = extract_text_from_pdf("test.pdf")
    
#     #Print first 500 characters
#     print("First 500 characters of text:")
#     print(text[:500])
    
#     #Split into chunks
#     print("\nSplitting into chunks...")
#     chunks = split_text_into_chunks(text)
    
#     #Print results
#     print(f"Total chunks created: {len(chunks)}")
#     print("\nFirst chunk:")
#     print(chunks[0])
#     print("\nSecond chunk:")
#     print(chunks[1])