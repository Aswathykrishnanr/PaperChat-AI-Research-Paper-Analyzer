
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_answer(question, relevant_chunks):
    # Combine all chunks into one context
    context = "\n\n".join(relevant_chunks)   
    prompt = f"""You are a helpful research assistant.
Use the following context from research papers to answer the question.
Only answer based on the context provided.
If answer is not in context say "I could not find this in the papers."

Context:{context}
Question: {question}
Answer:"""
    
    # Send to Groq API
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
               messages=[{"role": "user", "content": prompt}]) 
    # Extract and return answer
    return response.choices[0].message.content

# Testing
# if __name__ == "__main__":

#     sample_chunks = [
#         "Brain tumor detection using deep learning achieves 95% accuracy",
#         "CNN and U-Net models are used for MRI based tumor segmentation",
#         "The dataset used contains 3064 MRI images of brain tumors"]
#     question = "What accuracy was achieved in brain tumor detection?"
#     print(f"Question: {question}")
#     print("\nGetting answer from Groq...")
#     answer = get_answer(question, sample_chunks)
#     print(f"\nAnswer: {answer}")