import os
import chromadb
from chromadb.utils import embedding_functions

from phi.model.openai import OpenAIChat

# from groq import Groq
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

#Model initialization
load_dotenv()
model = OpenAIChat(id="gpt-5-mini")


faqs_path = Path(__file__).parent / "resources/faq_data.csv"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

chroma_client = chromadb.Client()
# groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))

collection_name_faq = 'faqs'


def ingest_faq_data(path):
    """Ingest FAQ data into ChromaDB"""
    if collection_name_faq not in [c.name for c in chroma_client.list_collections()]:
        print("Ingesting FAQ data into Chromadb...")
        collection = chroma_client.create_collection(
            name=collection_name_faq,
            embedding_function=ef
        )
        df = pd.read_csv(path)
        docs = df['question'].to_list()
        metadata = [{'answer': ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(docs))]
        collection.add(
            documents=docs,
            metadatas=metadata,
            ids=ids
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name_faq}")
    else:
        print(f"Collection: {collection_name_faq} already exists")


def get_relevant_qa(query):
    """Retrieve relevant Q&A from ChromaDB"""
    collection = chroma_client.get_collection(
        name=collection_name_faq,
        embedding_function=ef
    )
    result = collection.query(
        query_texts=[query],
        n_results=3  # Increased from 2 to 3 for better context
    )
    return result


def generate_answer(query, context):
    """Generate answer using OpenAIChat based on context"""

    prompt = f"""You are a helpful e-commerce customer service assistant.
Answer the question based ONLY on the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer directly and concisely based on the context
- If the exact answer isn't in the context but related information is, provide that
- If no relevant information is found, say:
  "I don't have that specific information, but you can contact our support team for help."
- Be friendly and professional
- Keep answers brief (3-4 sentences)
"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful e-commerce customer service assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = model.run(messages)
        return response.content

    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."



def faq_chain(query):
    """Main FAQ chain: retrieve context and generate answer"""
    try:
        result = get_relevant_qa(query)
        
        # Extract context from metadata
        if result and result['metadatas'] and len(result['metadatas'][0]) > 0:
            context = " ".join([r.get('answer', '') for r in result['metadatas'][0]])
            print(f"[DEBUG] FAQ Context found: {context[:100]}...")
        else:
            print("[DEBUG] No FAQ context found")
            return "I don't have specific information about that. Please contact our support team or try rephrasing your question."
        
        answer = generate_answer(query, context)
        return answer
    
    except Exception as e:
        print(f"[ERROR] FAQ chain error: {str(e)}")
        return "I apologize, but I'm having trouble accessing FAQ information right now. Please try again."


if __name__ == '__main__':
    ingest_faq_data(faqs_path)
    
    test_queries = [
        "What's your policy on defective products?",
        "Do you take cash as a payment option?",
        "Is online payment available?",
        "What are your shipping charges?",
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        answer = faq_chain(query)
        print(f"Answer: {answer}")