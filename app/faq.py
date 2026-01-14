import os
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

faqs_path = Path(__file__).parent / "resources/faq_data.csv"

# Embedding function
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Persistent ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection_name_faq = "faqs"

# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------

def ingest_faq_data(path: Path):
    """Ingest FAQ data into ChromaDB (runs only once)"""
    existing_collections = [c.name for c in chroma_client.list_collections()]
    if collection_name_faq in existing_collections:
        return

    print("Ingesting FAQ data into ChromaDB...")

    collection = chroma_client.create_collection(
        name=collection_name_faq,
        embedding_function=ef
    )

    df = pd.read_csv(path)
    documents = df["question"].tolist()
    metadatas = [{"answer": ans} for ans in df["answer"].tolist()]
    ids = [f"faq_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )


    print(f"FAQ data ingested into collection: {collection_name_faq}")

# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def get_relevant_qa(query: str):
    """Retrieve relevant Q&A from ChromaDB"""
    collection = chroma_client.get_collection(
        name=collection_name_faq,
        embedding_function=ef
    )

    return collection.query(
        query_texts=[query],
        n_results=3
    )

# ---------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------

def generate_answer(query: str, context: str) -> str:
    """Generate answer using OpenAI based on retrieved context"""

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
- Keep answers brief (3–4 sentences)
"""

    messages = [
        {"role": "system", "content": "You are a helpful e-commerce customer service assistant."},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=messages
        )
        return response.output_text

    except Exception:
        return (
            "I apologize, but I encountered an error while answering your question. "
            "Please try again."
        )

# ---------------------------------------------------------------------
# Public Chain
# ---------------------------------------------------------------------

def faq_chain(query: str) -> str:
    """Main FAQ chain: retrieve context and generate answer"""
    try:
        result = get_relevant_qa(query)

        if result and result.get("metadatas") and result["metadatas"][0]:
            context = " ".join(
                meta.get("answer", "") for meta in result["metadatas"][0]
            )
        else:
            return (
                "I don't have specific information about that. "
                "Please contact our support team or try rephrasing your question."
            )

        return generate_answer(query, context)

    except Exception:
        return (
            "I'm having trouble accessing our FAQ information right now. "
            "Please try again later."
        )

# ---------------------------------------------------------------------
# Local Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ingest_faq_data(faqs_path)

    test_queries = [
        "What's your policy on defective products?",
        "Do you take cash as a payment option?",
        "Is online payment available?",
        "What are your shipping charges?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print("Query:", q)
        print("Answer:", faq_chain(q))
