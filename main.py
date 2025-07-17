import os
import time
import arxiv
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import json

# Load environment variables
load_dotenv()

# Set API keys
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Constants
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_NAME = 'rag-latency-demo'
CATEGORY = 'cs.CL'
NUM_PAPERS = 15
CHUNK_SIZE = 3

def call_gemini_api(prompt):
    """Call Gemini API with the given prompt"""
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': GEMINI_API_KEY
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000,
            "topP": 0.8,
            "topK": 40
        }
    }
    
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "No response generated"
    else:
        raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

def download_papers(category, max_results=15):
    """Download papers from arXiv"""
    search = arxiv.Search(
        query=f'cat:{category}', 
        max_results=max_results, 
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = []
    
    print(f"Fetching {max_results} papers from category: {category}")
    
    for result in search.results():
        papers.append({
            'title': result.title,
            'abstract': result.summary,
            'authors': ', '.join([a.name for a in result.authors]),
            'published': result.published.date().isoformat()
        })
    
    return pd.DataFrame(papers)

def chunk_text(text, chunk_size=3):
    """Split text into chunks of sentences"""
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    
    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i+chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def embed_and_upsert(df, model, index):
    """Embed text chunks and upsert to Pinecone"""
    total_chunks = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
        chunks = chunk_text(row['abstract'], CHUNK_SIZE)
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                passage_id = f"{idx}-{i}"
                embedding = model.encode(chunk)
                meta = {
                    'title': row['title'], 
                    'published': row['published'], 
                    'chunk': chunk,
                    'authors': row['authors']
                }
                
                # Upsert to Pinecone
                index.upsert([(passage_id, embedding.tolist(), meta)])
                total_chunks += 1
    
    print(f"Successfully upserted {total_chunks} chunks to Pinecone")

def query_pipeline(user_query, model, index):
    """Complete RAG pipeline: embed query, search, generate answer"""
    latency = {}
    start_total = time.time()

    # Step 1: Embedding
    start = time.time()
    query_emb = model.encode(user_query)
    latency['embedding'] = time.time() - start

    # Step 2: Vector Search
    start = time.time()
    res = index.query(
        vector=query_emb.tolist(), 
        top_k=3, 
        include_metadata=True
    )
    latency['search'] = time.time() - start

    # Check if we have results
    if not res['matches']:
        latency['generation'] = 0
        latency['total'] = time.time() - start_total
        return "No relevant documents found. Please run the script first to populate the database.", latency

    # Step 3: Prepare context for LLM
    context_parts = []
    for match in res['matches']:
        context_parts.append(f"Paper: {match['metadata']['title']}\nContent: {match['metadata']['chunk']}")
    
    context = '\n\n'.join(context_parts)

    # Step 4: Generation with Gemini
    start = time.time()
    
    # Create comprehensive prompt for Gemini
    prompt = f"""You are a helpful assistant for academic Q&A. Answer the question based on the provided research context.

Context from recent research papers:
{context}

Question: {user_query}

Please provide a comprehensive and accurate answer based on the research context provided above. If the context doesn't contain enough information to fully answer the question, please say so."""

    try:
        answer = call_gemini_api(prompt)
        latency['generation'] = time.time() - start
        latency['total'] = time.time() - start_total
        return answer, latency
    except Exception as e:
        latency['generation'] = time.time() - start
        latency['total'] = time.time() - start_total
        return f"Error generating response: {str(e)}", latency

def check_index_status(index):
    """Check if index has data"""
    try:
        stats = index.describe_index_stats()
        return stats['total_vector_count'] > 0
    except:
        return False

if __name__ == "__main__":
    print("🚀 Mini RAG Pipeline - Gemini Edition")
    print("=" * 50)
    
    # Check if API keys are available
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY not found in environment variables")
        print("Please add your Gemini API key to your .env file")
        exit(1)
    
    if not PINECONE_API_KEY:
        print("❌ Error: PINECONE_API_KEY not found in environment variables")
        print("Please add your Pinecone API key to your .env file")
        exit(1)
    
    # Download papers
    print("📚 Downloading papers from arXiv...")
    try:
        df = download_papers(CATEGORY, NUM_PAPERS)
        print(f"✅ Downloaded {len(df)} papers")
    except Exception as e:
        print(f"❌ Error downloading papers: {e}")
        exit(1)

    # Load embedding model
    print("🧠 Loading embedding model...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Loaded {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # Check if index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🔧 Creating Pinecone index: {INDEX_NAME}")
        try:
            pc.create_index(
                name=INDEX_NAME,
                dimension=model.get_sentence_embedding_dimension(),
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            print("⏳ Waiting for index to be ready...")
            time.sleep(30)  # Wait for index to be ready
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            exit(1)

    # Connect to index
    index = pc.Index(INDEX_NAME)

    # Check if index has data, populate if empty
    if not check_index_status(index):
        print("📝 Populating index with embeddings...")
        try:
            embed_and_upsert(df, model, index)
            print("✅ Database population completed!")
        except Exception as e:
            print(f"❌ Error populating database: {e}")
            exit(1)
    else:
        print("✅ Index already contains data")

    # Query loop
    print("\n" + "=" * 50)
    print("🎯 RAG Pipeline Ready! Ask your questions:")
    print("💡 Example: 'What are recent advances in transformer architectures?'")
    print("🚪 Type 'exit' to quit")
    print("=" * 50)
    
    while True:
        user_query = input("\n🔍 Enter your question: ").strip()
        
        if user_query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if not user_query:
            print("❌ Please enter a valid question")
            continue
        
        try:
            print("⚙️ Processing your query...")
            answer, latency = query_pipeline(user_query, model, index)
            
            print(f"\n📝 Answer:\n{answer}\n")
            print("⚡ Performance Metrics:")
            print(f"  📊 Embedding: {latency['embedding']:.3f}s")
            print(f"  🔍 Search: {latency['search']:.3f}s") 
            print(f"  🤖 Generation: {latency['generation']:.3f}s")
            print(f"  ⏱️ Total: {latency['total']:.3f}s")
            
            # Log latency
            log_row = {
                'timestamp': time.time(),
                'query': user_query, 
                'answer': answer, 
                'embedding': latency['embedding'],
                'search': latency['search'],
                'generation': latency['generation'],
                'total': latency['total']
            }
            log_df = pd.DataFrame([log_row])
            
            if not os.path.exists('latency_log.csv'):
                log_df.to_csv('latency_log.csv', index=False)
                print("📊 Created latency log file")
            else:
                log_df.to_csv('latency_log.csv', mode='a', header=False, index=False)
                print("📊 Logged performance metrics")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Please try again or type 'exit' to quit")

    print("🎉 Done! Thanks for using the RAG Pipeline!")
