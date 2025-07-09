# import os
# import time
# import arxiv
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import pinecone
# import openai
# from tqdm import tqdm
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# # Load environment variables
# load_dotenv()

# # Set API keys
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')

# # Initialize OpenAI
# openai.api_key = OPENAI_API_KEY

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Constants
# EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'  # or 'intfloat/e5-small'
# INDEX_NAME = 'rag-latency-demo'
# CATEGORY = 'cs.CL'  # or 'cs.LG'
# NUM_PAPERS = 15
# CHUNK_SIZE = 3  # sentences per chunk

# # 1. Data Ingestion: Download papers from arXiv
# # TODO: Download 10-20 papers (title, abstract, authors)

# def download_papers(category, max_results=15):
#     search = arxiv.Search(query=f'cat:{category}', max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
#     papers = []
#     for result in search.results():
#         papers.append({
#             'title': result.title,
#             'abstract': result.summary,
#             'authors': ', '.join([a.name for a in result.authors]),
#             'published': result.published.date().isoformat()
#         })
#     return pd.DataFrame(papers)

# # 2. Chunk abstracts into passages
# # TODO: Split abstracts into 2-3 sentence chunks

# def chunk_text(text, chunk_size=3):
#     import re
#     sentences = re.split(r'(?<=[.!?]) +', text)
#     return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# # 3. Embedding & Pinecone upsert
# # TODO: Embed and store in Pinecone

# def embed_and_upsert(df, model, index):
#     for idx, row in tqdm(df.iterrows(), total=len(df)):
#         chunks = chunk_text(row['abstract'], CHUNK_SIZE)
#         for i, chunk in enumerate(chunks):
#             passage_id = f"{idx}-{i}"
#             embedding = model.encode(chunk)
#             meta = {'title': row['title'], 'published': row['published'], 'chunk': chunk}
#             index.upsert([(passage_id, embedding.tolist(), meta)])

# # 4. Query, Retrieval, Generation, and Latency Logging
# # TODO: Implement query flow and log latency

# def query_pipeline(user_query, model, index):
#     latency = {}
#     start_total = time.time()

#     # Embedding
#     start = time.time()
#     query_emb = model.encode(user_query)
#     latency['embedding'] = time.time() - start

#     # Search
#     start = time.time()
#     res = index.query(vector=query_emb.tolist(), top_k=3, include_metadata=True)
#     latency['search'] = time.time() - start

#     # Prepare context for LLM
#     context = '\n'.join([m['metadata']['chunk'] for m in res['matches']])

#     # Generation
#     start = time.time()
#     client = openai.OpenAI(api_key=OPENAI_API_KEY)
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant for academic Q&A."},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
#         ]
#     )
#     latency['generation'] = time.time() - start

#     latency['total'] = time.time() - start_total
#     answer = response.choices[0].message.content
#     return answer, latency

# # 5. Main execution
# if __name__ == "__main__":
#     # Download papers
#     print("Downloading papers from arXiv...")
#     df = download_papers(CATEGORY, NUM_PAPERS)

#     # Load embedding model
#     print("Loading embedding model...")
#     model = SentenceTransformer(EMBEDDING_MODEL_NAME)

#     # Check if index exists, create if not
#     if INDEX_NAME not in pc.list_indexes().names():
#         pc.create_index(
#             name=INDEX_NAME,
#             dimension=model.get_sentence_embedding_dimension(),
#             metric='cosine',  # or 'euclidean' if you prefer
#             spec=ServerlessSpec(
#                 cloud='aws',         # or 'gcp'
#                 region='us-west-2'   # or your Pinecone region
#             )
#         )

#     index = pc.Index(INDEX_NAME)

#     # Upsert embeddings
#     print("Embedding and upserting to Pinecone...")
#     embed_and_upsert(df, model, index)

#     # Query loop
#     while True:
#         user_query = input("\nEnter your question (or 'exit'): ")
#         if user_query.lower() == 'exit':
#             break
#         answer, latency = query_pipeline(user_query, model, index)
#         print(f"\nAnswer: {answer}\n")
#         print(f"Latency (s): {latency}")
#         # Log latency
#         log_df = pd.DataFrame([{**{'query': user_query, 'answer': answer}, **latency}])
#         if not os.path.exists('latency_log.csv'):
#             log_df.to_csv('latency_log.csv', index=False)
#         else:
#             log_df.to_csv('latency_log.csv', mode='a', header=False, index=False)

#     print("Done.") 
import os
import time
import arxiv
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Set API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Initialize OpenAI client (NEW WAY)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Constants
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_NAME = 'rag-latency-demo'
CATEGORY = 'cs.CL'
NUM_PAPERS = 15
CHUNK_SIZE = 3

def download_papers(category, max_results=15):
    search = arxiv.Search(query=f'cat:{category}', max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
    papers = []
    for result in search.results():
        papers.append({
            'title': result.title,
            'abstract': result.summary,
            'authors': ', '.join([a.name for a in result.authors]),
            'published': result.published.date().isoformat()
        })
    return pd.DataFrame(papers)

def chunk_text(text, chunk_size=3):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

def embed_and_upsert(df, model, index):
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        chunks = chunk_text(row['abstract'], CHUNK_SIZE)
        for i, chunk in enumerate(chunks):
            if chunk.strip():  # Skip empty chunks
                passage_id = f"{idx}-{i}"
                embedding = model.encode(chunk)
                meta = {'title': row['title'], 'published': row['published'], 'chunk': chunk}
                index.upsert([(passage_id, embedding.tolist(), meta)])

def query_pipeline(user_query, model, index):
    latency = {}
    start_total = time.time()

    # Embedding
    start = time.time()
    query_emb = model.encode(user_query)
    latency['embedding'] = time.time() - start

    # Search
    start = time.time()
    res = index.query(vector=query_emb.tolist(), top_k=3, include_metadata=True)
    latency['search'] = time.time() - start

    # Prepare context for LLM
    context = '\n'.join([m['metadata']['chunk'] for m in res['matches']])

    # Generation
    start = time.time()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for academic Q&A."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]
    )
    latency['generation'] = time.time() - start

    latency['total'] = time.time() - start_total
    answer = response.choices[0].message.content
    return answer, latency

if __name__ == "__main__":
    # Download papers
    print("Downloading papers from arXiv...")
    df = download_papers(CATEGORY, NUM_PAPERS)

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Check if index exists, create if not
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=model.get_sentence_embedding_dimension(),
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Match app.py region
            )
        )

    index = pc.Index(INDEX_NAME)

    # Upsert embeddings
    print("Embedding and upserting to Pinecone...")
    embed_and_upsert(df, model, index)

    # Query loop
    while True:
        user_query = input("\nEnter your question (or 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        try:
            answer, latency = query_pipeline(user_query, model, index)
            print(f"\nAnswer: {answer}\n")
            print(f"Latency (s): {latency}")
            
            # Log latency
            log_df = pd.DataFrame([{**{'query': user_query, 'answer': answer}, **latency}])
            if not os.path.exists('latency_log.csv'):
                log_df.to_csv('latency_log.csv', index=False)
            else:
                log_df.to_csv('latency_log.csv', mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error: {e}")

    print("Done.")
