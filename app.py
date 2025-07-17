import os
import time
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import plotly.graph_objects as go
from pinecone import Pinecone, ServerlessSpec
import streamlit_lottie
from datetime import datetime
import json

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Gemini API endpoint
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
INDEX_NAME = 'rag-latency-demo'

# Load embedding model
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

model = load_model_and_index()

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=model.get_sentence_embedding_dimension(),
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(INDEX_NAME)

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

def query_pipeline(user_query, model):
    latency = {}
    start_total = time.time()

    # Embedding
    start = time.time()
    query_emb = model.encode(user_query)
    latency['Embedding'] = time.time() - start

    # Search
    start = time.time()
    res = index.query(vector=query_emb.tolist(), top_k=3, include_metadata=True)
    latency['Search'] = time.time() - start

    # Check if we have results
    if not res['matches']:
        return "No relevant documents found. Please run main.py first to populate the index.", [], latency

    # Prepare context for LLM
    context = '\n'.join([m['metadata']['chunk'] for m in res['matches']])
    passages = [
        {
            'title': m['metadata']['title'],
            'chunk': m['metadata']['chunk'],
            'score': m['score']
        } for m in res['matches']
    ]

    # Generation with Gemini
    start = time.time()
    
    # Create prompt for Gemini
    prompt = f"""You are a helpful assistant for academic Q&A. Answer the question based on the provided context.

Context:
{context}

Question: {user_query}

Please provide a comprehensive and accurate answer based on the context provided above."""

    try:
        answer = call_gemini_api(prompt)
        latency['Generation'] = time.time() - start
        latency['Total'] = time.time() - start_total
        return answer, passages, latency
    except Exception as e:
        latency['Generation'] = time.time() - start
        latency['Total'] = time.time() - start_total
        return f"Error generating response: {str(e)}", passages, latency

# Initialize session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "query_count" not in st.session_state:
    st.session_state.query_count = 0

# Page configuration
st.set_page_config(
    page_title="🚀 Mini RAG Pipeline Pro - Gemini",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_loading = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_usmfx6bp.json")

# Theme toggle in sidebar
with st.sidebar:
    st.markdown("### 🎨 Theme Settings")
    theme_toggle = st.toggle("🌗 Dark Mode", value=st.session_state.theme == "dark")
    if theme_toggle != (st.session_state.theme == "dark"):
        st.session_state.theme = "dark" if theme_toggle else "light"
        st.rerun()

# Apply theme-based styling
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #f8f8f2;
    }
    .stSidebar {
        background: linear-gradient(180deg, #1a1a2e 0%, #0c0c0c 100%);
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: #f8f8f2;
    }
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #f8f8f2;
        border: 1px solid #4285f4;
    }
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 100%);
        border: 1px solid #4285f4;
    }
    .passage-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 100%);
        border: 1px solid #4285f4;
    }
    .answer-container {
        background: linear-gradient(135deg, #2d2d2d 0%, #1a1a2e 100%);
        border: 1px solid #4285f4;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 50%, #dee2e6 100%);
        color: #2c3e50;
    }
    .stSidebar {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
        color: #2c3e50;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #2c3e50;
        border: 1px solid #4285f4;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #4285f4;
    }
    .passage-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #4285f4;
    }
    .answer-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #4285f4;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced CSS styling
st.markdown("""
<style>
/* Global Styles */
:root {
    --primary-color: #4285f4;
    --secondary-color: #34a853;
    --accent-color: #ea4335;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --error-color: #dc3545;
    --border-radius: 15px;
    --box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    --transition: all 0.3s ease;
}

/* Button Styling */
.stButton > button {
    background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 2rem;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: var(--box-shadow);
    transition: var(--transition);
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, var(--accent-color) 0%, var(--primary-color) 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(66,133,244,0.3);
}

/* Form Styling */
.stTextInput > div > div > input {
    border-radius: var(--border-radius);
    border: 2px solid var(--primary-color);
    padding: 0.75rem 1rem;
    font-size: 1rem;
    transition: var(--transition);
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(66,133,244,0.1);
}

/* Header Styling */
.main-header {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: var(--box-shadow);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

/* Metric Cards */
.metric-card {
    padding: 1.5rem;
    border-radius: var(--border-radius);
    text-align: center;
    box-shadow: var(--box-shadow);
    transition: var(--transition);
    margin-bottom: 1rem;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(66,133,244,0.2);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.9rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Passage Cards */
.passage-card {
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 1rem;
    border-left: 4px solid var(--accent-color);
    box-shadow: var(--box-shadow);
    transition: var(--transition);
}

.passage-card:hover {
    box-shadow: 0 8px 30px rgba(66,133,244,0.25);
    transform: translateY(-3px);
}

/* Answer Container */
.answer-container {
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    border-left: 5px solid var(--success-color);
    box-shadow: var(--box-shadow);
    transition: var(--transition);
}

/* Loading Animation Container */
.loading-container {
    text-align: center;
    padding: 2rem;
    border-radius: var(--border-radius);
    margin: 2rem 0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid #e0e0e0;
    color: #666;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.markdown("### 📚 RAG Pipeline Overview")
    
    # System info
    st.markdown("""
    **🛠️ Technical Stack:**
    - 🧠 **LLM:** Google Gemini 2.0 Flash
    - 🔍 **Embeddings:** all-MiniLM-L6-v2
    - 🗄️ **Vector DB:** Pinecone
    - 📊 **Retrieval:** Top-3 Similarity Search
    """)
    
    # Statistics
    if os.path.exists('latency_log.csv'):
        try:
            log_df = pd.read_csv('latency_log.csv')
            st.markdown("### 📈 Usage Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Queries", len(log_df))
            with col2:
                st.metric("Avg Response", f"{log_df['Total'].mean():.2f}s")
        except:
            pass
    
    st.markdown("---")
    
    # Instructions
    st.markdown("""
    ### 🚀 Quick Start Guide
    1. **Setup**: Run `python main.py` to populate database
    2. **Query**: Enter your research question
    3. **Explore**: View retrieved passages and metrics
    4. **Analyze**: Check latency breakdown
    """)
    
    st.info("💡 **Tip**: Ask about recent NLP/ML research for best results!")

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: 2px;">
        🚀 Mini RAG Pipeline Pro
    </h1>
    <p style="font-size: 1.3rem; opacity: 0.95; margin-bottom: 0;">
        Advanced Retrieval-Augmented Generation powered by Google Gemini & Pinecone
    </p>
</div>
""", unsafe_allow_html=True)

# Main content layout
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 💬 Ask Your Question")
    
    with st.form(key="query_form"):
        user_query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., What are recent advances in transformer architectures?",
            help="Ask about recent developments in NLP, ML, or AI research"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit = st.form_submit_button("🔍 Get Answer")
        with col_btn2:
            clear = st.form_submit_button("🗑️ Clear")
    
    if clear:
        st.session_state.query_count = 0
        st.rerun()

with col2:
    # Results area
    results_container = st.container()
    
    if submit and user_query.strip():
        with results_container:
            # Create loading animation container
            loading_container = st.empty()
            
            # Show loading animation
            with loading_container.container():
                st.markdown('<div class="loading-container">', unsafe_allow_html=True)
                if lottie_loading:
                    streamlit_lottie.st_lottie(lottie_loading, height=120, key=f"loading_{st.session_state.query_count}")
                else:
                    st.markdown("### 🔄 Processing your query...")
                    st.markdown("Please wait while we search and generate your answer...")
                st.markdown('</div>', unsafe_allow_html=True)
            
            try:
                # Process query
                answer, passages, latency = query_pipeline(user_query, model)
                st.session_state.query_count += 1
                
                # Clear loading animation
                loading_container.empty()
                
                # Display results
                st.markdown('<div class="answer-container">', unsafe_allow_html=True)
                st.markdown("### 🎯 Answer")
                st.markdown(f"**{answer}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Performance metrics
                st.markdown("### ⚡ Performance Metrics")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{latency['Embedding']:.3f}s</div>
                        <div class="metric-label">Embedding</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_m2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{latency['Search']:.3f}s</div>
                        <div class="metric-label">Search</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_m3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{latency['Generation']:.3f}s</div>
                        <div class="metric-label">Generation</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_m4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{latency['Total']:.3f}s</div>
                        <div class="metric-label">Total</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced chart
                st.markdown("### 📊 Latency Breakdown")
                
                latency_df = pd.DataFrame(list(latency.items()), columns=["Stage", "Seconds"])
                
                colors = ['#4285f4', '#34a853', '#ea4335', '#fbbc04']
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=latency_df["Stage"],
                    y=latency_df["Seconds"],
                    marker_color=colors,
                    text=[f"{val:.3f}s" for val in latency_df["Seconds"]],
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Time: %{y:.3f}s<extra></extra>'
                ))
                
                fig.update_layout(
                    title="Response Time Analysis",
                    xaxis_title="Processing Stage",
                    yaxis_title="Time (seconds)",
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Retrieved passages
                if passages:
                    st.markdown("### 📚 Retrieved Knowledge")
                    
                    for i, p in enumerate(passages, 1):
                        st.markdown(f"""
                        <div class="passage-card">
                            <h4>📄 Passage {i} (Similarity: {p['score']:.3f})</h4>
                            <p><strong>Paper:</strong> {p['title']}</p>
                            <p style="margin-top: 1rem;">{p['chunk']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Save log
                log_row = {
                    'timestamp': datetime.now().isoformat(),
                    'query': user_query,
                    'answer': answer,
                    **latency
                }
                log_df = pd.DataFrame([log_row])
                
                if not os.path.exists('latency_log.csv'):
                    log_df.to_csv('latency_log.csv', index=False)
                else:
                    log_df.to_csv('latency_log.csv', mode='a', header=False, index=False)
                
                st.success("💾 Query and performance data saved to latency_log.csv")
                
            except Exception as e:
                loading_container.empty()
                st.error(f"❌ Error: {str(e)}")
                st.info("Please make sure you have run `python main.py` first to populate the database.")
    
    elif not submit:
        with results_container:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; opacity: 0.7;">
                <h3>👋 Welcome to RAG Pipeline Pro!</h3>
                <p>Enter your question on the left to get started</p>
                <p>🔍 Ask about recent research in NLP, ML, or AI</p>
                <p>⚡ Get instant answers with performance insights</p>
            </div>
            """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("""
<div class="footer">
    <p>🚀 <strong>Mini RAG Pipeline Pro</strong> | Powered by Google Gemini & Pinecone</p>
    <p>Built with ❤️ by <strong>Vijayshree Vaibhav</strong> | 7x National Hackathon Winner</p>
    <p><a href="https://vijayshreepathak.netlify.app/" style="color:var(--primary-color); text-decoration:none;">🌐 Portfolio</a> | 
    <a href="https://www.linkedin.com/in/vijayshreevaibhav" style="color:var(--primary-color); text-decoration:none;">💼 LinkedIn</a></p>
</div>
""", unsafe_allow_html=True)
