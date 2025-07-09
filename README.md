# Mini RAG Pipeline Pro 🚀

## 🌐 Live Demo

**Try it now:**
👉 [https://miniragpipeline.streamlit.app/](https://miniragpipeline.streamlit.app/)

*Experience the full RAG pipeline with real-time latency analytics directly in your browser — no setup required!*

---

## Overview

This project implements a modern **Retrieval-Augmented Generation (RAG)** pipeline for academic Q\&A, featuring a beautiful Streamlit web UI with dark/light mode, real-time latency analytics, and interactive querying.

It ingests academic papers from arXiv, embeds and stores them in Pinecone, and uses an LLM to answer questions based on retrieved content.
Each pipeline stage's latency is logged, visualized, and saved for analysis.

**🚀 [Access Live Demo](https://miniragpipeline.streamlit.app/)** — Try the application instantly without any local setup!

---

## 🧪 Model & Design Choices

* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
* **LLM:** OpenAI GPT-4 (via `openai` Python SDK)
* **Vector DB:** Pinecone (cloud = `aws`, region = `us-east-1`)
* **Chunking:** Abstracts split into 3-sentence passages
* **UI:** Streamlit with advanced CSS, theme toggle, Lottie animations, and Plotly for latency visualization

---

## ✨ Key Features

* 🌗 **Dark/Light Theme Toggle** — Instantly switch between beautiful dark and light modes from the sidebar.
* 📊 **Real-Time Latency Analytics** — Detailed latency metrics for each pipeline stage (Embedding, Search, Generation, Total) shown as metrics and a Plotly bar chart.
* 📝 **Query Logging** — Every query, answer, and latency breakdown is saved to `latency_log.csv` for later analysis.
* 🧑‍💻 **Modern UI/UX** — Enhanced sidebar with tech stack, usage stats, and a quick start guide. Lottie animations and custom CSS for a premium experience.
* 🧠 **Top-3 Passage Retrieval** — Displays the most relevant passages with similarity scores and paper titles.
* 🛡️ **Robust Error Handling** — Friendly error messages and guidance if the database is not yet populated.
* 🌐 **Live Deployment** — Fully deployed on Streamlit Cloud for instant access.
* 🏆 **Attribution & Links** — Footer credits and links to the author’s portfolio and LinkedIn.

---

## 🚀 Usage Options

### ✅ Option 1: Try the Live Demo (Recommended)

👉 **[Access the live app](https://miniragpipeline.streamlit.app/)**
No installation or setup required — just open the link and start asking questions about recent NLP/ML research!

---

### 👩‍💻 Option 2: Local Development Setup

1️⃣ **Clone the repository:**

```bash
git clone https://github.com/vijayshreepathak/PineRAG---Fast-RAG-with-Latency-Analytics.git
cd PineRAG---Fast-RAG-with-Latency-Analytics
```

2️⃣ **Install dependencies:**

```bash
pip install -r requirements.txt
```

3️⃣ **Set your API keys in a `.env` file:**

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

4️⃣ **Populate the Pinecone database:**

```bash
python main.py
```

5️⃣ **Launch the Streamlit UI:**

```bash
streamlit run app.py
```

---

## 🖥️ How to Use the Web App

### On the Live Demo

1. Visit 👉 [https://miniragpipeline.streamlit.app/](https://miniragpipeline.streamlit.app/)
2. Use the **Theme Toggle** in the sidebar to switch between dark & light modes.
3. Enter your research question and click "🔍 Get Answer".
4. View the generated answer, latency metrics, latency bar chart, and top-3 retrieved passages.

---

### Additional Features

* 🗑️ **Clear Query:** Reset the input and results.
* 📈 **Usage Stats:** Sidebar shows total queries & average response time.
* 📝 **Quick Start Guide:** Sidebar provides step-by-step instructions.
* ⏱️ **Real-Time Performance:** Monitor embedding, search & generation latencies.

---

## 🔎 Sample Query & Result

Try this example on the [live demo](https://miniragpipeline.streamlit.app/):

* **Query:** What are recent advances in language modeling?

* **Generated Answer:**

> Recent advances in language modeling involve the development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). These models produce text that is coherent, contextually relevant, and strikingly similar to human writing. They are versatile and adaptable to various styles and genres, producing content that is grammatically correct and semantically meaningful.

*(…truncated for brevity…)*

* **Top 3 Retrieved Passages:**

  1. *AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models*
  2. *OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model*
  3. *Open Vision Reasoner: Transferring Linguistic Cognitive Behavior for Visual Reasoning*

* **Latency:**

  * Embedding: 0.079s
  * Search: 1.365s
  * Generation: 15.26s
  * **Total:** 16.71s

---

## 🏗️ Architecture & Tech Stack

### Frontend

* **Streamlit** — Interactive web framework
* **Custom CSS** — Modern responsive design
* **Plotly** — Interactive latency visualizations
* **Lottie** — Smooth animations

### Backend

* **OpenAI GPT-4** — Language model for generation
* **Pinecone** — Vector database for semantic search
* **SentenceTransformers** — Text embedding models
* **arXiv API** — Academic paper data source

### Deployment

* **Streamlit Cloud** — Live hosting platform
* **GitHub Integration** — Continuous deployment

---

## 📂 Files

* `main.py` — Main pipeline script (data ingestion, embedding, upsert, CLI query)
* `app.py` — Streamlit web UI for interactive querying, theme toggle & latency visualization
* `latency_log.csv` — Logs of queries & latency
* `README.md` — This file
* `requirements.txt` — Python dependencies

---

## 🖼️ Screenshots

### Main UI

![Main UI](Screenshots/screenshot1.jpg)

### Query Example

![Query Example](Screenshots/screenshot2.jpg)

### Latency Metrics

![Latency Metrics](Screenshots/screenshot3.jpg)

### Theme Toggle

![Theme Toggle](Screenshots/screenshot4.jpg)

### Retrieved Passages

![Retrieved Passages](Screenshots/screenshot5.jpg)

---

## 🔗 Links

* 🌐 **Live Demo:** [https://miniragpipeline.streamlit.app/](https://miniragpipeline.streamlit.app/)
* 📂 **GitHub Repository:** [PineRAG---Fast-RAG-with-Latency-Analytics](https://github.com/vijayshreepathak/PineRAG---Fast-RAG-with-Latency-Analytics)
* 👨‍💻 **Developer Portfolio:** [vijayshreepathak.netlify.app](https://vijayshreepathak.netlify.app/)
* 💼 **LinkedIn:** [vijayshreevaibhav](https://www.linkedin.com/in/vijayshreevaibhav)

---

## 🙏 Credits

Built with ❤️ by **Vijayshree Vaibhav** | 7× National Hackathon Winner
[🌐 Portfolio](https://vijayshreepathak.netlify.app/) | [💼 LinkedIn](https://www.linkedin.com/in/vijayshreevaibhav)

⭐ **If you find this project helpful, please star the repository!**

---
