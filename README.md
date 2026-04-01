# 🏥 Medical Diagnosis RAG System
### Retrieval-Augmented Generation for Healthcare Decision Support

> **UT Austin Post Graduate Program in AI** — NLP Capstone Project  
> **Author:** Mohit Kataria | `mokataria@gmail.com`

---

## 📋 Table of Contents
- [Business Context](#business-context)
- [Objective](#objective)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Evaluation Framework](#evaluation-framework)
- [Sample Results](#sample-results)
- [Key Findings & Business Recommendations](#key-findings--business-recommendations)

---

## 🏥 Business Context

The healthcare industry faces a critical challenge: professionals must rapidly access accurate, up-to-date medical knowledge across thousands of conditions, treatments, and protocols — often under time pressure with patient outcomes at stake.

**The problem with raw LLMs in healthcare:**
- Knowledge cutoff limitations mean LLMs cannot answer questions about recent developments
- LLMs hallucinate with dangerous confidence in medical contexts
- Responses lack grounding in authoritative clinical references

This project builds a **production-style RAG system** grounded in the Merck Medical Manual — one of the most authoritative medical references in the world — to demonstrate how LLMs can be safely and reliably deployed as clinical decision-support tools.

**Representative queries the system addresses:**
- *"What is the protocol for managing sepsis in a critical care unit?"*
- *"What are the common symptoms for appendicitis, and can it be cured via medicine?"*
- *"What are the effective treatments for sudden patchy hair loss?"*
- *"What treatments are recommended for traumatic brain injury?"*
- *"What are the precautions for a fractured leg sustained during a hiking trip?"*

---

## 🎯 Objective

Design, build, and evaluate a RAG-based AI solution demonstrating:

1. **Understand** — the limitations of bare LLMs for domain-specific Q&A
2. **Apply** — RAG techniques to ground responses in authoritative clinical sources
3. **Analyse** — the quality improvement from LLM → LLM + Prompting → RAG
4. **Evaluate** — system quality using an LLM-as-judge framework (groundedness + relevance)
5. **Recommend** — actionable insights for production deployment in healthcare settings

---

## 📚 Dataset

**Source:** Merck Manual of Medical Information  
**Format:** PDF  
**Scale:** 4,114 pages across 23 medical sections  
**Coverage:** Disorders, tests, diagnoses, drugs, and treatment protocols  
**Published since:** 1899 — one of the most cited clinical references globally

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                           │
│                                                                 │
│  Medical Manual PDF                                             │
│       │                                                         │
│       ▼                                                         │
│  PyMuPDFLoader ──► RecursiveCharacterTextSplitter ──► Chunks   │
│                    (cl100k_base, 512 tokens, 100 overlap)       │
│                                                                 │
│       ▼                                                         │
│  BAAI/bge-base-en-v1.5 Embeddings                              │
│       │                                                         │
│       ▼                                                         │
│  ChromaDB Vector Store (persisted)                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                              │
│                                                                 │
│  User Query                                                     │
│       │                                                         │
│       ▼                                                         │
│  Embed Query ──► Similarity Search (ChromaDB)                  │
│                       │                                         │
│                       ▼ top-k chunks                            │
│              Context Assembly                                   │
│                       │                                         │
│                       ▼                                         │
│  System Prompt + Context + Question ──► Llama-2-13B-chat       │
│                                         (GGUF, GPU/CPU)        │
│                       │                                         │
│                       ▼                                         │
│                   Response                                      │
│                       │                                         │
│                       ▼                                         │
│         LLM-as-Judge Evaluation                                │
│         (Groundedness + Relevance)                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Stages

This notebook demonstrates a **three-stage progression** that clearly shows the value added at each step:

### Stage 1: Bare LLM (Baseline)
Plain LLM inference without any context or system prompting. Demonstrates limitations including hallucination and lack of grounding.

### Stage 2: LLM + Prompt Engineering
Structured system and user prompt templates guide the LLM's behaviour — constraining it to answer only from provided context and return a defined fallback when information is unavailable.

```
System: "You are an assistant... answer only using the provided context..."
User:   "###Context\n{context}\n\n###Question\n{question}"
```

### Stage 3: Full RAG Pipeline
The complete system — retrieval from ChromaDB vector store + augmented generation. Responses are grounded in the 4,114-page Merck Manual corpus, dramatically reducing hallucination and improving accuracy.

### Stage 4: RAG Evaluation
LLM-as-judge framework scores responses on two dimensions:
- **Groundedness** — Is the answer derived only from the retrieved context?
- **Relevance** — Does the answer address the user's question?

---

## 🛠️ Tech Stack

| Category | Tool / Library |
|---|---|
| **LLM** | Llama-2-13B-chat (GGUF quantized via TheBloke) |
| **LLM Inference** | llama-cpp-python 0.2.28 (GPU: CUBLAS / CPU) |
| **Embeddings** | BAAI/bge-base-en-v1.5 (selected from MTEB leaderboard) |
| **Vector Database** | ChromaDB 1.1.1 (persisted to disk) |
| **Document Loading** | PyMuPDF (pymupdf 1.26.5) |
| **Chunking** | LangChain RecursiveCharacterTextSplitter + tiktoken cl100k_base |
| **Orchestration** | LangChain 0.3.27 + langchain-community 0.3.31 |
| **Model Hub** | HuggingFace Hub (hf_hub_download) |
| **Data** | Pandas, NumPy |
| **Runtime** | Google Colab (GPU) / Jupyter Notebook (CPU) |

---

## 📁 Project Structure

```
Medical-Diagnosis-RAG/
│
├── Full_Code_NLP_RAG_Project_Notebook_.ipynb   # Main notebook
├── medical_diagnosis_manual.pdf                 # Merck Manual (source corpus)
├── Medical_DB/                                  # ChromaDB persisted vector store
│   └── (auto-generated on first run)
└── README.md                                    # This file
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Google Colab (recommended for GPU) or Jupyter Notebook

### Step 1: Install LLM inference backend

**For GPU (recommended):**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.28 --force-reinstall --no-cache-dir
```

**For CPU only:**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=off" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.28 --force-reinstall --no-cache-dir
```

> ⚠️ **Restart your runtime/kernel after this step before proceeding.**

### Step 2: Install all other dependencies
```bash
pip install \
  huggingface_hub==0.35.3 \
  pandas==2.2.2 \
  tiktoken==0.12.0 \
  pymupdf==1.26.5 \
  langchain==0.3.27 \
  langchain-community==0.3.31 \
  chromadb==1.1.1 \
  sentence-transformers==5.1.1 \
  numpy==2.3.3
```

> ⚠️ **Restart your runtime/kernel again after this step.**

### Step 3: Place the data file
Ensure `medical_diagnosis_manual.pdf` is available at:
```
/content/medical_diagnosis_manual.pdf   # Google Colab
```
or update the `pdf_path` variable in the notebook to match your local path.

---

## ▶️ How to Run

Run all cells sequentially from top to bottom. The notebook is structured into clearly labelled sections:

1. **Install dependencies** (restart kernel after each install block)
2. **Import libraries**
3. **Stage 1 — Bare LLM Q&A** (observe baseline limitations)
4. **Stage 2 — LLM + Prompt Engineering** (observe improvement)
5. **Stage 3 — Data preparation**: load PDF → chunk → embed → build ChromaDB
6. **Stage 4 — RAG Q&A** (full grounded responses)
7. **Stage 5 — Parameter fine-tuning** (k, max_tokens, temperature)
8. **Stage 6 — Evaluation** (LLM-as-judge groundedness + relevance scoring)

### Key parameters to tune

| Parameter | Default | Effect |
|---|---|---|
| `chunk_size` | 512 | Larger = more context per chunk, slower retrieval |
| `chunk_overlap` | 100 | Higher = better continuity, more redundancy |
| `k` | 3 | Number of chunks retrieved per query |
| `max_tokens` | 128–512 | Response length ceiling |
| `temperature` | 0 | Higher = more creative, lower = more deterministic |

---

## 📊 Evaluation Framework

The system is evaluated using an **LLM-as-judge** approach — the same Llama-2 model scores its own outputs on two criteria:

### Groundedness
*"Is the answer derived only from the information presented in the context?"*

Scoring rubric (1–5):
- **1** — Not derived from context at all
- **3** — Partially grounded in context
- **5** — Fully derived from context with no unsupported claims

### Relevance
*"Does the answer address the main aspects of the question?"*

Scoring rubric (1–5):
- **1** — Answer does not address the question
- **3** — Answer partially addresses the question
- **5** — Answer completely and accurately addresses the question

Both scores are returned as structured JSON with a step-by-step reasoning trace from the judge model.

---

## 🔬 Sample Results

### Stage Comparison — Sepsis Protocol Query

| Stage | Response Quality |
|---|---|
| **Bare LLM** | Generic response with potential hallucinations, no citation to clinical protocol |
| **LLM + Prompting** | More structured, but still limited by model training data |
| **RAG** | Grounded response derived directly from Merck Manual clinical protocols |

### Parameter Tuning Observations
- **k=3 → k=2**: Reduced noise for simpler queries; complex multi-part questions benefited from k=3
- **max_tokens=256**: Caused truncation on longer answers; 512 is the optimal ceiling
- **temperature=0**: Deterministic outputs preferred for medical Q&A to reduce hallucination risk

---

## 💡 Key Findings & Business Recommendations

### Findings
1. **RAG significantly outperforms bare LLM** for domain-specific medical queries — both in accuracy and groundedness
2. **Prompt engineering alone provides meaningful improvement** but cannot substitute for retrieval-grounded context
3. **k=3 with max_tokens=512** is the optimal baseline configuration for complex medical queries
4. **Embedding model choice matters** — BAAI/bge-base-en-v1.5 (top MTEB retrieval performer) delivers strong semantic matching across medical terminology

### Business Recommendations
1. **Deploy as a decision-support layer, not a replacement** — position the system as a tool to assist clinical professionals, clearly disclosing AI involvement to maintain regulatory compliance and trust
2. **Use a smaller 7B model** (e.g. LLaMA-2-7B) for production — delivers comparable quality with significantly better latency and lower infrastructure cost
3. **Implement groundedness monitoring** — track groundedness scores over time to detect model drift, unsupported claims, and hallucination patterns before they reach users
4. **Evaluate faithfulness separately from relevance** — a response can be relevant but not grounded; both dimensions are required for safe clinical deployment
5. **Prioritise citation transparency** — augment responses with source page references from the Merck Manual to enable clinical professionals to verify information directly

---

## 🔗 Related Projects

- [RAG-With-Trust](https://github.com/mohitkataria-25/RAG-With-Trust) — Production-architected RAG system with modular design and trust evaluation framework
- [AI-ML-Portfolio](https://github.com/mohitkataria-25/AI-ML-Portfolio) — Full portfolio of ML and deep learning projects

---

## 👨‍💻 Author

**Mohit Kataria**  
Senior Data Engineer | AI/ML Platform Specialist  
`mokataria@gmail.com` | [LinkedIn](https://linkedin.com/in/mohit-k-5241bb99/) | [GitHub](https://github.com/mohitkataria-25)

*Post Graduate Program in Artificial Intelligence — University of Texas at Austin & Great Learning (2024–2025)*
