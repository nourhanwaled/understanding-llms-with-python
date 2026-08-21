# 🗺️ AI Learning Roadmap — From LLMs to LangGraph

The goal of this roadmap is to understand AI concepts step by step — starting from the foundations of LLMs and gradually moving toward building applications and AI agents with LangGraph.

---

## 🟢 Part 1 — LLM Foundations

* **What is an LLM?**
  → Simple Next Token Prediction with Python

* **Tokens**
  → Build a tiny tokenizer with Python

* **Token IDs**
  → Convert tokens into numbers

* **Context Window**
  → Simulate a token limit

* **Embeddings**
  → Represent words and text as vectors

* **Cosine Similarity**
  → Build semantic similarity from scratch

* **Bag of Words / TF-IDF**
  → Compare traditional text representation with embeddings

---

## 🟡 Part 2 — Transformers

* **Attention**
  → Simple Query / Key / Value implementation

* **Self-Attention**
  → Understand how words look at other words

* **Multi-Head Attention**
  → Simplified implementation

* **Positional Encoding**
  → Give the model information about word order

* **Transformer Pipeline**

```text
Input
  ↓
Embedding
  ↓
Positional Encoding
  ↓
Attention
  ↓
Feed Forward
  ↓
Output
```

---

## 🟠 Part 3 — LLM Engineering

* **Prompting**
* **System / User / Assistant Messages**
* **Temperature**
* **Top-P**
* **Frequency & Presence Penalty**
* **Structured Output**
* **Model Evaluation**
* **Token-based Pricing & Cost Optimization**

---

## 🔵 Part 4 — RAG

This is where the previous Python concepts start connecting together.

* **Document Loading**
* **Text Cleaning**
* **Chunking**
* **Metadata**
* **Embeddings**
* **Vector Databases**
* **Similarity Search**
* **Top-K Retrieval**
* **MMR**
* **Build a Complete Simple RAG**

### Simple RAG Flow

```text
Documents
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector DB
   ↓
Retriever
   ↓
Context
   ↓
LLM
   ↓
Answer
```

---

## 🟣 Part 5 — LangChain

After understanding the concepts from scratch, we move to using a framework to build applications more easily.

* **LangChain Basics**
* **PromptTemplate**
* **Models**
* **Output Parsers**
* **Chains / LCEL**
* **Retrievers**
* **LangChain RAG**
* **Memory**
* **Conversation History**
* **Build a Small LangChain Application**

---

## 🔴 Part 6 — LangGraph

Finally, we move to building more structured and stateful AI applications.

* **Graph Concept**
* **Nodes**
* **Edges**
* **State**
* **Conditional Edges**
* **Stateful Applications**
* **Threads & Checkpoints**
* **Short-term vs Long-term Memory**
* **Context Summarization**
* **Build a Simple AI Agent with LangGraph**

---

## 🚀 Final Project

After covering all these concepts, we'll combine them into one project and put everything together:

**LLM + Embeddings + RAG + LangChain + LangGraph**

The goal is not just to learn the concepts, but to understand **what happens behind the scenes** and then use the right tools to build real AI applications.

**One concept at a time. 🧠🐍🚀**
