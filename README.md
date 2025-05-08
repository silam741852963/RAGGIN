# 🧠 RAGGIN: Retrieval-Augmented Generation for Guided Intelligence in Next.js

![Docker Pulls](https://img.shields.io/docker/pulls/melukootto/raggin)
![License](https://img.shields.io/github/license/silam741852963/raggin)
![Stars](https://img.shields.io/github/stars/silam741852963/raggin?style=social)

> Intelligent code assistant for modern web developers using Next.js — powered by Retrieval-Augmented Generation (RAG) and customizable local LLMs.

---

## ✨ Overview

**RAGGIN** is an open-source system that enhances the web development experience inside **Visual Studio Code**. Using **Retrieval-Augmented Generation (RAG)** and **local Large Language Models (LLMs)**, it helps developers query documentation, understand version changes, and get real-time suggestions — all while staying offline.

![411_KHMT_2153289-2152715-2152239](https://github.com/user-attachments/assets/81fbc98a-9148-4daa-a629-05808facc561)
Here is the completed version with the missing sections filled in:

---

## 🎯 Technical Fair Poster

![411\_KHMT\_2153289-2152715-2152239](https://github.com/user-attachments/assets/81fbc98a-9148-4daa-a629-05808facc561)

---

## 📜 Technical Report

You can access the full **Technical Report** of the RAGGIN project [here](https://github.com/silam741852963/RAGGIN/blob/main/report.pdf).

## Key Features

- 🔍 Semantic search across versioned Next.js documentation  
- ✍️ Intelligent, context-aware code generation and explanation  
- 🧠 Configurable LLM backend (Qwen, Mistral, LLaMA, etc.)  
- 🧩 Seamless integration with VSCode  
- 💡 Supports framework upgrades and migration strategies  
- ⚡ Fully offline-capable using Docker + Ollama

---

## ⚙️ Architecture

### Retriever
- Embeds structured and unstructured Next.js docs using [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- Stores embeddings in [Milvus](https://milvus.io/), supporting hybrid search
- Handles metadata-aware semantic retrieval for high precision

### Generator
- Works with local models via [Ollama](https://ollama.com/)'s API.
- Combines retrieved documentation with user queries for grounded response generation
- Generates helpful code examples, explanations, or upgrade paths

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/silam741852963/RAGGIN.git
cd RAGGIN
```

### 2. Launch Docker Stack

```bash
docker compose -f docker-compose.yml up
```

### 3. Install the VSCode Extension

[RAGGIN](https://marketplace.visualstudio.com/items/?itemName=raggin.raggin)

---

## 📂 Dataset

This project uses a curated dataset of **Next.js documentation (v13.0.0–15.x.x)**, prepared with:

- Sectioned markdown parsing
- Code-aware chunking
- Metadata tagging (e.g., version, page, language)
- Dense and sparse embeddings

📚 Available on Kaggle:  
👉 [Next.js Documentation for RAGGIN](https://www.kaggle.com/datasets/jiyujizai/nextjs-documentation-for-raggin)

---

## 📍 Project Links

- 🔗 GitHub: [github.com/silam741852963/RAGGIN](https://github.com/silam741852963/RAGGIN)
- 🐳 Docker Hub: [melukootto/raggin](https://hub.docker.com/r/melukootto/raggin)
- 📦 Kaggle Dataset: [Next.js Docs for RAGGIN](https://www.kaggle.com/datasets/jiyujizai/nextjs-documentation-for-raggin)
- 💻 VS Code Extension: [RAGGIN](https://marketplace.visualstudio.com/items/?itemName=raggin.raggin)

---

## 🛠️ Tech Stack

| Layer         | Tools/Tech                           |
|---------------|--------------------------------------|
| Embeddings    | BGE-M3 (Dense + Sparse)              |
| Vector DB     | Milvus                               |
| Backend       | FastAPI + LangChain                  |
| LLM Runtime   | Ollama (supports Qwen, LLaMA, etc.)  |
| IDE Plugin    | Visual Studio Code                   |
| Framework     | Next.js (Target)                     |

---

## 🎓 About the Project

This is the graduation project of:

- 🧑‍💻 **Nguyen Trang Sy Lam**, **Le Hoang Phuc**, **Bui Ho Hai Dang**
- 📘 **Faculty of Computer Science & Engineering**  
- 🏫 **HCMUT – Vietnam National University**  
- 👨‍🏫 Supervisors: Dr. Nguyen An Khuong, Pham Nhut Huy (Zalo AI)

---

## 📄 License

MIT License – Use freely for personal, academic, or commercial projects.  
Contributions are welcome via issues and pull requests!

---

## 💬 Contact

Feel free to open an issue or connect via GitHub Discussions.

> ✨ Stay tuned for updates on model support, extension UI, and multi-framework documentation integration!
