# 🧠 RAGGIN: Retrieval-Augmented Generation for Guided Intelligence in Next.js

![Docker Pulls](https://img.shields.io/docker/pulls/melukootto/raggin)
![License](https://img.shields.io/github/license/silam741852963/raggin)
![Stars](https://img.shields.io/github/stars/silam741852963/raggin?style=social)

> Intelligent code assistant for modern web developers using Next.js — powered by Retrieval-Augmented Generation (RAG) and customizable local LLMs.

---

## ✨ Overview

**RAGGIN** is an open-source system that enhances the web development experience inside **Visual Studio Code**. Using **Retrieval-Augmented Generation (RAG)** and **local Large Language Models (LLMs)**, it helps developers query documentation, understand version changes, and get real-time suggestions — all while staying offline.

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

Here’s the updated **Getting Started** section with the new installation guide integrated and the option for both Docker pull and local build preserved:

---

## 🚀 Getting Started

To make **RAGGIN** fully operational, ensure that the following components are installed and configured properly:

1. **Ollama and LLM**
2. **RAGGIN Docker**
3. **RAGGIN VS Code Extension**

---

### 🧠 Ollama Installation Guide

**RAGGIN** relies on **Ollama** to run a local **Large Language Model (LLM)** for answering Next.js-related questions. To ensure proper functionality, both Ollama and at least one LLM model must be installed on your system.

#### ✅ Step 1: Install Ollama

Visit [Ollama's official site](https://ollama.com) and download the installer for your operating system. Follow the installation instructions provided on the website.

#### ✅ Step 2: Install an LLM Model

After installing Ollama, open your terminal or command prompt and run the following command to install a model:

```bash
ollama pull <model-name>
```

For example, to install `qwen:1.8b`, run:

```bash
ollama pull qwen:1.8b
```

ℹ️ **You may use any supported model available via Ollama.**
A full list is available at [Ollama Model Library](https://ollama.com/library).

Once installed, **RAGGIN** will be able to use the selected model to generate accurate and contextual responses locally.

---

### 🐳 RAGGIN Docker Installation Guide

To make **RAGGIN** work properly, ensure that **Docker** is installed and running on your system.
👉 [Docker Installation Guide](https://docs.docker.com/engine/install/)

#### 🔹 **Quick Setup via Docker Hub**

To get started immediately with RAGGIN, simply pull the Docker image:

```bash
docker pull melukootto/raggin
```

And run it with:

```bash
docker compose up -d
```

---

#### 🔹 **Build and Run Locally**

If you want to explore or customize the source code, you can clone the repository:

```bash
git clone https://github.com/silam741852963/RAGGIN
cd RAGGIN
```

Then, use Docker Compose to build and run:

```bash
docker compose up -d --build
```

This method allows you to tweak configurations, adjust volume mounts, or modify code as needed.

---

### 🧩 RAGGIN VS Code Extension Installation Guide

To install the **RAGGIN** extension in **Visual Studio Code**:

1. Open **VS Code**.
2. Go to the **Extensions Sidebar** (or press `Ctrl+Shift+X`).
3. Search for **"RAGGIN"**.
4. Click **Install**.

Once installed, the extension will automatically connect to your local **RAGGIN backend** and **Ollama**, allowing you to interact with the **Next.js assistant** directly from your editor.

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
