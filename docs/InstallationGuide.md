## 🚀 RAGGIN Introduction

**RAGGIN** (Retrieval-Augmented Generation for Guided Intelligence in Next.js) is a powerful **Visual Studio Code extension** designed to assist developers working with the **Next.js framework**. It provides intelligent, real-time support for tasks such as:

- Questions answering
- Code generation, explanation, and suggestions


RAGGIN delivers answers directly inside the editor, so developers can stay focused in their coding environment without needing to switch between browser tabs or external documentation tools.

One of RAGGIN's key features is the ability to ask questions based on a specific version of the Next.js documentation. This ensures that answers are accurate and relevant to the version you're working with, helping avoid deprecated or outdated information.

RAGGIN operates **100% locally**, powered by **Docker** and **Ollama**, eliminating the need for an internet connection during use. However, some resources—such as versioned Next.js documentation and local LLMs—must be downloaded beforehand.

By ensuring that all processing happens on your machine, RAGGIN maintains **data privacy** and keeps your development environment secure from external data exposure. It's a smart, privacy-focused solution for modern web developers.


## 🔧 RAGGIN Installation Guide

To make RAGGIN runable, make sure these components are installed and configured properly:
- [Ollama and LLM](#-ollama-installation-guide)
- [RAGGIN Docker](#-raggin-docker-installation-guide)
- [RAGGIN VS Code Extension](#-raggin-vs-code-extension-installation-guide)

### 🧠 Ollama Installation Guide

RAGGIN relies on **Ollama** to run a local Large Language Model (LLM) for answering Next.js-related questions. To ensure RAGGIN works properly, both **Ollama** and <mark>at least one</mark> **LLM model** must be installed on your system.

#### ✅ Step 1: Install Ollama

Visit [https://ollama.com](https://ollama.com) and download the installer for your operating system. Follow the installation instructions provided on the website.

#### ✅ Step 2: Install an LLM Model

After installing Ollama, open your terminal or command prompt and run the following command to install a model:

```
ollama pull <model-name>
```

For example, to install **qwen:1.8b**, run:

```
ollama pull qwen:1.8b
```

> ℹ️ You may use any supported model available via Ollama. A full list is available at [https://ollama.com/library](https://ollama.com/library).

Once installed, RAGGIN will be able to use the selected model to generate accurate and contextual responses locally.

Here's a more polished and professional version of your installation guide:

---

### 🐳 RAGGIN Docker Installation Guide

To make RAGGIN work properly, ensure that **Docker** is installed and running on your system. You can follow the official installation guide here:  
👉 [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)

#### 🔹 Quick Setup via Docker

To get started immediately with RAGGIN, simply pull the Docker image:

```bash
docker pull melukootto/raggin
```

#### 🔹 Customize Locally

If you wish to modify or explore the source code, you can clone the repository:

```bash
git clone https://github.com/silam741852963/RAGGIN
```

Then, use this command to run docker:

```bash
docker compose up -d --build
```

---

### 🧩 RAGGIN VS Code Extension Installation Guide

To install the RAGGIN extension in Visual Studio Code:

1. Open **VS Code**.
2. Go to the **Extensions Sidebar** (or press `Ctrl+Shift+X`).
3. Search for **"RAGGIN"**.
4. Click **Install**.

Once installed, the extension will automatically connect to your local RAGGIN backend and Ollama, allow you to interact with the Next.js assistant directly from your editor.

You can install any supported version of Next.js directly through the extension, ensuring that your queries are answered with version-accurate information.

### 📊 Performance Benchmarks

The performance of RAGGIN may vary depending on the user's hardware capabilities. Larger LLMs typically require a more powerful GPU to run efficiently. For optimal results, consider using a model that matches your system's resources.

Below is a summary of performance benchmarks we evaluated to help you choose a suitable configuration based on your system's capabilities. These results provide insights into how different models perform under various hardware setups:


| Hardware       | Model         | Average Response Time (s)  |
|:---------------|:-------------:|:--------------------------:|
| RTX 3060 12GB  | qwen:1.8b     | 40                         |
| RTX 3060 12GB  | llama3.2:3b   | 22                         |

> ⚠️ Note: Actual performance may vary depending on system load, GPU availability, and model size. Use this as a general reference for selecting a model that balances speed and accuracy for your development needs. You can view our evaluation for some popular LLMs [here](./LLMEvaluation.md).


## 🤝 Contributors

- [silam741852963](https://github.com/silam741852963)
- [dangbui03](https://github.com/dangbui03)
- [PhucLe03](https://github.com/PhucLe03)

