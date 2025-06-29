
# Model Context Protocol (MCP) Client

Unleash the full potential of Large Language Models (LLMs) by connecting them to a universe of external tools! This project presents a robust, multi-server client built on the **Model Context Protocol (MCP)**, enabling your LLM to discover, understand, and invoke custom functionalities. Seamlessly integrate tools via STDIO or SSE, orchestrate complex tasks with Google's powerful Gemini models, and chat across **multi-turn conversations**.

---

## 🚀 Core Concept

Modern LLMs are incredibly powerful but are often limited to the information they were trained on. The **Model Context Protocol (MCP)** is a specification that allows an LLM to discover and invoke external tools securely and efficiently.

This project implements an MCP client that acts as an intelligent orchestrator. It takes a user's query, consults an LLM (like Gemini) to form a plan, and then calls the necessary tools hosted on various MCP servers to gather information and execute tasks — all while supporting **context-aware, multi-turn interactions**.

---

## 🏛️ Architecture

The architecture is designed for scalability and flexibility, allowing you to add new tools and servers without modifying the core client logic.

![MCP Architecture](https://miro.medium.com/v2/resize:fit:936/1*Sa0JkZPtlfD3oZyxprcV1A.png)

---

## ✨ Features

* ✅ **Multi-Turn Conversation Support**: Maintain memory across chat turns for smarter, context-aware interactions.
* 🌐 **Multi-Server Connectivity**: Connect to multiple MCP servers simultaneously.
* 🔌 **Flexible Transports**: Supports both `stdio` for local process-based servers and `sse` (Server-Sent Events) for remote HTTP-based servers.
* 🧠 **LLM-Powered Orchestration**: Uses Gemini to interpret complex requests, plan tool usage, and deliver intelligent responses.
* 🖥️ **Optional Web UI (FastAPI)**: RESTful API with Swagger/ReDoc docs for intuitive browsing and usage.
* 🔍 **Tool Discovery**: Dynamically detects and loads available tools from all connected servers.
* ⚙️ **UI-Based Config & Logs**: Configure API keys, tools, and servers from the web interface. Easily view logs and debug sessions.
* ⚡ **Async Architecture**: Built on `asyncio` for non-blocking performance.
* 🧾 **Readable Configuration**: JSON-based server configs and `.env` for secure credential management.
* 🛠 **Dev-Friendly Logs**: Inspect agent reasoning and tool calls for full transparency.

---

## 🏁 Getting Started

### 1. Prerequisites

* Python 3.10+
* Google AI Studio API key

### 2. Installation

```bash
git clone https://github.com/Abdulabin/Model-Context-Protocol.git
cd Model-Context-Protocol
pip install -r requirements.txt
```

### 3. Configuration

**.env File (API Key):**

```env
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

**Tool Server Config (`mcp-servers-configs.json`):**

```json
{
  "mcpServers": {
    "arxiv_server": {
      "disabled": false,
      "transport_type": "stdio",
      "command": "python",
      "args": ["mcp_servers/arxiv_server.py"]
    },
    "remote_tool_server": {
      "disabled": true,
      "transport_type": "sse",
      "url": "http://localhost:8001/mcp"
    }
  }
}
```

---

## 🚀 Usage

### CLI Usage 💻

```bash
python mcp_client.py
```

Then ask questions like:

> Find 3 recent papers on "Large Language Models" and summarize the first one.

The client understands the multi-step process, delegates tasks to appropriate tools, and returns an answer — all with memory across interactions.

---

### Web Interface 🌐

**Launch the UI with:**

```bash
uvicorn app:app
```

**Open in Browser:**

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

#### Web UI Capabilities:

* 🧠 Chat with memory (multi-turn conversation)
* 🛠 Add/Edit tool servers
* 🔐 Set API keys and model configs
* 📜 View detailed logs of interactions

---

## 🧾 Tool Config JSON Schema (UI Compatible)

Supports both `stdio` and `sse`. Example UI-loadable config:

```json
[
  {
    "mcpServers": {
      "LocalScript": {
        "disabled": false,
        "command": "python3",
        "args": ["script.py"],
        "transport_type": "stdio"
      },
      "WeatherAPI": {
        "disabled": true,
        "transport_type": "sse",
        "url": "http://api.weather.com/sse",
        "timeout": 30
      }
    }
  }
]
```

---

## 🧭 Pages in the Web App

| Page        | Purpose                                   |
| ----------- | ----------------------------------------- |
| 🏠 Home     | Introduction and overview                 |
| ⚙️ Settings | Add API keys, select models, manage tools |
| 💬 Chat     | Chat with the LLM + tools (multi-turn)    |
| 📜 Logs     | Debug and monitor tool calls and plans    |

---

## 🎯 Ideal Use-Cases

* 🤖 AI Assistants with real-time tools
* 📚 Academic research bots using APIs
* 🧪 AI development environments
* 🎓 Educational demos combining LLMs + tooling

---

## ✅ Summary

**MCP-Client** bridges LLMs and real-world tools with seamless configuration, extensible architecture, and support for intelligent, **multi-turn conversations**. Whether you're building smart agents, research assistants, or developer tools — it's your gateway to next-level AI integration.

> Connect your tools. Configure your model. Start chatting. 🚀
