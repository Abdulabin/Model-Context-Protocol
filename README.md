# Model Context Protocol (MCP) Client

Unleash the full potential of Large Language Models (LLMs) by connecting them to a universe of external tools! This project presents a robust, multi-server client built on the **Model Context Protocol (MCP)**, enabling your LLM to discover, understand, and invoke custom functionalities. Seamlessly integrate tools via STDIO or SSE, and orchestrate complex tasks with Google's powerful Gemini models.

---

## 🚀 Core Concept

Modern LLMs are incredibly powerful but are often limited to the information they were trained on. The **Model Context Protocol (MCP)** is a specification that allows an LLM to discover and invoke external tools securely and efficiently.

This project implements an MCP client that acts as an intelligent orchestrator. It takes a user's query, consults an LLM (like Gemini) to form a plan, and then calls the necessary tools hosted on various MCP servers to gather information and execute tasks.

---

## 🏛️ Architecture

The architecture is designed for scalability and flexibility, allowing you to add new tools and servers without modifying the core client logic.

```
+-----------------+      +------------------+      +-----------------+
|                 |      |                  |      |                 |
|   User Query    |----->|    MCP Client    |<---->|   Google Gemini |
| (CLI/UI/API)    |      | (mcp_client.py)  |      |       LLM       |
|                 |      |                  |      |                 |
+-----------------+      +--------+---------+      +-----------------+
                                  |
                                  | (Tool Discovery & Invocation)
                                  |
                  +---------------+---------------+
                  |                               |
      +-----------v-----------+       +-----------v-----------+
      |      MCP Server 1     |       |      MCP Server 2     |
      | (e.g., arxiv_server)  |       | (e.g., another_tool)  |
      | (Transport: STDIO)    |       | (Transport: SSE)      |
      +-----------+-----------+       +-----------+-----------+
                  |                               |
          +-------v-------+               +-------v-------+
          |   Tool A, B   |               |   Tool C, D   |
          +---------------+               +---------------+
```

---

## ✨ Features

*   **Multi-Server Connectivity**: Connect to multiple MCP servers simultaneously.
*   **Flexible Transports**: Supports both `stdio` for local process-based servers and `sse` (Server-Sent Events) for remote HTTP-based servers.
*   **LLM-Powered Orchestration**: Leverages Google's Gemini models to understand user intent, create execution plans, and call the right tools.
*   **Web Interface (FastAPI)**: An optional, user-friendly web interface built with FastAPI, providing a RESTful API and interactive documentation (Swagger UI/ReDoc) for easy interaction.
*   **Dynamic Tool Discovery**: Automatically discovers and registers tools from all connected servers.
*   **Asynchronous Architecture**: Built with `asyncio` for high-performance, non-blocking I/O.
*   **Easy Configuration**: Simple JSON-based configuration for servers and `.env` for credentials.
*   **Detailed Logging**: Comprehensive logging for easy debugging and tracing of the agent's thought process.

---

## 🏁 Getting Started

Follow these steps to get the MCP Client up and running on your local machine.

### 1. Prerequisites

*   Python 3.10+
*   An active Google AI Studio account and API Key.

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abdulabin/Model-Context-Protocol.git
    cd Model-Context-Protocol
    ```

2.  **Install dependencies** :
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration
⚙️
1.  **Set up your Google API Key:**
    Create a file named `.env` in the project root and add your Google AI Studio API key:
    ```env
    # .env
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

2.  **Configure your MCP Servers:**
    The client uses `mcp-servers-configs.json` to connect to tool servers. You can enable, disable, and configure servers here.

    **Example `mcp-servers-configs.json`:**
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
    *   `disabled`: Set to `false` to enable a server.
    *   `transport_type`: Can be `stdio` or `sse`.
    *   `stdio` requires `command` and `args` to launch the server as a subprocess.
    *   `sse` requires the `url` of the remote server's MCP endpoint.


---

## 🚀 Usage

The MCP Client offers both a command-line interface (CLI) for direct interaction and an optional web interface for API access and a user-friendly UI.

### CLI Usage 💻

Once the client is running, you can start asking questions that require tool usage directly in your terminal.

**Example Query:**
> Find 3 recent papers on "Large Language Models" and then get me the details for the first one.

The client will intelligently chain tool calls—first searching for papers and then getting details for a specific one—before presenting a final, summarized answer.

---

### Web Interface  🌐

For a more interactive experience, you can run the FastAPI application which exposes the MCP Client's capabilities via a REST API. This is perfect for integrating with other services or simply using a browser-based UI.

#### Features of the Web Interface:
*   **RESTful API**: Programmatic access to the MCP Client's query processing.
*   **Interactive Documentation**: Automatically generated Swagger UI (`/docs`) and ReDoc (`/redoc`) for exploring endpoints and testing queries directly in your browser.
*   **Easy Integration**: Simplifies connecting the MCP Client to front-end applications or other backend services.

#### Running the Web Interface:
1.  **Ensure your MCP Servers are configured and running** (especially for SSE transport).
2.  **Start the FastAPI application:**
    ```bash
    uvicorn app:app 
    ```
    *   `app:app`: Refers to the `app` object within `app.py`.


3.  **Access the UI:**
    *   Open your web browser and navigate to: `http://localhost:8000/docs` (for Swagger UI)
    *   Or: `http://localhost:8000/redoc` (for ReDoc)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

