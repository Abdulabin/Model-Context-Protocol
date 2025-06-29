import os
import json
import markdown
from typing import Optional
from starlette import status
from markupsafe import Markup
from fastapi.responses import HTMLResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi import FastAPI, HTTPException, Query, Request, Form

from mcp_client import MCPClient

def markdown_filter(text):
    return Markup(markdown.markdown(text or ""))

app = FastAPI()
mcp_client = MCPClient()
app.mount("/static", StaticFiles(directory="statics/css"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.filters["markdown"] = markdown_filter
app.add_middleware(SessionMiddleware, secret_key="your-secret-session-key")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



CHAT_HISTORY_FILE = "chat_history.json"
CONFIG_FILE_PATH = "mcp-servers-configs.json"
LOG_FILE = "mcp-client.log"

def save_chat(response: dict, filename=CHAT_HISTORY_FILE):
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                chats = json.load(f)
                if not isinstance(chats, list):
                    chats = []
        else:
            chats = []
        chats.append(response)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chats, f, indent=4)
    except Exception as e:
        mcp_client.logger.error(f"Failed to save chat: {e}")

def clear_chat(filename=CHAT_HISTORY_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4)

def read_json_file():
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=4)

def save_json_file(config_text: str):
    try:
        config = json.loads(config_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

def get_servers_info():
    return {
        name: {
            "active": data.get("active"),
            "tools": [tool.get("name") for tool in data.get("tools", [])],
            "msg": data.get("msg", None)
        }
        for name, data in mcp_client.servers.items()
    }

def render_server_config_template(request: Request, config_text: str, llm_config: dict, server_names=None, flash_message=None, flash_type=None):
    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config_text": config_text,
        "llm_config": llm_config,
        "server_names": server_names or {},
        "flash_message": flash_message,
        "flash_type": flash_type,

    })


import json
from google.genai import types
from google.genai.types import Part

def build_reformulation_prompt(chats, new_query, n=5):
    system_text = (
        "You are a query reformulator, specializing in context-aware rewriting. Your job is *only* to rewrite user queries.\n"
        "Given a new user query and recent conversation history, your goal is to produce a clear, concise, self-contained, grammatically correct, and correctly spelled query, while trying to understand the user's intent.\n"
        "If the query is contextually relevant to the conversation, reformulate it accordingly.\n"
        "If the query is unrelated, ambiguous, or unclear, return the original query, corrected for grammar and spelling where possible.\n"
        "Do not provide explanations or answer the query; only output the final, reformulated query.\n"
        "\n"
        "**Instructions:**\n"
        "1.  Carefully analyze the user's new query and the preceding conversation to understand the user's intent.\n"
        "2.  Check for grammar and spelling errors in the user's query and correct them.\n"
        "3.  Determine if the new query refers to information discussed earlier in the conversation.\n"
        "4.  If contextually relevant, rewrite the query, incorporating relevant details from the conversation to make it self-contained, grammatically correct, and correctly spelled.\n"
        "5.  If not contextually relevant or unclear, return the original query, after correcting grammar and spelling.\n"
        "\n"
        "### Few-shot Examples:\n"
        "Conversation:\n"
        "User: What’s the latest iPhone model?\n"
        "Assistant: The latest iPhone model is the iPhone 15 Pro Max.\n"
        "User: How much dose it cost?\n"
        "→ Reformulated: How much does the iPhone 15 Pro Max cost?\n"
        "\n"
        "Conversation:\n"
        "User: Who founded Tesla?\n"
        "Assistant: Tesla was founded by Martin Eberhard and Marc Tarpenning, but Elon Musk later became a key figure.\n"
        "User: Were is he now?\n"
        "→ Reformulated: Where is Elon Musk now?\n"
        "\n"
        "Conversation:\n"
        "User: Hi\n"
        "→ Reformulated: Hi\n"
        "\n"
        "Conversation:\n"
        "User: Tell me a story.\n"
        "Assistant: Sure! What kind of story would you like?\n"
        "User: about a cat\n"
        "→ Reformulated: Tell me a story about a cat.\n"
        "\n"
        "Conversation:\n"
        "User:  what is the wether today?\n"
        "→ Reformulated: What is the weather today?\n"
    )

    system_part = types.Content(role="model", parts=[Part.from_text(text=system_text)])
    past_conv = [system_part]

    for msg in chats[-n:]:
        past_conv.append(types.Content(role="user", parts=[Part.from_text(text=msg["User"])]))
        past_conv.append(types.Content(role="assistant", parts=[Part.from_text(text=msg["Assistant"])]))

    past_conv.append(types.Content(role="user", parts=[Part.from_text(text=new_query)]))
    return past_conv

def get_recent_chats(n=5,):
    try:
        with open(CHAT_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

async def reformulate_query(new_query, n=5):
    mcp_client.logger.info(f"-------- Reformulating New User Query  --------")

    mcp_client.logger.info(f"Starting query reformulation for: '{new_query}'")
    
    chats = get_recent_chats(n)
    mcp_client.logger.debug(f"Loaded last {n} chat(s) for context: {len(chats)} message pairs")

    payload = build_reformulation_prompt(chats, new_query, n)
    mcp_client.logger.debug(f"Sending reformulation prompt to LLM with total parts: {len(payload)}")

    try:
        resp = await mcp_client.generate_content(payload, without_tools=True)
        mcp_client.logger.debug(f"Reformulated query: '{resp.text.strip()}'")
        return resp.text
    except Exception as e:
        mcp_client.logger.error(f"Failed to reformulate query: {e}", exc_info=True)
        return new_query 

@app.post("/initialize")
async def initialize_server():
    if not os.path.isfile(CONFIG_FILE_PATH):
        raise HTTPException(status_code=404, detail="Config file not found.")

    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        await mcp_client.initialize_servers(config)
        return {"message": "Servers initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize servers: {e}")

@app.post("/llm")
async def connect_llm():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API key not set in environment.")

    llm_config = {
        "provider": "google",
        "model_name": "gemini-2.0-flash-lite",
        "api_key": api_key
    }

    try:
        msg = await mcp_client.connect_llm(llm_config)
        return {"message": msg}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to LLM: {e}")

@app.post("/query/{user_query}")
async def get_response(user_query: str):
    try:
        res = await mcp_client.process_user_query(user_query)
        response = {"User": user_query, "Assistant": res}
        save_chat(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.get("/servers")
async def get_servers():
    try:
        servers_info = get_servers_info()
        return {"servers": servers_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching servers: {e}")

@app.get("/llm_status")
async def llm_status():
    return {"llm_connected": bool(mcp_client.llm_client)}


@app.get("/chat_history")
async def get_chat_history(filename: str = Query(CHAT_HISTORY_FILE)):
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Chat history file not found.")
    
    try:
        with open(filename, "r", encoding="utf-8") as f:
            chats = json.load(f)
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading chat history: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}

#########################   FRONTEND 

@app.get("/")
async def home(request:Request):
    return templates.TemplateResponse("home.html", {"request": request, "flash_message": None,"flash_type": None,})

@app.get("/settings", name="settings")
async def render_server_configs(request: Request):
    config_text = read_json_file()
    servers_info = get_servers_info()
    llm_config = request.session.get("config", {
        "provider": "",
        "model_name": "",
        "api_key": ""
    })

    return render_server_config_template(
        request=request,
        config_text=config_text,
        llm_config=llm_config,
        server_names=servers_info,
        flash_message=None,
        flash_type=None
    )


@app.post("/settings", response_class=HTMLResponse)
async def handle_server_config(request: Request):
    form = await request.form()
    config_text = form.get("server_config", "")
    llm_config = request.session.get("config", {
        "provider": "",
        "model_name": "",
        "api_key": ""
    })

    flash_message = None
    flash_type = None

    if "save_llm" in form:
        provider = form.get("provider")
        model_name = form.get("model_name")
        api_key = form.get("api_key")

        if not api_key:
            return render_server_config_template(
                request,
                config_text,
                llm_config,
                get_servers_info(),
                flash_message="API key is required.",
                flash_type="error"
            )

        llm_config = {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key
        }
        request.session["config"] = llm_config
        mcp_client.servers = {}
        mcp_client.formatted_tools = []
        mcp_client.llm_client = None

        flash_message = "LLM config saved successfully!"
        flash_type = "success"

        return render_server_config_template(
            request,
            config_text,
            llm_config,
            get_servers_info(),
            flash_message=flash_message,
            flash_type=flash_type
        )

    elif "save_servers" in form:
        try:
            save_json_file(config_text)
            mcp_client.servers = {}
            mcp_client.formatted_tools = []
            mcp_client.llm_client = None
            flash_message = "Server config saved successfully!"
            flash_type = "success"
        except ValueError as e:
            return render_server_config_template(
                request,
                config_text,
                llm_config,
                {},
                flash_message=str(e),
                flash_type="error"
            )

        return render_server_config_template(
            request,
            config_text,
            llm_config,
            get_servers_info(),
            flash_message=flash_message,
            flash_type=flash_type
        )

    elif "load_servers" in form:
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)

            await mcp_client.initialize_servers(config)
            config_text = read_json_file()
            server_info = get_servers_info()

            # Try connecting to LLM automatically after loading servers
            try:
                msg = await mcp_client.connect_llm(llm_config)
                flash_message = f"Servers initialized successfully. {msg}"
                flash_type = "success"
                error_msg = None
            except Exception as llm_error:
                flash_message = "Servers initialized successfully."
                error_msg = f"LLM connection failed: {llm_error}"
                flash_type = "info"

            return render_server_config_template(
                request,
                config_text,
                llm_config,
                server_info,
                flash_message=flash_message,
                flash_type=flash_type if error_msg is None else "success"  # keep success if no error
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize servers: {e}")

    # No valid form action
    return render_server_config_template(
        request,
        config_text,
        llm_config,
        {},
        flash_message="No valid button pressed.",
        flash_type="error"
    )

@app.get("/chat", name="chat")
async def chat_page(request: Request):
    if not os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)  

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        chats = json.load(f)

    flash_message = None
    flash_type = None

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": chats,
            "flash_message": flash_message,
            "flash_type": flash_type,
        },
    )

@app.post("/chat", name="chat")
async def chat_page(request: Request):
    form = await request.form()

    if "clear_chat" in form:
        clear_chat()
        return RedirectResponse(url="/chat", status_code=status.HTTP_303_SEE_OTHER)

    if "ask" in form:
        user_query = form.get("user_query", "").strip()
        if mcp_client.llm_client is None:
            return RedirectResponse(url="/settings", status_code=status.HTTP_303_SEE_OTHER)
        try:
            reform_query = await reformulate_query(user_query,n=5)
            res = await mcp_client.process_user_query(reform_query)
            response = {"User": user_query, "Assistant": res}
            save_chat(response)
            return RedirectResponse(url="/chat", status_code=status.HTTP_303_SEE_OTHER)
        except Exception as e:

            return RedirectResponse(url="/chat", status_code=status.HTTP_303_SEE_OTHER)


@app.get('/logs',name="logs")
def logs(request:Request):
    return templates.TemplateResponse("logs.html", {"request": request, "flash_message": None,
            "flash_type": None,})


@app.get('/logs/data')
def logs_data():
    with open(LOG_FILE, "r") as f:
        return {"logs": f.read()}

