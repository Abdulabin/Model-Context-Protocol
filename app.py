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
            "tools": [tool.get("name") for tool in data.get("tools", [])]
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
        servers_info = {
            name: {
                "active": data.get("active"),
                "tools": [tool.get("name") for tool in data.get("tools", [])]
            }
            for name, data in mcp_client.servers.items()
        }
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
async def home():
    return RedirectResponse(url="/chat", status_code=status.HTTP_303_SEE_OTHER)

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
            res = await mcp_client.process_user_query(user_query)
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

