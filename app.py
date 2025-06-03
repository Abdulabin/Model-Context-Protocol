from flask import Flask, request, render_template, jsonify, redirect, url_for, session,flash
from markupsafe import Markup
import json
import os
import ast
import asyncio
import markdown
from mcp_client_stdio import MCPClient
app = Flask(__name__)
LOG_FILE = "mcp-client.log"
app.secret_key = "your_secret_key_here"  
import google.generativeai as genai

class McpClientApp(MCPClient):
    def __init__(self):
        super().__init__()
        self.llm_client = None
        
    async def connect_llm(self,llm_config):

        # Validate presence of configuration
        if not llm_config:
            return None, "❌ LLM configuration not found. Please select LLM and API Key."

        provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        api_key = llm_config.get("api_key")

        # Validate provider
        if provider != "google":
            return None, "❌ Only Google Gemini models are supported currently."

        # Validate required fields
        if not model_name or not api_key:
            return None, "❌ Model name or API key missing in LLM config."

        try:
            genai.configure(api_key=api_key)

            
            self.llm_client = genai.GenerativeModel(model_name)
            _ = self.llm_client.generate_content("Test connection to Gemini",)

            return  "✅ Connected to Google Gemini successfully."
        except Exception as e:
            self.llm_client = None
            return  f"❌ Failed to connect to Gemini: {str(e)}"


mcp_client = McpClientApp()

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Log Initialized\n")

@app.template_filter('markdown')
def markdown_filter(text):
    return Markup(markdown.markdown(text))

@app.context_processor
def inject_global_status():
    # You can replace these with actual logic later
    return {
        'llm_connected': mcp_client.llm_client is not None,
        'mcp_server_connected': True
    }

@app.route('/')
def home():
    return redirect(url_for("chat"))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []

    if request.method == 'POST':
        user_query = request.form.get('user_query', '').strip()

        # Check if LLM client is available
        if mcp_client.llm_client is None:
            flash("❌ LLM client not connected. Please configure and connect via LLM Config.", "error")
            return redirect(url_for("llm_config"))

        try:
            # Call the LLM
            response = mcp_client.llm_client.generate_content(user_query)
            assistant_response = str(response.candidates[0].content.parts[0].text)#
            # Save to session chat history
            session['chat_history'].append({
                'user': user_query,
                'assistant': assistant_response
            })
            session.modified = True

        except Exception as e:
            flash(f"❌ Failed to generate response: {str(e)}", "error")

    return render_template("chat.html", chat_history=session.get('chat_history', []))

@app.route('/server-config', methods=['GET', 'POST'])
def server_config():
    config_text = ""
    config_file = "mcp-servers-configs.json"

    # Load existing config
    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            config_text = f.read()

    # Handle POST submission
    if request.method == 'POST':
        try:
            config_text = request.form['server_config']
            config_dict = json.loads(config_text)  
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=4)
            flash("✅ mcp-servers-configs.json updated successfully!", "success")
            config_text = json.dumps(config_dict, indent=4)  # Pretty print
        except json.JSONDecodeError as e:
            flash(f"❌ JSON Error: {e}", "error")
        except Exception as e:
            flash(f"❌ Unexpected Error: {e}", "error")

    return render_template("server_config.html", config_text=config_text)

@app.route('/llm-config', methods=['GET', 'POST'])
def llm_config():
    message = None

    if request.method == 'POST':
        provider = request.form.get('provider')
        model_name = request.form.get('model_name')
        api_key = request.form.get('api_key')

        llm_config_dict = {
            "provider": provider,
            "model_name": model_name,
            "api_key": api_key
        }

        session['llm_config'] = llm_config_dict
        
        flash("✅ LLM Config saved in session!","success")
        try:
            message = asyncio.run(mcp_client.connect_llm(llm_config_dict))
            if not mcp_client.llm_client:
                flash(message, "error")
            else:
                flash(message, "success")
        except Exception as e:
            flash(f"❌ Exception: {str(e)}", "error")

    current_config = session.get('llm_config', {})

    return render_template("llm_config.html", config=current_config)

@app.route('/logs')
def logs():
    return render_template("logs.html")

@app.route('/logs/data')
def logs_data():
    with open(LOG_FILE, "r") as f:
        return jsonify({"logs": f.read()})

if __name__ == '__main__':
    app.run(debug=True)
