import os
import json
import shutil
import asyncio
import logging
import warnings

from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration, Part

warnings.filterwarnings("ignore")

class MCPClientError(Exception):
    pass

class MCPClient:

    def __init__(self) -> None:
        """Initializes the MCPClient."""
        load_dotenv()
        self.logger = self._setup_logger()
        self.logger.info("MCPClient Object Created...")

        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.cleanup_lock: asyncio.Lock = asyncio.Lock()
        
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.formatted_tools: List[Tool] = []
        
        self.model_name: Optional[str] = None
        self.llm_client: Optional[genai.Client] = None

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("mcp-client")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler("mcp-client.log", mode="a")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    async def initialize_servers(self, config: dict) -> None:

        self.logger.info("Initializing MCP servers...")
        await self._clear_server_resources()

        server_config = config.get("mcpServers", {})
        if not server_config:
            self.logger.warning("No MCP servers found in the configuration.")
            return

        for name, conf in server_config.items():
            if conf.get("disabled", False):
                self.logger.info(f"Server '{name}' is disabled, skipping.")
                self.servers[name] = {"session": None, "active": False, "tools": [],"msg":None}
                continue

            self.logger.info(f"Setting up server '{name}'")
            transport_type = conf.get("transport_type")
            if not transport_type:
                self.logger.warning(f"Transport type not specified for server '{name}'. Skipping.")
                self.servers[name] = {"session": None, "active": False, "tools": [],"msg":f"Transport type not specified for server '{name}'."}
                continue

            try:
                session = await self._connect_to_server(name, transport_type.lower(), conf)
                if session:
                    await self._register_server_tools(name, session)
            except Exception as e:
                self.logger.error(f"Failed to initialize server '{name}': {e}", exc_info=True)
                self.servers[name] = {"session": None, "active": False, "tools": [],"msg":f"Failed to initialize server '{name}':{e.__str__().splitlines()[0]}"}
                # await self.shutdown()
                # raise MCPClientError(f"Failed to initialize server '{name}'") from e
        self.logger.info("Server initialization complete.")

    async def _connect_to_server(self, name: str, transport_type: str, conf: dict) -> Optional[ClientSession]:
        self.logger.debug(f"Connecting to server '{name}' via {transport_type}.")
        if transport_type == "stdio":

            cmd = conf.get("command")
            args = conf.get("args", [])
            if not isinstance(cmd, str) or not args:
                self.logger.error(f"Invalid command or args for server '{name}'")
                    # raise MCPClientError(f"Invalid command or args for server '{name}'")

            binary = shutil.which(cmd) or cmd
            env = conf.get("env")
            params = StdioServerParameters(command=binary, args=args, env=env)
            transport = await self.exit_stack.enter_async_context(stdio_client(params))
        elif transport_type == "sse":

            server_url = conf.get("url",None)
            if not server_url:
                self.logger.error(f"URL not specified for SSE server '{name}'")

                # raise MCPClientError(f"URL not specified for SSE server '{name}'")
            timeout = conf.get("timeout",60)
            transport = await self.exit_stack.enter_async_context(sse_client(url=server_url, timeout=timeout))
        else:
            self.logger.warning(f"Unsupported transport type '{transport_type}' for server '{name}'.")
            return None

        reader, writer = transport
        session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))
        await session.initialize()
        self.logger.info(f"Successfully connected to server '{name}'.")
        return session

    async def _register_server_tools(self, name: str, session: ClientSession):
        resp = await session.list_tools()
        tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in resp.tools]

        for tool_info in tools:
            tool_name = tool_info["name"]
            self.formatted_tools.append(self.format_tool(tool_info))
            self.tool_to_session[tool_name] = session
            self.logger.debug(f"Registered tool '{tool_name}' from server '{name}'.")

        self.logger.info(f"Server '{name}' registered {len(tools)} tools.")
        self.servers[name] = {
            "session": session,
            "active": True,
            "tools": tools,
            "msg":None
        }

    def clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return schema
        schema.pop("title", None)
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                props[key] = self.clean_schema(sub)
        return schema

    def format_tool(self, info: Dict[str, Any]) -> Tool:
        self.logger.debug(f"Formatting tool: {info['name']}")
        fn_decl = FunctionDeclaration(
            name=info["name"],
            description=info["description"],
            parameters=self.clean_schema(json.loads(json.dumps(info["input_schema"])))
        )
        return Tool(function_declarations=[fn_decl])

    async def connect_llm(self, config: dict) -> str:
        self.logger.info("Connecting to LLM...")
        self.model_name = config.get("model_name")
        api_key = config.get("api_key")

        if not api_key or not self.model_name:
            self.logger.error("LLM API key or model name is missing in the configuration.")
            # raise MCPClientError("Missing LLM configuration")

        try:
            self.llm_client = genai.Client(api_key=api_key)
            # A simple ping to verify connectivity and credentials
            await self.generate_content([types.Content(role="user", parts=[types.Part.from_text(text="Ping")])])
            self.logger.info(f"Successfully connected to LLM model: {self.model_name}")
            return f"Connected to LLM ({self.model_name}) successfully"
        except Exception as e:
            self.llm_client = None
            self.logger.error(f"Failed to connect to LLM: {e}", exc_info=True)
            # raise MCPClientError(f"Failed to connect to LLM: {str(e)}")

    async def generate_content(self, contents: List[types.Content], without_tools= False) -> types.GenerateContentResponse:
        if not self.llm_client or not self.model_name:
            self.logger.error("LLM client or model name not initialized before calling generate_content.")
            raise RuntimeError("LLM not initialized. Call connect_llm first.")

        self.logger.debug(f"Generating content with model '{self.model_name}' and {len(self.formatted_tools)} tools.")
        try:
            tool_config= None
            if not without_tools:
                tool_config = types.GenerateContentConfig(tools=self.formatted_tools) if self.formatted_tools else None
            
            return self.llm_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=tool_config
            )
        except Exception as e:
            self.logger.error(f"LLM content generation failed: {e}", exc_info=True)
            # raise MCPClientError(f"LLM generation failed: {str(e)}")

    async def invoke_tool(self, name: str, args: Any, retries: int = 2, delay: float = 1.0) -> Dict[str, Any]:
        self.logger.info(f"Attempting to invoke tool '{name}' with args: {args}")
        session = self.tool_to_session.get(name)
        if not session:
            self.logger.error(f"Tool '{name}' not found or its server is inactive.")
            # raise MCPClientError(f"Tool '{name}' not available")

        for attempt in range(retries):
            try:
                result = await session.call_tool(name, args)
                self.logger.info(f"Tool '{name}' invoked successfully.")
                return {"result": result.content}
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{retries} for tool '{name}' failed: {e}", exc_info=True)
                if attempt + 1 < retries:
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Tool '{name}' failed after {retries} attempts.")
                    return {"error": f"Tool '{name}' failed after {retries} attempts with error: {str(e)}"}
        return {"error": f"Tool '{name}' failed after all retries."} # Should not be reached

    async def process_user_query(self, query:str, max_tool_calls:int = 10) -> str:
        def mentions_tool_call(resp_content: Part) -> bool:
            return any(getattr(part, "function_call", None) for part in resp_content.parts)

        system_prompt = """You are a smart and helpful assistant with access to specific tools.

Your tasks:

1. **Determine Query Type**
   - If a question can be answered directly, provide the answer clearly.
   - If tools are needed, design a workflow using ONLY the available tools.

2. **Using Tools**
   - Create a clear, step-by-step plan to solve the query.
   - In your **first response**:
     - Briefly explain your plan.
     - Immediately invoke the **first tool** using `function_call`. Do **not** delay execution.
   - After each tool call:
     - Analyze the result.
     - If more steps are needed, call the next tool using `function_call`.
     - Repeat until the task is complete.

3. **Handling Tool Responses**
   - Use tool outputs to inform the next step.
   - Extract relevant insights — avoid echoing raw data.
   - Never simulate tool outputs or do calculations manually.

4. **Final Answer Formatting**
   - Format results clearly using **Markdown**:
     - Use headings, bullet points, or code blocks when helpful.
   - Ensure the final response is concise, readable, and informative.

**Rules**
- Always use `function_call` for tool invocations.
- Do not simulate or invent tools.
- Only respond with plain text if tools are not needed.
- Do **not** mention whether tools were used or not.

Your role: Reason, plan, and act — one tool call per turn — until the task is fully resolved.
"""

        self.logger.info(f"-------- New Query Processing Started --------")
        self.logger.info(f"User Query: '{query}'")
        
        system_part = types.Content(role="model", parts=[types.Part.from_text(text=system_prompt)])
        user_part = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        contents = [system_part, user_part]
        call_num = 0

        while call_num < max_tool_calls:
            self.logger.info(f"Tool call iteration {call_num + 1}/{max_tool_calls}")
            self.logger.debug(f"Sending to LLM. Conversation history length: {len(contents)}")
            
            try:
                response = await self.generate_content(contents)

            except Exception as e:
                self.logger.error(f"Error generating content: {e}", exc_info=True)
                raise MCPClientError(f"Error generating content: {e}")
            
            resp_content = response.candidates[0].content
            contents.append(resp_content)
            self.logger.debug(f"LLM response received: {resp_content.parts}")

            if mentions_tool_call(resp_content):
                for part in resp_content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_name = part.function_call.name
                        tool_args = dict(part.function_call.args)
                        self.logger.info(f"LLM requested tool call: `{tool_name}` with args: {tool_args}")
                        try:
                            tool_result = await self.invoke_tool(tool_name, tool_args)
                            self.logger.info(f"Tool `{tool_name}` result: {tool_result}")
                        except Exception as e:
                            self.logger.error(f"Tool `{tool_name}` failed during invocation: {e}", exc_info=True)
                            # raise MCPClientError(f"Tool `{tool_name}` failed: {e}")
                        # tool_result = await self.tool_invoke_approval(tool_name,tool_args)
                        tool_response_part = types.Part.from_function_response(name=tool_name, response=tool_result)
                        tool_response_content = types.Content(role="tool", parts=[tool_response_part])
                        contents.append(tool_response_content)
                call_num += 1
            else:
                final_answer = response.text
                self.logger.info(f"Final Answer generated: {final_answer}")
                self.logger.info(f"-------- Query Processing Finished --------")
                return final_answer
        else:
            self.logger.warning(f"Reached maximum tool call limit of {max_tool_calls}.")
            return f"Reached maximum tool call limit ({max_tool_calls}). Please simplify the query or increase the limit."
        
    async def tool_invoke_approval(self, tool_name, tool_args):

        print("\n🔧 Tool Invocation Request")
        print(f"Tool Name : {tool_name}")
        print(f"Arguments : {tool_args}")
        print("\nType 'y' to approve or 'n' to reject this tool call.")

        user_response = ""
        while user_response not in ("y", "n"):
            user_response = input("Approve? [y/n]: ").strip().lower()

        if user_response == "y":
            try:
                tool_result = await self.invoke_tool(tool_name, tool_args)
                self.logger.info(f"Tool `{tool_name}` executed successfully. Result: {tool_result}")
                return tool_result
            except Exception as e:
                self.logger.error(f"Tool `{tool_name}` failed during execution: {e}", exc_info=True)
                # raise MCPClientError(f"Tool `{tool_name}` execution failed: {e}")

        else:
            msg = f"Tool invocation for `{tool_name}` was rejected by the user."
            self.logger.warning(msg)
            return {
                "result": "User rejected the tool invocation request. Chat halted due to lack of required tool execution."
            }

    async def _clear_server_resources(self):
        self.logger.debug("Clearing existing server resources.")
        self.servers.clear()
        self.formatted_tools.clear()
        self.tool_to_session.clear()

    async def shutdown(self):
        async with self.cleanup_lock:
            if not self.servers and not self.llm_client:
                self.logger.info("Shutdown called but no active resources to clean up.")
                return

            self.logger.info("Shutting down MCPClient...")
            try:
                await self.exit_stack.aclose()
                self.logger.info("Async exit stack closed successfully.")
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            finally:
                await self._clear_server_resources()
                self.llm_client = None
                self.model_name = None
                self.logger.info("All client resources have been cleared.")

async def main():
    load_dotenv()
    api_key = "AIzaSyACaAQPODxmm5WVoust1O1QKG_f8nWXRLo" #os.environ.get("GEMINI_API_KEY")
    model_name = "gemini-2.0-flash-lite"
    llm_config = {
        "provider": "google",
        "model_name": model_name,
        "api_key": api_key
    }
    config_path = "mcp-servers-configs.json"
    client = MCPClient()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        await client.initialize_servers(config)
        await client.connect_llm(llm_config)
        
        print("\nMCP Client Ready! (type 'quit' to exit)\n")
        try:
            while True:
                query = input("Query: ").strip()
                if query.lower() == "quit":
                    break
                try:
                    answer = await client.process_user_query(query)
                    print("\n" + answer + "\n")
                except MCPClientError as e:
                    print(f"\n[Error] {e}\n")
        finally:
            await client.shutdown()

    except (MCPClientError, FileNotFoundError) as e:
        client.logger.error(f"Fatal error during setup: {e}", exc_info=True)
    finally:
        await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
