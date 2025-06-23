import os
import json
import shutil
import asyncio
import logging
import warnings

from contextlib import AsyncExitStack
from typing import Any, Dict, List

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration

warnings.filterwarnings("ignore")

class MCPClientError(Exception):
    pass

class MCPClient:
    def __init__(self) -> None:
        load_dotenv()
        self.logger = logging.getLogger("mcp")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler("mcp-client.log", mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("MCPClient initialized")

        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.servers = {}
        self.formatted_tools: List[Tool] = []
        self.model_name = None
        self.llm_client = None

    async def initialize_servers(self, config: dict) -> None:
        self.logger.info("Initializing servers")
        self.servers = {}
        server_config = config.get("mcpServers", {})

        for name, conf in server_config.items():
            if conf.get("disabled", True):
                self.logger.info(f"Server '{name}' is disabled, skipping")
                self.servers[name] = {"session": None, "active": False, "tools": []}
                continue

            self.logger.info(f"Setting up server '{name}'")
            transport_type = conf.get("transport_type", None)
            if transport_type is None:
                self.logger.info(f"Server '{name}' Transport not specified")
                self.servers[name] = {"session": None, "active": False, "tools": []}
                continue
            try:
            
                if transport_type.lower() =="stdio":
                    self.logger.info("Opening STDIO transport connection...")

                    cmd = conf.get("command")
                    args = conf.get("args", [])
                    if not isinstance(cmd, str) or len(args)==0:
                        self.logger.error(f"Invalid command or args for server '{name}'")
                        raise MCPClientError(f"Invalid command or args for server '{name}'")

                    binary = shutil.which(cmd) or cmd
                    env = conf.get("env") or None
                    params = StdioServerParameters(command=binary, args=args, env=env)
                    transport = await self.exit_stack.enter_async_context(stdio_client(params))
                    reader, writer = transport
                    session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))

                if transport_type.lower() =="sse":
                    self.logger.info("Opening SSE transport connection...")

                    server_url = conf.get("url",None)
                    timeout = conf.get("timeout",60)
                    stdio_transport = await self.exit_stack.enter_async_context(sse_client(url=server_url, timeout=timeout,))
                    reader, writer = stdio_transport
                    session = await self.exit_stack.enter_async_context(ClientSession(reader, writer))


                await session.initialize()

                resp = await session.list_tools()
                tools = [{
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                } for tool in resp.tools]

                for tool in tools:
                    self.formatted_tools.append(self.format_tool(tool))

                self.logger.info(f"Server '{name}' registered {len(tools)} tools")
                self.servers[name] = {
                    "session": session,
                    "active": True,
                    "tools": tools
                }

            except Exception as e:
                self.logger.error(f"Failed to initialize server '{name}': {e}", exc_info=True)
                self.servers[name] = {"session": None, "active": False, "tools": []}
                await self.shutdown()
                raise MCPClientError(f"Failed to initialize server '{name}'") from e

    def clean_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return schema
        schema.pop("title", None)
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, sub in props.items():
                props[key] = self.clean_schema(sub)
        return schema

    def format_tool(self, info) -> Tool:

        fn_decl = FunctionDeclaration(
            name=info["name"],
            description=info["description"],
            parameters=self.clean_schema(json.loads(json.dumps(info["input_schema"])))
        )
        return Tool(function_declarations=[fn_decl])

    async def connect_llm(self, config: dict):
        self.logger.info("Connecting to LLM")
        self.model_name = config.get("model_name")
        api_key = config.get("api_key")

        if not api_key or not self.model_name:
            self.logger.error("Missing LLM API key or model name")
            raise MCPClientError("❌ Missing LLM configuration")

        try:
            self.llm_client = genai.Client(api_key=api_key)
            await self.generate_content([types.Content(role="user", parts=[types.Part.from_text(text="Ping")])])
            self.logger.info("Connected to LLM successfully")
            return "Connected to LLM successfully"
        except Exception as e:
            self.llm_client = None
            self.logger.error("Failed to connect to LLM", exc_info=True)
            raise MCPClientError(f"Failed to connect to LLM: {str(e)}")

    async def generate_content(self, contents: List[types.Content]):
        if not self.llm_client or not self.model_name:
            self.logger.error("LLM client or model name not initialized")
            raise RuntimeError("LLM not initialized")

        try:
            return self.llm_client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(tools=self.formatted_tools) if self.formatted_tools else None
            )
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            raise MCPClientError(f"LLM generation failed: {str(e)}")

    async def invoke_tool(self, name: str, args: Any, retries: int = 2, delay: float = 1.0):
        def _get_session(tool_name: str):
            for server in self.servers.values():
                for tool in server["tools"]:
                    if tool_name in tool["name"]:
                        return server["session"]
            return None
        
        session = _get_session(name)
        if not session:
            self.logger.error(f"Tool '{name}' not available")
            raise MCPClientError(f"Tool '{name}' not available")

        for attempt in range(retries):
            try:
                result = await session.call_tool(name, args)

                return {"result": result.content}
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for tool '{name}': {e}", exc_info=True)
                if attempt + 1 < retries:
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"Tool '{name}' failed after {retries} attempts")
                    return {"error": f"Tool '{name}' failed after {retries} attempts with error: {str(e)}"}

    async def process_user_query(self, query:str, max_tool_calls:int = 10):
        def mentions_tool_call(resp_content):
            """Check if any part in the response includes a function_call."""
            return any(getattr(part, "function_call", None) for part in resp_content.parts)

        system_prompt = """You are a smart and helpful assistant with access to specific tools.

Your tasks:

1. When the user asks a question:
   - Determine whether the query can be answered directly or requires tool use.
   - If tools are needed, **create a workflow** using ONLY the available tools.

2. Tool usage instructions:
   - Design a **step-by-step plan** to answer the question using tools.
   - On the **first response**, provide a short explanation AND immediately perform the **first tool call** using the `function_call` API.
   - Do NOT delay execution — call the first tool in the same turn.
   - After a tool is executed and its response is available, **analyze it** and **perform the next tool call** (if needed), step by step.
   - Continue until all tool steps are completed.

3. Tool response handling:
   - After receiving tool output, use it to inform the next step.
   - Use `function_call` again for follow-up tool steps.
    - Focus only on the most relevant insights.
   - Avoid simply echoing the tool’s raw data.
   - When all tool steps are complete, summarize the final answer naturally and concisely.

4. Presentation:
   - If the final response contains structured results (like lists, steps, calculations, comparisons), format it cleanly using **Markdown**.
   - Use headings, bullet points, or code blocks when helpful.
   - The goal is to make your response easier to read and understand.
   
  Rules:
- Use `function_call` for each tool invocation. Do not describe the plan only in text.
- Never simulate tool output or perform math in your head.
- Do not invent tools or assume behavior not defined.
- Only respond with plain text if **no tools are needed** at all.

Your job is to reason, plan, and act — one tool call per turn — until the solution is complete.
"""
        self.logger.info(f"--------"*5)
        self.logger.info(f"Processing query: {query}")
        system_part = types.Content(role="model", parts=[types.Part.from_text(text=system_prompt)])
        user_part = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        contents = [system_part, user_part]
        call_num = 0

        while call_num < max_tool_calls:
            self.logger.info(f"Iteration {call_num + 1}")
            
            try:
                response = await self.generate_content(contents)

            except Exception as e:
                self.logger.errot(f"Error generating content: {e}")
                raise MCPClientError(f"Error generating content: {e}")
            
            resp_content = response.candidates[0].content
            contents.append(resp_content)

            if mentions_tool_call(resp_content):
                for part in resp_content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_name = part.function_call.name
                        tool_args = part.function_call.args
                        self.logger.info(f"Calling tool `{tool_name}` with args: {tool_args}")
                        try:
                            tool_result = await self.invoke_tool(tool_name, tool_args)
                            self.logger.info(f"Tool `{tool_name}` result: {tool_result}")
                        except Exception as e:
                            self.logger.errot(f"Tool `{tool_name}` failed: {e}")
                            raise MCPClientError(f"Tool `{tool_name}` failed: {e}")

                        tool_response_part = types.Part.from_function_response(name=tool_name, response=tool_result)
                        tool_response_content = types.Content(role="tool", parts=[tool_response_part])
                        contents.append(tool_response_content)
                call_num += 1
            else:
                final_answer = response.text
                self.logger.info(f"Final Answer:{final_answer}\n\n")
                return final_answer
        else:
            self.logger.info("Reached maximum tool call limit.")
            return f"Reached maximum tool call limit {max_tool_calls}. Increase the Limit"
    
    async def shutdown(self):
        async with self.cleanup_lock:
            self.logger.info("Shutting down MCPClient")
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}", exc_info=True)
            finally:
                self.servers.clear()
                self.formatted_tools.clear()

async def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
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
                    # client.logger.info("\n" + answer + "\n")
                except MCPClientError as e:
                    print(f"\n[Error] {e}\n")
        finally:
            await client.shutdown()

    except MCPClientError as e:
        client.logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

