
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

from google import genai
from google.genai import types
from google.genai.types import Tool, FunctionDeclaration


warnings.filterwarnings("ignore")

logger = logging.getLogger("mcp")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("mcp-client.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

class MCPClientError(Exception):
    pass

class MCPClient:

    def __init__(self) -> None:
        load_dotenv()
        logger.info("Initializing MCPClient")
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self.cleanup_lock: asyncio.Lock = asyncio.Lock()

        self.available_servers: List[ClientSession] = []
        self.server_map: Dict[str, int] = {}
        self.available_tools: List[Dict[str, Any]] = []
        self.formatted_tools: List[Tool] = []

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise MCPClientError("GEMINI_API_KEY environment variable is missing")
        # Use the latest flash model
        self.model_name: str = "gemini-2.0-flash-001"
        self.genai_client = genai.Client(api_key=api_key)

    @staticmethod
    def load_json_file(path: str) -> Dict[str, Any]:

        if not path.lower().endswith(".json"):
            raise MCPClientError(f"Expected a .json file for server_config, got: {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise MCPClientError(f"Server configuration file not found: {path}") from e
        except json.JSONDecodeError as e:
            raise MCPClientError(f"Invalid JSON in server configuration: {path}") from e

    async def initialize_servers(self, config_path: str) -> None:

        try:
            raw = self.load_json_file(config_path)
            servers_config = raw.get("mcpServers")
            if not isinstance(servers_config, dict):
                raise MCPClientError("Field 'mcpServers' missing or not a dict in config file")
        except MCPClientError:
            raise

        for server_name, srv_conf in servers_config.items():
            logger.info(f"Launching MCP server '{server_name}'")
            cmd = srv_conf.get("command")
            if not isinstance(cmd, str):
                raise MCPClientError(f"Invalid command for server '{server_name}'")

            binary = shutil.which(cmd) or cmd
            args = srv_conf.get("args", [])
            env = srv_conf.get("env") or None

            server_params = StdioServerParameters(
                command=binary,
                args=args,
                env=env,
            )

            try:
                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                reader, writer = stdio_transport
                session = await self.exit_stack.enter_async_context(
                    ClientSession(reader, writer)
                )
                await session.initialize()
                self.available_servers.append(session)
                logger.info(f"Server '{server_name}' initialized successfully")
            except Exception as e:
            
                logger.error(f"Failed to initialize server '{server_name}': {e}", exc_info=True)
                asyncio.create_task(self.shutdown())
                raise MCPClientError(f"Could not start server '{server_name}'") from e

    async def discover_tools(self) -> None:

        if not self.available_servers:
            raise MCPClientError("No servers have been initialized yet")

        logger.info("Discovering tools from all servers")
        for idx, server in enumerate(self.available_servers):
            try:
                response = await server.list_tools()
            except Exception as e:
                logger.error(f"Error fetching tools from server #{idx}: {e}", exc_info=True)
                continue  # Skip this server but keep others running

            for tool in response.tools:
                name = tool.name
                desc = tool.description or ""
                schema = tool.inputSchema or {}

                self.available_tools.append({
                    "name": name,
                    "description": desc,
                    "input_schema": schema,
                })
                self.server_map[name] = idx
                logger.info(f"Discovered tool '{name}' on server #{idx}")

        self.formatted_tools = self.build_formatted_tool_list()

    def clean_input_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:

        if not isinstance(schema, dict):
            return schema

        schema.pop("title", None)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                properties[key] = self.clean_input_schema(subschema)
        return schema

    def build_formatted_tool_list(self) -> List[Tool]:

        formatted: List[Tool] = []
        for tool_info in self.available_tools:
            name = tool_info["name"]
            desc = tool_info["description"]
            raw_schema = tool_info["input_schema"]

            schema_copy = json.loads(json.dumps(raw_schema))
            cleaned = self.clean_input_schema(schema_copy)

            fn_decl = FunctionDeclaration(
                name=name,
                description=desc,
                parameters=cleaned,
            )
            formatted.append(Tool(functionDeclarations=[fn_decl]))
            logger.debug(f"Formatted tool schema for '{name}'")
        return formatted

    async def generate_content(self,prompt_parts):
        if not self.formatted_tools:
            raise MCPClientError("No tools have been formatted for LLM calls")

        return self.genai_client.models.generate_content(
            model=self.model_name,
            contents=prompt_parts,
            config=types.GenerateContentConfig(
                tools=self.formatted_tools
            ),
        )

    async def call_tool_with_retries(
        self,
        tool_name: str,
        tool_args: Any,
        max_retries: int = 3,
        delay: float = 1.0,
    ) -> Dict[str, Any]:

        if tool_name not in self.server_map:
            raise MCPClientError(f"Tool '{tool_name}' is not known or not discovered")

        server_idx = self.server_map[tool_name]
        session = self.available_servers[server_idx]

        attempt = 0
        while attempt < max_retries:
            try:
                logger.info(f"Invoking tool '{tool_name}' (attempt {attempt+1}) on server #{server_idx}")
                tool_response = await session.call_tool(tool_name, tool_args)
                logger.info(f"Tool '{tool_name}' Responce : '{tool_response.content}'")
                return {"result": tool_response.content}
            except Exception as e:
                logger.warning(
                    f"Error calling tool '{tool_name}' on server #{server_idx}: {e}",
                    exc_info=True
                )
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Max retries hit for tool '{tool_name}'")
                    return {"error": str(e)}

    async def process_query(self, query: str) -> str:

        if not query:
            raise MCPClientError("Empty query provided to process_query")

        logger.info(f"Processing user query: '{query}'")

        user_part = types.Content(role="user", parts=[types.Part.from_text(text=query)])

        llm_response = await self.generate_content([user_part])
        logger.info

        final_text = []
        for candidate in llm_response.candidates:
            print(candidate)

            if candidate.content.parts:
                for part in candidate.content.parts:
                    if isinstance(part,types.Part):
                        if part.function_call:
                            tool_name = part.function_call.name
                            tool_args = part.function_call.args
                            logger.info(f"LLM requested tool '{tool_name}' with args: {tool_args}")
                            tool_responce = await self.call_tool_with_retries(tool_name, tool_args)
                            tool_responce_part = types.Part.from_function_response(name=tool_name,
                                                                                   response=tool_responce)
                            tool_responce_content = types.Content(role="tool",
                                                                  parts=[tool_responce_part])
                            
                            response = await self.generate_content([user_part,
                                                                       part,
                                                                       tool_responce_content
                                                                       ])
                            final_responce = response.candidates[0].content.parts[0].text
                            print(response.candidates)
                            print("------------------------")

                        else:
                            logger.info("LLM answered directly without a tool call")
                            final_responce = part.text
                        logger.info(f"LLM Responce : {final_responce}")
                        final_text.append(final_responce)     
        return "\n".join(final_text)
    
    async def shutdown(self) -> None:
        """
        Gracefully close all sessions and the exit stack. Safe to call multiple times.
        """
        async with self.cleanup_lock:
            logger.info("Shutting down MCPClient and cleaning up resources")
            try:
                await self.exit_stack.aclose()
            except Exception as e:
                logger.error(f"Error during shutdown: {e}", exc_info=True)
            finally:
                # Clear local references
                self.available_servers.clear()
                self.server_map.clear()
                self.available_tools.clear()
                self.formatted_tools.clear()

    async def chat_loop(self) -> None:

        print("\nMCP Client Ready! (type 'quit' to exit)\n")
        try:
            while True:
                query = input("Query: ").strip()
                if query.lower() == "quit":
                    break
                try:
                    answer = await self.process_query(query)
                    print("\n" + answer + "\n")
                    logger.info("\n" + answer + "\n")
                except MCPClientError as e:
                    print(f"\n[Error] {e}\n")
        finally:
            await self.shutdown()


async def main():
        config_path = "mcp-servers-configs.json"
        client = MCPClient()

        try:
            await client.initialize_servers(config_path)
            await client.discover_tools()
            query = "What are tools available?"
            summary = await client.process_query(query)
            # await client.chat_loop()
            
        except MCPClientError as e:
            logger.error(f"Fatal error in main: {e}", exc_info=True)
        finally:
            await client.shutdown()

if __name__ == "__main__":
    asyncio.run(main())

