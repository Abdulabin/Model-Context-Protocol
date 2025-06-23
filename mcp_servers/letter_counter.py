# import gradio as gr

# def letter_counter(word, letter):
#     """
#     Count the number of occurrences of a letter in a word or text.

#     Args:
#         word (str): The input text to search through
#         letter (str): The letter to search for

#     Returns:
#         str: A message indicating how many times the letter appears
#     """
#     word = word.lower()
#     letter = letter.lower()
#     count = word.count(letter)
#     return count

# demo = gr.Interface(
#     fn=letter_counter,
#     inputs=[gr.Textbox("strawberry"), gr.Textbox("r")],
#     outputs=[gr.Number()],
#     title="Letter Counter",
#     description="Enter text and a letter to count how many times the letter appears in the text."
# )

# if __name__ == "__main__":
#     demo.launch(mcp_server=True)

# sse_server.py
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from mcp.server import Server
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
import uvicorn

# Create the MCP server
mcp = FastMCP("SSE Example Server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet a user by name"""
    return f"Hello, {name}! Welcome to the SSE server."

@mcp.tool()
def add(a: int, b: int) -> str:
    """Add two numbers and return the result"""
    return f"The sum of {a} and {b} is {a + b}."

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """Create a Starlette application that can serve the MCP server with SSE."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

if __name__ == "__main__":
    # Get the underlying MCP server
    mcp_server = mcp._mcp_server
    
    # Create Starlette app with SSE support
    starlette_app = create_starlette_app(mcp_server, debug=True)
    
    port = 8080
    print(f"Starting MCP server with SSE transport on port {port}...")
    print(f"SSE endpoint available at: http://localhost:{port}/sse")
    
    # Run the server using uvicorn
    uvicorn.run(starlette_app, port=port)