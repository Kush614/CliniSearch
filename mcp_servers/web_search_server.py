import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from duckduckgo_search import DDGS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web Search MCP Server",
    description="An MCP-compliant server for performing web searches.",
    version="0.1.0",
)

class MCPQuery(BaseModel):
    query: str

class MCPResultItem(BaseModel):
    title: str
    snippet: str
    url: str

class MCPResponse(BaseModel):
    source: str
    status: str
    results: list[MCPResultItem] | None = None
    error_message: str | None = None

@app.post("/execute", response_model=MCPResponse)
async def execute_web_search(mcp_query: MCPQuery):
    query = mcp_query.query
    if not query:
        logger.warning("Web Search Server: Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        logger.info(f"Web Search Server: Received query: '{query}'")
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=5, region='wt-wt', safesearch='moderate'))

        formatted_results = []
        for res in search_results:
            formatted_results.append(
                MCPResultItem(
                    title=res.get("title", "N/A"),
                    snippet=res.get("body", "N/A"),
                    url=res.get("href", "N/A"),
                )
            )
        
        logger.info(f"Web Search Server: Found {len(formatted_results)} results for query '{query}'.")
        return MCPResponse(
            source="web_search_mcp_server", # Consistent source name
            status="success",
            results=formatted_results,
        )
    except Exception as e:
        logger.error(f"Web Search Server: Error processing query '{query}': {str(e)}", exc_info=True)
        return MCPResponse(
            source="web_search_mcp_server", status="error", error_message=str(e)
        )

if __name__ == "__main__":
    logger.info("Starting Web Search MCP Server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)