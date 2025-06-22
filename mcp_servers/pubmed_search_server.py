import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from Bio import Entrez
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="PubMed Search MCP Server",
    description="An MCP-compliant server for performing PubMed searches.",
    version="0.1.0",
)

NCBI_EMAIL = os.getenv("NCBI_EMAIL")
if not NCBI_EMAIL:
    logger.warning("NCBI_EMAIL not set in .env file. PubMed queries might be throttled or blocked.")
Entrez.email = NCBI_EMAIL

class MCPQuery(BaseModel):
    query: str

class PubMedResultItem(BaseModel):
    title: str
    snippet: str
    id: str
    url: str

class MCPResponse(BaseModel):
    source: str
    status: str
    results: list[PubMedResultItem] | None = None
    error_message: str | None = None

MAX_RESULTS_PUBMED = 3

@app.post("/execute", response_model=MCPResponse)
async def execute_pubmed_search(mcp_query: MCPQuery):
    query = mcp_query.query
    if not query:
        logger.warning("PubMed Server: Received empty query.")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if not Entrez.email:
        logger.warning("PubMed Server: NCBI_EMAIL not configured. Proceeding but this is not recommended.")

    try:
        logger.info(f"PubMed Server: Received query: '{query}'")
        
        handle_search = Entrez.esearch(db="pubmed", term=query, retmax=str(MAX_RESULTS_PUBMED), sort="relevance")
        search_record = Entrez.read(handle_search)
        handle_search.close()
        id_list = search_record["IdList"]

        if not id_list:
            logger.info(f"PubMed Server: No results found for '{query}'.")
            return MCPResponse(source="pubmed_mcp_server", status="success", results=[]) # Consistent source name

        formatted_results = []
        for pmid in id_list:
            try:
                handle_fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="medline", retmode="text")
                article_text = handle_fetch.read()
                handle_fetch.close()

                title = "N/A"
                abstract = "Abstract not found."
                current_title_lines = []
                current_abstract_lines = []
                in_title = False
                in_abstract = False

                for line in article_text.splitlines():
                    if line.startswith("TI  - "):
                        current_title_lines.append(line[6:])
                        in_title = True; in_abstract = False
                    elif line.startswith("AB  - "):
                        current_abstract_lines.append(line[6:])
                        in_abstract = True; in_title = False
                    elif line.startswith("    ") and in_title:
                        current_title_lines.append(line[4:])
                    elif line.startswith("    ") and in_abstract:
                        current_abstract_lines.append(line[4:])
                    elif not line.startswith(" "):
                        in_title = False; in_abstract = False
                
                if current_title_lines: title = " ".join(current_title_lines)
                if current_abstract_lines: abstract = " ".join(current_abstract_lines)
                
                snippet = abstract[:500] + ("..." if len(abstract) > 500 else "")

                formatted_results.append(PubMedResultItem(
                    title=title, snippet=snippet, id=pmid, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))
            except Exception as e_article:
                logger.error(f"PubMed Server: Error processing article {pmid} for query '{query}': {str(e_article)}", exc_info=True)
                formatted_results.append(PubMedResultItem(
                    title=f"Error fetching article {pmid}", snippet=str(e_article), id=pmid, url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))

        logger.info(f"PubMed Server: Processed {len(formatted_results)} results for query '{query}'.")
        return MCPResponse(source="pubmed_mcp_server", status="success", results=formatted_results)
    except Exception as e:
        logger.error(f"PubMed Server: Error processing query '{query}': {str(e)}", exc_info=True)
        return MCPResponse(source="pubmed_mcp_server", status="error", error_message=str(e))

if __name__ == "__main__":
    logger.info("Starting PubMed MCP Server on http://localhost:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)