# utils/rag_processing.py

import httpx
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import fitz  # PyMuPDF
from typing import List, Dict, Tuple

# --- Configuration (remains the same) ---
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
WEB_SEARCH_MCP_URL = "http://localhost:8001/execute"
PUBMED_MCP_URL = "http://localhost:8002/execute"


# --- Tool Interaction (remains the same) ---
async def query_mcp_server(url: str, query: str) -> Dict:
    """Queries an MCP server asynchronously."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json={"query": query}, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            print(f"Error querying MCP server at {url}: {e}")
            return {"status": "error", "error_message": f"Connection Error: {e}. Is the server running?"}
        except Exception as e:
            print(f"Unhandled error querying MCP server at {url}: {e}")
            return {"status": "error", "error_message": str(e)}

# --- PDF Parsing, Chunking, VectorStore (all remain the same) ---
def chunk_text(text: str, chunk_size=512, chunk_overlap=50) -> List[str]:
    if not text: return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def parse_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    document_chunks = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = ""
        for page_num, page in enumerate(doc):
            full_text += page.get_text() + "\n"
        text_chunks = chunk_text(full_text)
        for i, chunk in enumerate(text_chunks):
            document_chunks.append({
                "source": f"PDF: {filename}", "content": chunk,
                "metadata": {"page_num_approx": 1 + (i * (512 - 50) // 400)}
            })
        return document_chunks
    except Exception as e:
        print(f"Error parsing PDF '{filename}': {e}")
        return []

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents = []
    def add_documents(self, docs: List[Dict]):
        contents = [doc['content'] for doc in docs]
        if not contents: return
        embeddings = EMBEDDING_MODEL.encode(contents, convert_to_tensor=False)
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(docs)
    def search(self, query: str, k=3) -> List[Dict]:
        if self.index.ntotal == 0: return []
        query_embedding = EMBEDDING_MODEL.encode([query], convert_to_tensor=False)
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k=min(k, self.index.ntotal))
        return [self.documents[i] for i in indices[0] if i != -1]


# --- Main RAG Orchestration (MODIFIED AND CORRECTED) ---
async def perform_rag(query: str, vector_store: VectorStore, use_web=True, use_pubmed=True) -> Tuple[str, List[Dict]]:
    """
    Orchestrates the RAG process for specified sources and returns both the
    formatted context string and a structured list of source documents.
    """
    print(f"Performing RAG for query: '{query}' with flags Web={use_web}, PubMed={use_pubmed}")
    
    context = ""
    sources = []

    # --- Step 1: Query external tools if requested ---
    tasks = []
    if use_web:
        tasks.append(query_mcp_server(WEB_SEARCH_MCP_URL, query))
    if use_pubmed:
        tasks.append(query_mcp_server(PUBMED_MCP_URL, query))

    if tasks:
        tool_results = await asyncio.gather(*tasks)
        # Consolidate and format context and sources from external tools
        for res in tool_results:
            if res and res.get('status') == 'success' and res.get('results'):
                is_pubmed = "pubmed" in res.get('source', '').lower()
                source_name = "PubMed" if is_pubmed else "Web Search"
                
                context += f"--- Context from {source_name} ---\n"
                
                for item in res['results'][:3]: # Take top 3 results
                    title = item.get('title', 'N/A')
                    snippet = item.get('snippet', 'N/A')
                    
                    context += f"Title: {title}\nSnippet: {snippet}\n\n"
                    
                    if is_pubmed:
                        pmid = item.get('id', '')
                        sources.append({
                            "type": "PubMed", "title": title,
                            "link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "#"
                        })
                    else:
                        sources.append({
                            "type": "Web Search", "title": title,
                            "link": item.get('url', '#')
                        })

    # --- Step 2: Query internal vector store if no external tools were used ---
    # This logic assumes that if a user turns off both Web and PubMed, they *only* want to
    # query the documents they have uploaded.
    if not use_web and not use_pubmed:
        print("Querying uploaded documents only...")
        semantic_results = vector_store.search(query, k=5) # Get more results if it's the only source
        if semantic_results:
            context += "--- Context from Uploaded Documents ---\n"
            for res in semantic_results:
                context += f"Source: {res.get('source', 'Uploaded Document')} (Page ~{res.get('metadata', {}).get('page_num_approx', 'N/A')}) - Content: {res.get('content', 'N/A')}\n\n"
                sources.append({
                    "type": "PDF Document",
                    "title": res.get('source', 'Uploaded Document'),
                    "link": f"Page ~{res.get('metadata', {}).get('page_num_approx', 'N/A')}"
                })
        else:
            context += "No relevant information found in uploaded documents.\n"

    # If both external tools and the vector store were meant to be queried together,
    # the logic in app.py would need to be different, but for separate answers, this is cleaner.

    print("RAG process completed. Context and sources built.")
    return context.strip(), sources