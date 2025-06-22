Generated markdown
# Spectra AI: A Multimodal Medical Research & Radiology Assistant

**CliniSearch** is an advanced, multimodal AI agent developed for the UCB Hackathon. It is designed to be a powerful assistant for medical professionals, particularly radiologists, by streamlining clinical research and providing AI-powered preliminary image analysis. The agent integrates multiple state-of-the-art technologies, including premium LLMs (Google Gemini), a multi-source RAG pipeline, and a user-friendly web interface built with Streamlit.

---

## 🚀 Application & Real-World Benefit

In the fast-paced medical field, professionals face the dual challenges of information overload and time scarcity. Spectra AI is designed to address these critical issues directly.

*   **For Clinicians & Researchers:** It acts as an intelligent research assistant, capable of querying real-time web data, peer-reviewed PubMed articles, and user-uploaded documents (like research papers or reports). By providing synthesized, source-cited answers from these distinct domains, it dramatically accelerates literature reviews, deepens contextual understanding, and helps identify research gaps.
*   **For Radiologists:** The "Radiology Image Analysis" tab offers a cutting-edge tool for decision support. A radiologist can upload a medical image (e.g., an X-ray, CT scan) and receive a preliminary analysis from a multimodal AI. By combining this visual analysis with contextual information from uploaded patient reports, Spectra AI can help identify potential abnormalities, suggest differential diagnoses, and reduce cognitive load, acting as a "second pair of eyes" to enhance diagnostic confidence and efficiency.

---

## ✨ Features

*   **User-Friendly Web Interface:** A clean, tabbed UI built with Streamlit separates the text-based RAG Q&A from the specialized Radiology Image Analysis tool.
*   **Multi-Source RAG Pipeline:**
    *   Dynamically queries **Web Search** (via DuckDuckGo) and **PubMed** (via NCBI Entrez) for real-time information.
    *   Allows users to upload their own **PDF documents**, creating a private, searchable knowledge base for highly contextualized answers.
*   **Multimodal Radiology Analysis:**
    *   Leverages **Google Gemini Pro Vision** to analyze uploaded medical images.
    *   **Context-Aware Analysis:** Uniquely combines image analysis with a RAG search of user-uploaded PDFs (e.g., patient reports), providing a holistic preliminary assessment.
*   **High-Quality AI Models:** Powered by the **Google Gemini API** (`gemini-1.5-flash`) for state-of-the-art text synthesis and multimodal understanding.
*   **Structured & Sourced Outputs:** All answers are presented in a clean, readable format with clearly listed sources and clickable links, ensuring transparency and enabling further verification.
*   **GCP-Ready Architecture:** The modular design (MCP servers, API clients, RAG processing) is built to be scalable and easily deployable on Google Cloud Platform services like Cloud Run, Vertex AI, and Cloud Storage.

---

## 🛠️ Technical Complexity & Design

CliniSearch demonstrates a strong command of modern AI engineering principles and technologies.

*   **Asynchronous Architecture:** Utilizes `asyncio` and `httpx` for efficient, non-blocking calls to the backend MCP tool servers.
*   **Advanced RAG Implementation:** The system implements a full RAG pipeline, including:
    *   **Document Parsing:** Using `PyMuPDF` to extract text from PDFs.
    *   **Text Chunking & Embedding:** Using `sentence-transformers` for local text embedding.
    *   **Vector Search:** Using `faiss-cpu` to create an efficient, in-memory vector store that simulates the functionality of a production service like GCP Vertex AI Vector Search.
*   **Sophisticated Prompt Engineering:** Prompts are dynamically constructed to be context-aware, instructing the LLM to use only the provided information and to cite its sources.
*   **Multimodal Fusion:** The radiology tool showcases a complex workflow where insights from a text-based RAG search (on PDFs) are fused into the prompt for a visual analysis task, demonstrating a true multimodal approach.
*   **Modular Codebase:** The project is well-organized into a Streamlit frontend (`app.py`), backend API clients (`utils/api_clients.py`), RAG logic (`utils/rag_processing.py`), and tool servers (`mcp_servers/`), promoting maintainability and scalability.

---
Detailed System Architecture & Data Flow
The diagram above provides a comprehensive overview of the Spectra AI technology stack and the flow of data from user interaction to final output. The system is divided into four logical domains: Frontend, Backend/Orchestrator, Local Tools & Services, and External APIs.
1. Frontend (UI - Blue)
<i class='fab fa-streamlit'></i> Streamlit App (app.py): This is the user's single point of interaction. It's responsible for rendering the web interface, managing user inputs (text queries, file uploads), and displaying the final, formatted results.
2. Application Backend / Orchestrator (Purple)
RAG & Multimodal Logic: This is the "brain" of the application, also residing within app.py. It orchestrates the entire workflow, deciding which tools to call, when to process data, which LLMs to query for synthesis, and how to format the final response.
3. Local Tools & Services (Green)
This domain contains components that run locally alongside the main application.
<i class='fas fa-server'></i> MCP Tool Servers (FastAPI):
These are two independent, lightweight servers built with FastAPI. They act as modular tools that the main orchestrator can call.
Web Search Server: Receives a query, uses the duckduckgo-search library to get results from the internet, and returns them in a standard JSON format.
PubMed Server: Receives a query, uses the BioPython library to interact with the NCBI Entrez API, and returns formatted PubMed abstracts.
<i class='fas fa-database'></i> Local RAG Pipeline Components: This sub-domain handles the processing of user-uploaded documents.
(Step 3a) PDF Parser (PyMuPDF): When a user uploads a PDF, this library extracts the raw text.
(Step 3b) Embedding Model (Sentence Transformer): This is a crucial local LLM. It takes the text chunks from the PDF and converts them into numerical vector embeddings. We use a lightweight but effective model like all-MiniLM-L6-v2.
(Step 3c) Vector Store (FAISS CPU): The generated embeddings are stored in this in-memory vector database. FAISS (Facebook AI Similarity Search) allows for incredibly fast and efficient semantic searches, simulating the functionality of a production service like GCP Vertex AI Vector Search.
4. External APIs & Data Sources (Orange)
This domain represents all the third-party services the system relies on.
<i class='fab fa-google'></i> Google Gemini API: The primary engine for high-level reasoning. It's used for:
Text Synthesis: Generating the final, human-readable answers based on the context provided by the RAG pipeline.
Vision Analysis: Analyzing the content of uploaded medical images.
<i class='fas fa-brain'></i> Anthropic Claude API (Optional): The system is built to be model-agnostic. The Claude API can be used as an alternative to Gemini for text synthesis, allowing for flexibility and comparison.
<i class='fab fa-duckduckgo'></i> DuckDuckGo Search: The data source for real-time web search.
<i class='fas fa-book-medical'></i> NCBI PubMed: The data source for peer-reviewed medical literature.
A Typical Workflow (Tracing the Arrows)
A User asks a question or uploads an image and a PDF to the Streamlit Frontend.
The frontend triggers the RAG & Multimodal Logic in the backend.
If a PDF was uploaded, it is parsed by PyMuPDF, its text is converted to embeddings by the Sentence Transformer model, and these embeddings are stored in the FAISS Vector Store.
The orchestrator sends the user's query to the MCP Servers for Web and PubMed results and simultaneously performs a semantic search on the FAISS Vector Store.
The MCP servers in turn query their respective data sources, DuckDuckGo and PubMed.
The orchestrator gathers all the retrieved context (from web, PubMed, and local PDFs) and constructs a detailed prompt. This prompt, along with the user's original question (and an image, if applicable), is sent via API call to a powerful external LLM like Google Gemini.
The orchestrator receives the synthesized answer from the LLM.
The answer is formatted neatly (with sources and links) and displayed back to the User in the Streamlit interface.

### System Workflow Diagram

The diagram below illustrates the flow of data and logic within Spectra AI. It shows how a user's query is processed through retrieval from multiple sources (Web, PubMed, local vector store) and finally, synthesis by powerful LLMs.

![Untitled design](https://github.com/user-attachments/assets/609768ea-0530-4756-985c-71a150dc8f6a)


Explanation of the Workflow:

User Interaction: The user interacts with the Streamlit App, either by asking a text question in the "Medical RAG Q&A" tab or by uploading an image and text prompt in the "Radiology Image Analysis" tab.

RAG Orchestration: The app's RAG Orchestrator processes the request.

Information Retrieval (Parallel):

It sends queries to the Web Search and PubMed MCP Servers for real-time data.

Simultaneously, it performs a semantic search against the local Vector Store (FAISS), which contains the indexed content of any uploaded PDFs.

Context Synthesis:

For RAG Q&A: The formatted context from each enabled source (Web, PubMed, PDF) is sent separately to the LLM Synthesizer (Gemini). This ensures the answers are based on distinct evidence trails.

For Radiology Analysis: The Multimodal LLM (Gemini) receives the uploaded image, the user's text prompt, and context retrieved from the PDF vector store.

Final Output: The synthesized text answers and image analyses are formatted and displayed in the Streamlit UI, with sources clearly cited.

⚙️ Setup & Running the System

1. Prerequisites:

Python 3.9+

Git

2. Clone the Repository:

Generated bash
git clone <your-repo-url>
cd <your-project-directory>
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

3. Set up Virtual Environment:

Generated bash
# Create the environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

4. Install Dependencies:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

5. Configure API Keys:

Rename .env.example to .env.

Open .env and add your GOOGLE_API_KEY and your email for NCBI_EMAIL.

6. Run the MCP Servers:

You need two separate terminal windows for these background services. Ensure your virtual environment is active in each.

Terminal 1 (Web Search Server):

Generated bash
python -m uvicorn mcp_servers.web_search_server:app --reload --port 8001
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Terminal 2 (PubMed Server):

Generated bash
python -m uvicorn mcp_servers.pubmed_search_server:app --reload --port 8002
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

7. Run the Main Web Application:

In a third terminal (with the virtual environment active), run the Streamlit app:

Generated bash
streamlit run app.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Your browser should open with the Spectra AI application running.

⚖️ Ethical Considerations

This tool is a powerful assistant but is not a substitute for professional medical judgment.

For Educational & Research Use Only: All outputs, especially image analyses, must be verified by a qualified medical professional.

Data Privacy: Users must ensure any uploaded documents or images are properly anonymized and comply with HIPAA and other privacy regulations.

AI Bias: The underlying LLMs can have biases. Results should be critically evaluated.

