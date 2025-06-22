# app.py (Final version with UI fix and combined Radiology RAG)

import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
from PIL import Image

# Import our utility modules
from utils.api_clients import gemini_client
from utils.rag_processing import VectorStore, perform_rag, parse_pdf, EMBEDDING_MODEL

# Load environment variables from .env file
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="IntraIntel AI Agent for Radiology & Research",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for Output Boxes ---
# This CSS creates a bordered box that allows text to wrap naturally.
st.markdown("""
<style>
    .output-box {
        border: 1px solid #444;
        border-radius: 5px;
        padding: 10px;
        background-color: #1a1a1a; /* A slightly different shade for the box */
        margin-bottom: 20px; /* Add space between boxes */
    }
    .output-box p {
        margin-bottom: 5px; /* Adjust paragraph spacing inside the box */
    }
    .output-box a {
        color: #1c83e1; /* Make links stand out */
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State ---
if 'vector_store' not in st.session_state:
    embedding_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    st.session_state.vector_store = VectorStore(dimension=embedding_dim)

if "messages" not in st.session_state:
    st.session_state.messages = []

# This state is specific to the radiology tab's uploader
if 'radiology_vector_store' not in st.session_state:
    embedding_dim = EMBEDDING_MODEL.get_sentence_embedding_dimension()
    st.session_state.radiology_vector_store = VectorStore(dimension=embedding_dim)


# --- Helper Functions ---
def handle_pdf_upload(uploaded_files, vector_store_key):
    """Processes uploaded PDF files and adds them to the specified vector store."""
    if uploaded_files:
        with st.spinner("Processing uploaded PDFs..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                bytes_data = uploaded_file.read()
                chunks = parse_pdf(bytes_data, uploaded_file.name)
                if chunks:
                    all_chunks.extend(chunks)
            if all_chunks:
                st.session_state[vector_store_key].add_documents(all_chunks)
                st.success(f"Processed and indexed {len(uploaded_files)} PDF(s).")
            else:
                st.error("Could not extract text from the uploaded PDF(s).")

def format_output_as_html(source_name: str, answer: str, sources: list) -> str:
    """
    Creates an HTML string with a custom-styled box for the output.
    """
    # Start the styled container
    output_html = '<div class="output-box">'
    output_html += f"<p><strong>✅ Answer based on {source_name}:</strong></p>"
    output_html += "<hr style='border-color: #444;'>"
    # Replace newlines in the answer with <br> for HTML display
    output_html += f"<p>{answer.replace(chr(10), '<br>')}</p>"
    output_html += "<hr style='border-color: #444;'>"
    
    if sources:
        output_html += f"<p><strong>Sources from {source_name}:</strong></p>"
        for i, source in enumerate(sources):
            title = source.get('title', 'N/A')
            link = source.get('link', '#')
            
            if source.get('type') == "PubMed":
                pmid = link.split('/')[-2] if link.endswith('/') else link.split('/')[-1]
                output_html += f"<p>{i+1}. {title}<br>   PubMed ID: {pmid} (<a href='{link}' target='_blank'>Link</a>)</p>"
            elif source.get('type') == "Web Search":
                output_html += f"<p>{i+1}. {title}<br>   <a href='{link}' target='_blank'>Link</a></p>"
            else: # For PDF
                output_html += f"<p>{i+1}. {title} ({link})</p>"
    else:
        output_html += f"<p>No sources were retrieved from {source_name}.</p>"
    
    output_html += '</div>'
    return output_html


# --- UI Layout ---
st.title("🧠  CliniSearch AI Agent")
st.caption("A multimodal assistant for medical research and radiological image analysis.")

# --- Main Content Tabs ---
tab1, tab2 = st.tabs(["Medical RAG Q&A", "Radiology Image Analysis"])

# --- TAB 1: Medical RAG Q&A ---
with tab1:
    col1, col2 = st.columns([3, 1]) # Main content area and a sidebar-like column
    
    with col2:
        st.subheader("Controls & Tools")
        st.markdown("---")
        use_web_search = st.toggle("Enable Web Search", value=True, help="Include real-time web search results.")
        use_pubmed = st.toggle("Enable PubMed", value=True, help="Include PubMed article abstracts.")
        use_uploaded_docs_tab1 = st.toggle("Enable Uploaded Docs", value=True, help="Include your uploaded PDFs.")
        
        st.markdown("---")
        st.subheader("Upload Documents for RAG")
        st.info("Upload research papers or reports (PDFs) to include them in your questions.")
        
        uploaded_files_tab1 = st.file_uploader("Upload PDFs for Q&A", type="pdf", accept_multiple_files=True, key="pdf_uploader_tab1")
        if uploaded_files_tab1:
            if 'processed_files_tab1' not in st.session_state or st.session_state.processed_files_tab1 != [f.name for f in uploaded_files_tab1]:
                handle_pdf_upload(uploaded_files_tab1, 'vector_store')
                st.session_state.processed_files_tab1 = [f.name for f in uploaded_files_tab1]

    with col1:
        st.header("Medical Research & Question Answering")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"], unsafe_allow_html=True)

        if prompt := st.chat_input("Ask a medical research question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                final_outputs = []
                if use_web_search:
                    with st.spinner("Searching the web and synthesizing..."):
                        web_context, web_sources = asyncio.run(perform_rag(prompt, st.session_state.vector_store, use_web=True, use_pubmed=False))
                        if web_context:
                            web_synthesis_prompt = f"Based ONLY on Web Search context, answer: {prompt}"
                            web_answer = gemini_client.generate_text(web_synthesis_prompt)
                            final_outputs.append(format_output_as_html("Web Search", web_answer, web_sources))
                
                if use_pubmed:
                    with st.spinner("Searching PubMed and synthesizing..."):
                        pubmed_context, pubmed_sources = asyncio.run(perform_rag(prompt, st.session_state.vector_store, use_web=False, use_pubmed=True))
                        if pubmed_context:
                            pubmed_synthesis_prompt = f"Based ONLY on PubMed context, answer: {prompt}"
                            pubmed_answer = gemini_client.generate_text(pubmed_synthesis_prompt)
                            final_outputs.append(format_output_as_html("PubMed Search", pubmed_answer, pubmed_sources))

                if use_uploaded_docs_tab1:
                    with st.spinner("Searching uploaded documents and synthesizing..."):
                        doc_context, doc_sources = asyncio.run(perform_rag(prompt, st.session_state.vector_store, use_web=False, use_pubmed=False))
                        if doc_context and "No relevant information" not in doc_context:
                            doc_synthesis_prompt = f"Based ONLY on uploaded document context, answer: {prompt}"
                            doc_answer = gemini_client.generate_text(doc_synthesis_prompt)
                            final_outputs.append(format_output_as_html("Uploaded Documents", doc_answer, doc_sources))

                if not final_outputs:
                    st.warning("Please enable at least one data source to get an answer.")
                else:
                    full_response = "".join(final_outputs)
                    st.markdown(full_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- TAB 2: Radiology Image Analysis ---
with tab2:
    st.header("Radiology Image Analysis")
    st.warning("⚠️ This tool is for research and educational purposes only. It is NOT a substitute for professional medical advice or diagnosis.")
    
    col_rad1, col_rad2 = st.columns([1, 2])
    
    with col_rad1:
        st.subheader("Upload Image & Context")
        uploaded_image = st.file_uploader("1. Upload a Medical Image (JPG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")
        
        st.markdown("---")
        st.info("Optionally, upload a relevant PDF (e.g., patient report, prior study) to provide context for the image analysis.")
        uploaded_files_tab2 = st.file_uploader("2. Upload Context PDF(s)", type="pdf", accept_multiple_files=True, key="pdf_uploader_tab2")
        if uploaded_files_tab2:
            if 'processed_files_tab2' not in st.session_state or st.session_state.processed_files_tab2 != [f.name for f in uploaded_files_tab2]:
                handle_pdf_upload(uploaded_files_tab2, 'radiology_vector_store')
                st.session_state.processed_files_tab2 = [f.name for f in uploaded_files_tab2]

    with col_rad2:
        st.subheader("Analysis Prompt & Results")
        image_analysis_prompt_template = "As a helpful radiology assistant, analyze this image. Describe key findings, structures, and potential abnormalities. List possible differential diagnoses based on the visual evidence. {context_section}"
        
        if st.session_state.radiology_vector_store.index.ntotal > 0:
            st.success("Context from uploaded PDF(s) is ready to be used in the analysis.")
        
        image_analysis_prompt = st.text_area(
            "3. Enter or modify your prompt for the analysis:", 
            image_analysis_prompt_template.format(context_section="Additionally, consider the following context from the provided documents when forming your analysis."),
            height=150,
            key="image_prompt"
        )
        
        if uploaded_image and image_analysis_prompt:
            st.image(uploaded_image, caption="Image to be analyzed", width=300)
            
            if st.button("Analyze Image with Context", use_container_width=True, type="primary"):
                with st.spinner("Searching documents & analyzing image with Gemini..."):
                    # 1. Get context from the radiology vector store based on the prompt
                    context_query = f"Context relevant to: {image_analysis_prompt}"
                    context, sources = asyncio.run(perform_rag(
                        context_query, 
                        st.session_state.radiology_vector_store, 
                        use_web=False, use_pubmed=False
                    ))

                    # 2. Build final prompt for Gemini Vision
                    final_vision_prompt = f"""{image_analysis_prompt}

                    --- Provided Context from Documents ---
                    {context if context else "No additional context provided."}
                    --- End of Context ---
                    """
                    
                    # 3. Call Gemini Vision API
                    image_bytes = uploaded_image.getvalue()
                    analysis_result = gemini_client.analyze_image(final_vision_prompt, image_bytes)
                    
                    st.subheader("Analysis Results")
                    st.markdown(analysis_result)

                    # 4. Display the sources from the PDF that were used as context
                    if sources:
                        with st.expander("View Context Sources from PDF(s)"):
                            for source in sources:
                                st.markdown(f"**{source['title']}** ({source['link']})")