"""
Component 4: Streamlit UI
- Reflect the airline use case
- View KG-retrieved context
- View final LLM answer
- User can write questions or select predefined ones
- Integration with RAG pipeline
- Interface remains functional after receiving answers

NOTE: If you get DLL errors on Windows, run with: python start_app.py
"""

import streamlit as st
import sys
from typing import Dict, Any, Optional
import json

# Import our components
from questions import get_all_questions, get_question_text_only
from utils import get_neo4j_connection, format_kg_results
from component_1_input_preprocessing import InputPreprocessor
from component_2_graph_retrieval import GraphRetriever, NodeEmbeddingRetriever
from component_3_llm_layer import LLMHandler


# Page configuration
st.set_page_config(
    page_title="Airline RAG System",
    page_icon="✈️",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables"""
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'graph_retriever' not in st.session_state:
        st.session_state.graph_retriever = None
    if 'embedding_retriever' not in st.session_state:
        st.session_state.embedding_retriever = None
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


def load_models():
    """Load all required models"""
    if not st.session_state.models_loaded:
        with st.spinner("Loading models... This may take a few minutes on first run."):
            try:
                # Load preprocessor
                st.session_state.preprocessor = InputPreprocessor()
                
                # Load graph retriever
                st.session_state.graph_retriever = GraphRetriever()
                
                # Load embedding retriever (default model)
                st.session_state.embedding_retriever = NodeEmbeddingRetriever(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                
                # Load LLM handler (default model)
                st.session_state.llm_handler = LLMHandler("Qwen/Qwen2.5-1.5B-Instruct")
                
                st.session_state.models_loaded = True
                st.success("✅ All models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.session_state.models_loaded = False


def process_question(question: str, 
                     use_embeddings: bool = True,
                     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                     llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Complete RAG pipeline:
    1. Preprocess input
    2. Retrieve from KG (baseline + embeddings)
    3. Generate answer with LLM
    """
    results = {
        "question": question,
        "preprocessing": None,
        "kg_results": None,
        "embedding_results": None,
        "llm_answer": None,
        "error": None
    }
    
    try:
        # Step 1: Preprocess
        with st.spinner("Preprocessing input..."):
            preprocessing = st.session_state.preprocessor.preprocess(question)
            results["preprocessing"] = preprocessing
        
        # Step 2: Retrieve from KG (Baseline)
        with st.spinner("Querying Knowledge Graph..."):
            kg_results = st.session_state.graph_retriever.get_query_by_intent(
                preprocessing["intent"],
                preprocessing["entities"]
            )
            results["kg_results"] = kg_results
        
        # Step 3: Retrieve using embeddings (if enabled)
        if use_embeddings:
            with st.spinner("Retrieving similar nodes using embeddings..."):
                # Reload embedding retriever if model changed
                if st.session_state.embedding_retriever.model_name != embedding_model:
                    st.session_state.embedding_retriever = NodeEmbeddingRetriever(
                        model_name=embedding_model
                    )
                
                embedding_results = st.session_state.embedding_retriever.retrieve_by_similarity(
                    question,
                    preprocessing["embedding"],
                    top_k=5
                )
                results["embedding_results"] = embedding_results
        
        # Step 4: Generate answer with LLM
        with st.spinner("Generating answer..."):
            # Reload LLM if model changed
            if st.session_state.llm_handler.model_name != llm_model:
                st.session_state.llm_handler = LLMHandler(llm_model)
            
            llm_result = st.session_state.llm_handler.process_query(
                question,
                kg_results,
                results.get("embedding_results")
            )
            results["llm_answer"] = llm_result
        
    except Exception as e:
        results["error"] = str(e)
        st.error(f"Error processing question: {e}")
    
    return results


def main():
    """Main Streamlit app"""
    init_session_state()
    
    # Header
    st.title("✈️ Airline Knowledge Graph RAG System")
    st.markdown("""
    This system uses a Knowledge Graph of airline data combined with Large Language Models 
    to answer questions about flights, passengers, satisfaction, and more.
    """)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("LLM Model")
        llm_model = st.selectbox(
            "Select LLM",
            [
                "Qwen/Qwen2.5-1.5B-Instruct",
                "google/gemma-2-2b-it",
                "Qwen/Qwen2.5-7B-Instruct",
            ],
            help="Choose which language model to use for generating answers"
        )
        
        st.subheader("Embedding Model")
        use_embeddings = st.checkbox("Use node embeddings", value=True)
        embedding_model = st.selectbox(
            "Select Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            disabled=not use_embeddings,
            help="Choose which embedding model to use for similarity search"
        )
        
        st.divider()
        
        # Load models button
        if st.button("🔄 Load/Reload Models", type="primary"):
            st.session_state.models_loaded = False
            load_models()
        
        if st.session_state.models_loaded:
            st.success("✅ Models ready")
        else:
            st.warning("⚠️ Click 'Load/Reload Models' to start")
    
    # Main content
    if not st.session_state.models_loaded:
        st.info("👈 Please load the models from the sidebar to begin")
        return
    
    # Question input section
    st.header("❓ Ask a Question")
    
    # Tabs for input methods
    tab1, tab2 = st.tabs(["📝 Write Your Own", "📋 Select Predefined"])
    
    with tab1:
        user_question = st.text_area(
            "Enter your question about airline data:",
            height=100,
            placeholder="e.g., Which flights have the longest delays?"
        )
        if st.button("Submit Question", key="submit_custom"):
            if user_question.strip():
                st.session_state.current_question = user_question
                st.session_state.results = process_question(
                    user_question,
                    use_embeddings,
                    embedding_model,
                    llm_model
                )
    
    with tab2:
        predefined_questions = get_all_questions()
        question_options = [f"Q{q['id']}: {q['question']}" for q in predefined_questions]
        
        selected_question = st.selectbox(
            "Choose a predefined question:",
            [""] + question_options
        )
        
        if st.button("Submit Selected Question", key="submit_predefined"):
            if selected_question:
                # Extract the question text
                question_text = selected_question.split(": ", 1)[1]
                st.session_state.current_question = question_text
                st.session_state.results = process_question(
                    question_text,
                    use_embeddings,
                    embedding_model,
                    llm_model
                )
    
    # Display results
    if st.session_state.results:
        results = st.session_state.results
        
        st.divider()
        st.header("📊 Results")
        
        # Question
        st.subheader("Question")
        st.info(results["question"])
        
        # Create columns for different sections
        col1, col2 = st.columns(2)
        
        with col1:
            # Preprocessing results
            st.subheader("1️⃣ Input Analysis")
            if results["preprocessing"]:
                with st.expander("View preprocessing details", expanded=True):
                    st.write("**Detected Intent:**", results["preprocessing"]["intent"])
                    st.write("**Extracted Entities:**")
                    st.json(results["preprocessing"]["entities"])
        
        with col2:
            # KG Results
            st.subheader("2️⃣ Knowledge Graph Context")
            if results["kg_results"]:
                with st.expander("View KG query results", expanded=True):
                    if results["kg_results"]:
                        st.dataframe(results["kg_results"])
                    else:
                        st.write("No results from Knowledge Graph")
            
            # Embedding Results
            if results["embedding_results"]:
                st.subheader("3️⃣ Similar Nodes (Embeddings)")
                with st.expander("View embedding-based retrieval"):
                    st.markdown(f"**Model:** {results['embedding_results']['model']}")
                    for node in results["embedding_results"]["similar_nodes"][:3]:
                        st.markdown(f"- {node['description']} *(similarity: {node['similarity']:.3f})*")
        
        # LLM Answer (full width)
        st.subheader("4️⃣ Final Answer")
        if results["llm_answer"]:
            with st.container():
                # Use markdown to render formatted text properly
                st.markdown(f"**Answer:**")
                st.markdown(results["llm_answer"]["answer"])
                
                # Metadata
                with st.expander("View generation details"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Model", results["llm_answer"]["model"].split("/")[-1])
                    with col_b:
                        st.metric("Generation Time", f"{results['llm_answer']['generation_time']:.2f}s")
                    with col_c:
                        st.metric("Answer Length", f"{results['llm_answer']['answer_length']} tokens")
        
        # Error display
        if results.get("error"):
            st.error(f"Error: {results['error']}")
        
        # Export results
        st.divider()
        if st.button("💾 Export Results as JSON"):
            # Prepare results for export (remove numpy arrays)
            export_data = {
                "question": results["question"],
                "intent": results["preprocessing"]["intent"] if results["preprocessing"] else None,
                "entities": results["preprocessing"]["entities"] if results["preprocessing"] else None,
                "kg_results": results["kg_results"],
                "answer": results["llm_answer"]["answer"] if results["llm_answer"] else None,
                "model": results["llm_answer"]["model"] if results["llm_answer"] else None,
                "generation_time": results["llm_answer"]["generation_time"] if results["llm_answer"] else None
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"rag_result_{results['question'][:30]}.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
