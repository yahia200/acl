# Airline Knowledge Graph RAG System

A complete RAG (Retrieval-Augmented Generation) system that combines Neo4j Knowledge Graph with Large Language Models to answer questions about airline data.

## 📋 Project Overview

This system implements 4 components:

1. **Input Preprocessing** - Intent classification, entity extraction, and input embedding
2. **Graph Retrieval Layer** - Baseline Cypher queries + node embeddings with 2 models
3. **LLM Layer** - Answer generation using 3 different LLM models with structured prompts
4. **Streamlit UI** - Interactive interface for querying and viewing results

## Features

- **12+ Predefined Questions** covering various airline data aspects
- **Custom Question Input** for flexible queries
- **Dual Retrieval Strategy**: Direct Cypher queries + embedding-based similarity search
- **Multiple Model Comparison**: Test different LLMs and embedding models
- **Interactive Visualization** of preprocessing, KG context, and LLM answers
- **Export Functionality** to save results as JSON

## Project Structure

```
acl/
├── config.txt                              # Neo4j connection configuration
├── create_kg.py                            # Knowledge Graph creation script
├── Airline_surveys_sample.csv              # Dataset
│
├── requirements.txt                        # Python dependencies
├── utils.py                                # Shared utility functions
├── questions.py                            # Predefined questions (12+)
│
├── component_1_input_preprocessing.py      # Component 1: Intent & Entity extraction
├── component_2_graph_retrieval.py          # Component 2: Cypher + Embeddings
├── component_3_llm_layer.py                # Component 3: LLM integration
├── component_4_ui_app.py                   # Component 4: Streamlit UI
├── cypher_templates.py                     # Cypher query template library
├── model_comparison.py                     # Model comparison experiment (3.c)
│
├── MODEL_COMPARISON.md                     # Model comparison documentation
└── README.md                               # This file
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Neo4j Desktop or Neo4j AuraDB instance
- 8GB+ RAM (for running LLMs locally)
- GPU recommended but not required

### 2. Install Neo4j

**Option A: Neo4j Desktop**
1. Download from https://neo4j.com/download/
2. Create a new project
3. Add a local database (set password)
4. Start the database
5. Note the connection URI (usually `neo4j://localhost:7687`)

**Option B: Neo4j AuraDB (Cloud)**
1. Sign up at https://neo4j.com/cloud/aura/
2. Create a free instance
3. Download connection credentials

### 3. Configure Connection

Edit `config.txt` with your Neo4j credentials:

```
URI=neo4j://localhost:7687
USERNAME=neo4j
PASSWORD=your_password_here
```

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: Initial installation may take 10-15 minutes due to large ML libraries (PyTorch, Transformers, etc.)

### 5. Create the Knowledge Graph

Run the KG creation script to populate Neo4j with airline data:

```bash
python create_kg.py
```

You should see: `Knowledge Graph successfully created.`

Verify in Neo4j Browser:
```cypher
MATCH (n) RETURN count(n)
```

### 6. Test Individual Components (Optional)

Test each component independently:

```bash
# Test Component 1: Input Preprocessing
python component_1_input_preprocessing.py

# Test Component 2: Graph Retrieval
python component_2_graph_retrieval.py

# Test Component 3: LLM Layer
python component_3_llm_layer.py
```

### 7. Run Model Comparison Experiment (Optional)

Compare all 3 LLM models on multiple questions:

```bash
# Run comparison on 5 questions (default)
python model_comparison.py

# Quick test on 1 question
python model_comparison.py quick
```

Results saved to `model_comparison_report.json`. See `MODEL_COMPARISON.md` for details.

### 8. Run the Streamlit UI

Launch the complete system:

```bash
streamlit run component_4_ui_app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the System

### First Time Setup in UI

1. Click **"Load/Reload Models"** in the sidebar (takes 2-5 minutes)
2. Wait for "✅ Models ready" confirmation

### Asking Questions

**Method 1: Select Predefined Questions**
- Go to "Select Predefined" tab
- Choose from 12+ questions covering different query types
- Click "Submit Selected Question"

**Method 2: Write Your Own**
- Go to "Write Your Own" tab
- Type your question about the airline data
- Click "Submit Question"

### Viewing Results

The interface displays:

1. **Input Analysis** - Detected intent and extracted entities
2. **Knowledge Graph Context** - Results from Cypher queries
3. **Similar Nodes** - Embedding-based retrieval results
4. **Final Answer** - LLM-generated natural language answer

### Changing Models

Use the sidebar to:
- Select different **LLM models** (FLAN-T5, Mistral, Llama)
- Switch **embedding models** (MiniLM, MPNet)
- Enable/disable embedding-based retrieval
- Click "Load/Reload Models" to apply changes

### Exporting Results

Click **"Export Results as JSON"** to download query results

## 📊 Components Deep Dive

### Component 1: Input Preprocessing

**File**: `component_1_input_preprocessing.py`

**Capabilities**:
- **Intent Classification**: Rule-based pattern matching for 10+ intent types
- **Entity Extraction**: Extracts airports, classes, generations, loyalty levels, aircraft types
- **Input Embedding**: Converts questions to vector representations
- **Error Analysis**: Accuracy metrics and error tracking

**Key Classes**:
- `InputPreprocessor` - Main preprocessing pipeline

### Component 2: Graph Retrieval Layer

**File**: `component_2_graph_retrieval.py`

**Capabilities**:

**Baseline Approach** (12 Cypher Queries):
1. Longest delays
2. Route search
3. Common aircraft
4. Satisfaction by class
5. Satisfaction by generation
6. Satisfaction by loyalty
7. Longest routes
8. Popular departure airports
9. Popular arrival airports
10. Generation travel stats
11. Journey complexity
12. Delay-satisfaction correlation

**Embedding Approach**:
- Node embeddings using 2 models
- Cosine similarity-based retrieval
- Top-K similar node extraction

**Key Classes**:
- `GraphRetriever` - Cypher query execution
- `NodeEmbeddingRetriever` - Embedding-based retrieval

### Component 3: LLM Layer

**File**: `component_3_llm_layer.py`

**Capabilities**:
- **Structured Prompts**: Context + Persona + Task format
- **Multi-Model Support**: 3 LLMs (Qwen, Gemma)
- **Answer Generation**: Natural language responses
- **Quantitative Analysis**: Generation time, token counts
- **Hugging Face Inference API**: No local model downloads needed

**Key Classes**:
- `LLMHandler` - Model interaction via HF Inference API

**Supported Models**:
1. `Qwen/Qwen2.5-1.5B-Instruct` - Fast, efficient model
2. `google/gemma-2-2b-it` - Balanced performance
3. `Qwen/Qwen2.5-7B-Instruct` - High quality answers

#### Model Comparison (Component 3.c)

**File**: `model_comparison.py`  
**Documentation**: See `MODEL_COMPARISON.md` for detailed methodology

Run comprehensive model comparison experiment:

```bash
# Full comparison (5 questions)
python model_comparison.py

# Custom number of questions
python model_comparison.py 10

# Quick test (1 question)
python model_comparison.py quick
```

**Output**: 
- Console: Real-time comparison results
- File: `model_comparison_report.json` with quantitative & qualitative analysis

**Comparison Metrics**:
- **Quantitative**: Generation time, answer length, success rate, consistency
- **Qualitative**: Answer quality, data utilization, clarity, specificity, coherence

All 3 models are tested with:
- Same questions
- Same KG context
- Same prompt structure
- Controlled parameters for fair comparison

### Component 4: Streamlit UI

**File**: `component_4_ui_app.py`

**Features**:
- Model loading and configuration
- Dual input methods (predefined + custom)
- Real-time progress indicators
- Structured result display
- JSON export functionality
- Persistent state across interactions

## 🔍 Example Questions

1. Which flights have the longest arrival delays?
2. Show me all flights from LAX to IAX
3. What are the most common aircraft types used?
4. What is the average food satisfaction score for Economy class?
5. Which generation of passengers gives the highest food satisfaction ratings?
6. How satisfied are premier gold loyalty members compared to non-elite?
7. What are the longest flight routes by miles?
8. Which airports are the most popular departure points?
9. Which airports are the most popular arrival destinations?
10. Which generation travels the most?
11. How many passengers take multi-leg journeys versus direct flights?
12. What is the relationship between flight delays and passenger satisfaction?

## 🧪 Testing and Evaluation

### Performance Metrics

The system tracks:
- Intent classification accuracy
- Query execution time
- LLM generation time
- Answer length (tokens)
- Embedding similarity scores

### Model Comparison

To compare LLMs:
1. Ask the same question with different models
2. Compare generation times
3. Evaluate answer quality
4. Check consistency with KG data

## 🛠️ Troubleshooting

### Common Issues

**1. "Import errors" when running**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate virtual environment

**2. "Connection refused" to Neo4j**
- Check Neo4j is running
- Verify `config.txt` has correct URI and credentials
- Test connection in Neo4j Browser

**3. "Out of memory" errors**
- Use smaller models (`google/flan-t5-base`)
- Enable 8-bit quantization (for GPU)
- Reduce batch sizes

**4. Models take too long to load**
- First run downloads models from Hugging Face (normal)
- Subsequent runs use cached models
- Consider using smaller models for faster iteration

**5. Streamlit won't start**
- Check port 8501 is available
- Try: `streamlit run component_4_ui_app.py --server.port 8502`

## 📝 Development Notes

### Adding New Questions

Edit `questions.py`:

```python
{
    "id": 13,
    "question": "Your question here?",
    "intent": "your_intent",
    "entities": {"key": "value"}
}
```

### Adding New Cypher Queries

Add to `GraphRetriever` class in `component_2_graph_retrieval.py`:

```python
def query_13_your_query(self, param: str) -> List[Dict]:
    """Description"""
    query = """
    MATCH (n)
    RETURN n
    """
    return self.conn.execute_query(query, {"param": param})
```

### Testing New LLM Models

Update model list in `component_4_ui_app.py`:

```python
llm_model = st.selectbox(
    "Select LLM",
    [
        "google/flan-t5-base",
        "your/new-model",
    ]
)
```

## 📚 Resources

- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/)
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🤝 Component Limitations

### Component 1: Input Preprocessing
- Rule-based intent classification may miss complex queries
- Entity extraction limited to predefined lists
- No handling of typos or variations

### Component 2: Graph Retrieval
- Fixed Cypher queries may not cover all question variations
- Node embeddings require recomputation for graph updates
- Limited to structured data in the KG

### Component 3: LLM Layer
- Answer quality depends on KG context relevance
- Larger models require significant compute resources
- May hallucinate if KG data is insufficient

### Component 4: UI
- Model loading time can be slow on first run
- Limited error recovery for malformed queries
- No query history or session persistence across restarts

## 📄 License

This project is for academic purposes (GUC CSEN903 - Advanced Computer Lab).

## 👤 Author

Milestone 3 - Airline RAG System
German University in Cairo

---

**Note**: This system is designed for educational purposes and demonstration of RAG architecture with Knowledge Graphs and LLMs.
