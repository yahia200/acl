"""
Quick Start Script - Test All Components
Run this to verify your setup is working correctly
"""

import sys
import os

# Fix for Windows DLL loading issue with PyTorch
# Set environment variables before importing torch-dependent packages
if sys.platform == "win32":
    # Add conda DLL path
    import pathlib
    conda_path = pathlib.Path(sys.executable).parent.parent
    dll_paths = [
        conda_path / "Library" / "bin",
        conda_path / "Library" / "usr" / "bin",
        conda_path / "Library" / "mingw-w64" / "bin",
    ]
    for dll_path in dll_paths:
        if dll_path.exists():
            os.add_dll_directory(str(dll_path))
    
    # Also try setting PATH
    os.environ["PATH"] = os.pathsep.join([
        str(conda_path / "Library" / "bin"),
        os.environ.get("PATH", "")
    ])

def test_imports():
    """Test if all required packages are installed"""
    print("="*80)
    print("Testing Package Imports...")
    print("="*80)
    
    packages = [
        ("neo4j", "Neo4j driver"),
        ("sentence_transformers", "Sentence Transformers"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("streamlit", "Streamlit"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]
    
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_good = False
    
    return all_good


def test_config():
    """Test configuration file"""
    print("\n" + "="*80)
    print("Testing Configuration...")
    print("="*80)
    
    try:
        from utils import load_config
        config = load_config()
        
        required_keys = ["URI", "USERNAME", "PASSWORD", "HF_TOKEN"]
        all_present = True
        
        for key in required_keys:
            if key in config and config[key] and config[key] != f"your_{key.lower()}_here":
                print(f"✅ {key} configured")
            else:
                print(f"❌ {key} missing or not set")
                all_present = False
        
        if not all_present:
            print("\n⚠️  Please update config.txt with your credentials")
            print("   Get HF token from: https://huggingface.co/settings/tokens")
        
        return all_present
        
    except Exception as e:
        print(f"❌ Error reading config.txt: {e}")
        return False


def test_neo4j_connection():
    """Test Neo4j connection"""
    print("\n" + "="*80)
    print("Testing Neo4j Connection...")
    print("="*80)
    
    try:
        from utils import get_neo4j_connection
        conn = get_neo4j_connection()
        
        if conn.verify_connectivity():
            print("✅ Neo4j connection successful!")
            
            # Check if KG has data
            result = conn.execute_query("MATCH (n) RETURN count(n) as node_count")
            node_count = result[0]["node_count"] if result else 0
            print(f"✅ Knowledge Graph contains {node_count} nodes")
            
            conn.close()
            return True
        else:
            print("❌ Neo4j connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j is running")
        print("2. Check config.txt has correct credentials")
        print("3. Run 'python create_kg.py' to populate the database")
        return False


def test_component_1():
    """Test Component 1: Input Preprocessing"""
    print("\n" + "="*80)
    print("Testing Component 1: Input Preprocessing...")
    print("="*80)
    
    try:
        from component_1_input_preprocessing import InputPreprocessor
        
        preprocessor = InputPreprocessor()
        test_question = "Which flights have the longest delays?"
        
        result = preprocessor.preprocess(test_question)
        
        print(f"✅ Intent: {result['intent']}")
        print(f"✅ Entities: {result['entities']}")
        print(f"✅ Embedding shape: {result['embedding_shape']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Component 1: {e}")
        return False


def test_component_2():
    """Test Component 2: Graph Retrieval"""
    print("\n" + "="*80)
    print("Testing Component 2: Graph Retrieval...")
    print("="*80)
    
    try:
        from component_2_graph_retrieval import GraphRetriever
        
        retriever = GraphRetriever()
        
        # Test a simple query
        results = retriever.query_3_common_aircraft(limit=3)
        
        print(f"✅ Retrieved {len(results)} results from Knowledge Graph")
        if results:
            print(f"   Sample: {results[0]}")
        
        retriever.close()
        return True
        
    except Exception as e:
        print(f"❌ Error in Component 2: {e}")
        return False


def test_component_3():
    """Test Component 3: LLM Layer (Uses HF Inference API)"""
    print("\n" + "="*80)
    print("Testing Component 3: LLM Layer...")
    print("="*80)
    print("ℹ️  This uses Hugging Face Inference API (no model download needed)")
    
    response = input("Do you want to test Component 3? (y/n): ")
    if response.lower() != 'y':
        print("⏭️  Skipping Component 3 test")
        return True
    
    try:
        from component_3_llm_layer import LLMHandler
        
        print("Initializing Inference Client...")
        llm = LLMHandler("Qwen/Qwen2.5-1.5B-Instruct")
        
        sample_kg_results = [
            {"generation": "Boomer", "avg_satisfaction": 3.2}
        ]
        
        result = llm.process_query(
            "Which generation has highest satisfaction?",
            sample_kg_results
        )
        
        print(f"✅ Generated answer: {result['answer'][:100]}...")
        print(f"✅ Generation time: {result['generation_time']:.2f}s")
        print(f"✅ Model: {result['model']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Component 3: {e}")
        if "401" in str(e) or "authorization" in str(e).lower():
            print("\n⚠️  Authentication failed. Check your HF_TOKEN in config.txt")
            print("   Get a token from: https://huggingface.co/settings/tokens")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🚀 AIRLINE RAG SYSTEM - QUICK START TEST")
    print("="*80)
    
    results = {
        "Packages": test_imports(),
        "Config": test_config(),
        "Neo4j": test_neo4j_connection(),
        "Component 1": test_component_1(),
        "Component 2": test_component_2(),
        "Component 3": test_component_3(),
    }
    
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nYou're ready to run the Streamlit app:")
        print("  streamlit run component_4_ui_app.py")
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the issues above before running the app.")
        print("See README.md for troubleshooting help.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
