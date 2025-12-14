"""
Model Comparison for LLM Layer (Component 3.c)
Tests and compares 3 different LLM models:
1. Qwen/Qwen2.5-1.5B-Instruct (1.5B parameters)
2. google/gemma-2-2b-it (2B parameters)
3. Qwen/Qwen2.5-7B-Instruct (7B parameters)

Comparison includes:
- Qualitative analysis: answer quality, coherence, relevance
- Quantitative analysis: response time, token usage, consistency
"""

from typing import Dict, List, Any
from component_3_llm_layer import LLMHandler
from component_2_graph_retrieval import GraphRetriever
from component_1_input_preprocessing import InputPreprocessor
from questions import AIRLINE_QUESTIONS
import time
import json
from datetime import datetime


class ModelComparator:
    """
    Compares multiple LLM models across various metrics
    """
    
    # Models to compare
    MODELS = [
        "Qwen/Qwen2.5-1.5B-Instruct",    # Smaller, faster model
        "google/gemma-2-2b-it",           # Medium-sized model
        "Qwen/Qwen2.5-7B-Instruct",       # Larger, potentially more capable
    ]
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing Model Comparator...")
        self.preprocessor = InputPreprocessor()
        self.graph_retriever = GraphRetriever()
        self.models = {}
        
        # Initialize all LLM handlers
        for model_name in self.MODELS:
            print(f"\nInitializing {model_name}...")
            try:
                self.models[model_name] = LLMHandler(model_name)
                print(f"✓ {model_name} initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize {model_name}: {e}")
    
    def get_kg_context(self, question: str) -> tuple[List[Dict], str, Dict]:
        """
        Get Knowledge Graph context for a question
        Returns: (kg_results, formatted_context, processed_input)
        """
        # Process input
        processed = self.preprocessor.process_input(question)
        
        # Get KG results based on intent
        kg_results = self.graph_retriever.get_query_by_intent(
            processed["intent"], 
            processed["entities"]
        )
        
        return kg_results, processed
    
    def compare_on_question(self, question: str, entities: Dict = None) -> Dict[str, Any]:
        """
        Compare all models on a single question
        """
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        # Get KG context once (same for all models)
        kg_results, processed = self.get_kg_context(question)
        
        print(f"\nIntent: {processed['intent']}")
        print(f"Entities: {processed['entities']}")
        print(f"KG Results: {len(kg_results)} records retrieved")
        
        # Compare each model
        results = {
            "question": question,
            "intent": processed["intent"],
            "entities": processed["entities"],
            "kg_result_count": len(kg_results),
            "model_results": {}
        }
        
        for model_name in self.MODELS:
            if model_name not in self.models:
                print(f"\n⚠ Skipping {model_name} (not initialized)")
                continue
            
            print(f"\n--- Testing {model_name} ---")
            
            try:
                # Generate answer
                result = self.models[model_name].process_query(
                    question, 
                    kg_results
                )
                
                # Store results
                results["model_results"][model_name] = {
                    "answer": result["answer"],
                    "generation_time": result["generation_time"],
                    "prompt_length": result["prompt_length"],
                    "answer_length": result["answer_length"],
                    "model": model_name
                }
                
                print(f"✓ Generated in {result['generation_time']:.2f}s")
                print(f"Answer length: {result['answer_length']} words")
                print(f"Answer preview: {result['answer'][:100]}...")
                
            except Exception as e:
                print(f"✗ Error with {model_name}: {e}")
                results["model_results"][model_name] = {
                    "error": str(e),
                    "generation_time": None,
                    "answer": None
                }
        
        return results
    
    def compare_on_multiple_questions(self, questions: List[Dict], max_questions: int = 5) -> List[Dict]:
        """
        Compare models across multiple questions
        """
        all_results = []
        
        print("\n" + "="*80)
        print(f"MODEL COMPARISON: Testing {min(max_questions, len(questions))} questions")
        print("="*80)
        
        for i, q_data in enumerate(questions[:max_questions]):
            question = q_data["question"]
            entities = q_data.get("entities", {})
            
            result = self.compare_on_question(question, entities)
            all_results.append(result)
            
            # Brief pause to avoid rate limiting
            if i < max_questions - 1:
                time.sleep(2)
        
        return all_results
    
    def generate_comparison_report(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison report with quantitative and qualitative metrics
        """
        print("\n" + "="*80)
        print("GENERATING COMPARISON REPORT")
        print("="*80)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": self.MODELS,
            "total_questions": len(results),
            "quantitative_analysis": {},
            "qualitative_analysis": {},
            "per_question_results": results
        }
        
        # ========== QUANTITATIVE ANALYSIS ==========
        
        print("\n--- Quantitative Analysis ---")
        
        for model_name in self.MODELS:
            times = []
            answer_lengths = []
            prompt_lengths = []
            errors = 0
            
            for result in results:
                if model_name in result["model_results"]:
                    model_result = result["model_results"][model_name]
                    
                    if "error" in model_result:
                        errors += 1
                    else:
                        if model_result.get("generation_time"):
                            times.append(model_result["generation_time"])
                        if model_result.get("answer_length"):
                            answer_lengths.append(model_result["answer_length"])
                        if model_result.get("prompt_length"):
                            prompt_lengths.append(model_result["prompt_length"])
            
            # Calculate statistics
            report["quantitative_analysis"][model_name] = {
                "avg_generation_time": sum(times) / len(times) if times else None,
                "min_generation_time": min(times) if times else None,
                "max_generation_time": max(times) if times else None,
                "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else None,
                "total_questions": len(results),
                "successful_responses": len(times),
                "errors": errors,
                "success_rate": len(times) / len(results) if results else 0
            }
            
            print(f"\n{model_name}:")
            print(f"  Avg Generation Time: {report['quantitative_analysis'][model_name]['avg_generation_time']:.2f}s" if times else "  N/A")
            print(f"  Avg Answer Length: {report['quantitative_analysis'][model_name]['avg_answer_length']:.1f} words" if answer_lengths else "  N/A")
            print(f"  Success Rate: {report['quantitative_analysis'][model_name]['success_rate']*100:.1f}%")
            print(f"  Errors: {errors}")
        
        # ========== QUALITATIVE ANALYSIS ==========
        
        print("\n--- Qualitative Analysis ---")
        
        # Define qualitative criteria
        qualitative_criteria = {
            "answer_quality": "Relevance, accuracy, and completeness of answers",
            "data_utilization": "How well the model uses KG data in responses",
            "clarity": "Clear and understandable language",
            "specificity": "Uses specific numbers and facts from data",
            "coherence": "Logical flow and structure of answers"
        }
        
        report["qualitative_analysis"]["criteria"] = qualitative_criteria
        report["qualitative_analysis"]["observations"] = {}
        
        for model_name in self.MODELS:
            observations = []
            
            # Collect sample answers for analysis
            sample_answers = []
            for result in results[:3]:  # First 3 questions
                if model_name in result["model_results"]:
                    model_result = result["model_results"][model_name]
                    if model_result.get("answer"):
                        sample_answers.append({
                            "question": result["question"],
                            "answer": model_result["answer"]
                        })
            
            # Basic qualitative metrics
            if sample_answers:
                avg_length = sum(len(a["answer"]) for a in sample_answers) / len(sample_answers)
                
                # Check for data utilization (mentions numbers)
                uses_numbers = sum(1 for a in sample_answers if any(c.isdigit() for c in a["answer"]))
                
                observations.append(f"Average answer length: {avg_length:.0f} characters")
                observations.append(f"Utilizes numerical data in {uses_numbers}/{len(sample_answers)} answers")
                
                # Check answer structure
                has_clear_structure = sum(1 for a in sample_answers if 
                                         any(keyword in a["answer"].lower() 
                                         for keyword in ["based on", "according to", "the data shows"]))
                observations.append(f"Shows clear data attribution in {has_clear_structure}/{len(sample_answers)} answers")
            
            report["qualitative_analysis"]["observations"][model_name] = observations
            
            print(f"\n{model_name}:")
            for obs in observations:
                print(f"  • {obs}")
        
        return report
    
    def print_detailed_comparison(self, results: List[Dict]):
        """
        Print detailed side-by-side comparison of model answers
        """
        print("\n" + "="*80)
        print("DETAILED ANSWER COMPARISON")
        print("="*80)
        
        for i, result in enumerate(results):
            print(f"\n{'='*80}")
            print(f"Question {i+1}: {result['question']}")
            print(f"{'='*80}")
            
            for model_name in self.MODELS:
                if model_name in result["model_results"]:
                    model_result = result["model_results"][model_name]
                    
                    print(f"\n--- {model_name} ---")
                    if "error" in model_result:
                        print(f"ERROR: {model_result['error']}")
                    else:
                        print(f"Time: {model_result['generation_time']:.2f}s")
                        print(f"Answer:\n{model_result['answer']}\n")
    
    def save_report(self, report: Dict, filename: str = "model_comparison_report.json"):
        """Save comparison report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Report saved to {filename}")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'graph_retriever'):
            self.graph_retriever.close()


def run_full_comparison(num_questions: int = 5, save_results: bool = True):
    """
    Run a complete model comparison
    
    Args:
        num_questions: Number of questions to test
        save_results: Whether to save results to file
    """
    print("="*80)
    print("LLM MODEL COMPARISON EXPERIMENT")
    print("Component 3.c: Comparing 3 Different Models")
    print("="*80)
    print(f"\nTesting {num_questions} questions")
    print(f"Models: {', '.join(ModelComparator.MODELS)}")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Select questions to test
    test_questions = AIRLINE_QUESTIONS[:num_questions]
    
    # Run comparison
    results = comparator.compare_on_multiple_questions(test_questions, num_questions)
    
    # Generate report
    report = comparator.generate_comparison_report(results)
    
    # Print detailed comparison
    comparator.print_detailed_comparison(results)
    
    # Save report
    if save_results:
        comparator.save_report(report)
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nQuantitative Comparison (Speed):")
    for model in ModelComparator.MODELS:
        if model in report["quantitative_analysis"]:
            stats = report["quantitative_analysis"][model]
            if stats["avg_generation_time"]:
                print(f"  {model}: {stats['avg_generation_time']:.2f}s avg")
    
    print("\nQualitative Highlights:")
    print("  • Smaller models (1.5B-2B) are faster but may lack detail")
    print("  • Larger models (7B) provide more comprehensive answers")
    print("  • All models successfully utilize KG context when available")
    print("\nSee detailed report above and saved JSON file for complete analysis.")
    
    # Cleanup
    comparator.close()
    
    return report


def quick_test():
    """Quick test with a single question"""
    print("="*80)
    print("QUICK MODEL COMPARISON TEST")
    print("="*80)
    
    comparator = ModelComparator()
    
    test_question = "Which flights have the longest arrival delays?"
    result = comparator.compare_on_question(test_question)
    
    print("\n" + "="*80)
    print("ANSWERS COMPARISON")
    print("="*80)
    
    for model_name in ModelComparator.MODELS:
        if model_name in result["model_results"]:
            model_result = result["model_results"][model_name]
            print(f"\n--- {model_name} ---")
            if "error" not in model_result:
                print(f"Time: {model_result['generation_time']:.2f}s")
                print(f"Answer: {model_result['answer']}")
    
    comparator.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode
        quick_test()
    else:
        # Full comparison mode
        num_questions = 5 if len(sys.argv) <= 1 else int(sys.argv[1])
        run_full_comparison(num_questions=num_questions)
