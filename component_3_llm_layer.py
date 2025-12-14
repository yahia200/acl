"""
Component 3: LLM Layer
- Combines KG results from baseline and embeddings
- Uses structured prompts (context, persona, task)
- Compares 3 different LLM models
- Provides qualitative and quantitative analysis
"""

from typing import Dict, List, Any, Optional
from huggingface_hub import InferenceClient
import os
from utils import format_kg_results, load_config
import time


class LLMHandler:
    """
    Handles LLM interaction for generating final answers
    Tests 3 different models and compares their performance
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", api_token: Optional[str] = None):
        """Initialize with Hugging Face Inference Client
        
        Args:
            model_name: HF model to use (e.g., 'Qwen/Qwen2.5-1.5B-Instruct', 'google/gemma-2-2b-it', 'Qwen/Qwen2.5-7B-Instruct')
            api_token: HF API token (optional, reads from config.txt, then HF_TOKEN env var)
        """
        self.model_name = model_name
        
        # Get API token from: 1) parameter, 2) config.txt, 3) environment variable
        token = api_token
        if not token:
            try:
                config = load_config()
                token = config.get("HF_TOKEN")
            except:
                pass
        if not token:
            token = os.getenv("HF_TOKEN")
        
        print(f"Initializing Inference Client for {model_name}...")
        self.client = InferenceClient(model=model_name, token=token)
        
        # Determine model type for prompt handling
        if "flan-t5" in model_name.lower():
            self.model_type = "seq2seq"
        else:
            self.model_type = "causal"
        
        print(f"Client initialized successfully!")
    
    def create_structured_prompt(self, 
                                 question: str, 
                                 kg_context: str,
                                 embedding_context: Optional[str] = None) -> str:
        """
        Create a structured prompt with:
        1. Persona - who the AI should act as
        2. Context - information from the Knowledge Graph
        3. Task - what to do with the information
        """
        prompt = f"""You are an expert airline data analyst with access to a comprehensive airline database.

CONTEXT - Knowledge Graph Data:
{kg_context}
"""
        
        if embedding_context:
            prompt += f"""
Additional Context from Similar Nodes:
{embedding_context}
"""
        
        prompt += f"""
TASK:
Answer the following question based on the context provided above. 
Be specific, use numbers from the data, and provide clear insights.
If the data doesn't fully answer the question, state what information is available.

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, 
                       prompt: str, 
                       max_length: int = 512,
                       temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate an answer using the Hugging Face Inference API
        Returns the answer and generation metadata
        """
        start_time = time.time()
        
        try:
            # Use chat_completion for conversational models (Qwen, Gemma)
            # Use text_generation for seq2seq models
            if self.model_type == "seq2seq":
                response = self.client.text_generation(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    return_full_text=False
                )
                answer = response.strip()
            else:
                # For causal/conversational models, use chat completion
                messages = [{"role": "user", "content": prompt}]
                response = self.client.chat_completion(
                    messages=messages,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9
                )
                answer = response.choices[0].message.content.strip()
            
            # Try to extract just the answer part if it contains "ANSWER:"
            if "ANSWER:" in answer:
                answer = answer.split("ANSWER:")[-1].strip()
            
        except Exception as e:
            # Fallback error handling
            answer = f"Error generating response: {str(e)}"
            print(f"Generation error: {e}")
        
        generation_time = time.time() - start_time
        
        # Approximate token counts (since we don't have tokenizer)
        prompt_length = len(prompt.split())
        answer_length = len(answer.split())
        
        return {
            "answer": answer,
            "model": self.model_name,
            "generation_time": generation_time,
            "prompt_length": prompt_length,
            "answer_length": answer_length
        }
    
    def process_query(self,
                     question: str,
                     kg_results: List[Dict],
                     embedding_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Complete pipeline: create prompt and generate answer
        """
        # Format KG results
        kg_context = format_kg_results(kg_results)
        
        # Format embedding results if available
        embedding_context = None
        if embedding_results and "similar_nodes" in embedding_results:
            embedding_context = "Relevant entities:\n"
            for node in embedding_results["similar_nodes"][:3]:
                embedding_context += f"- {node['description']} (similarity: {node['similarity']:.3f})\n"
        
        # Create prompt
        prompt = self.create_structured_prompt(question, kg_context, embedding_context)
        
        # Generate answer
        result = self.generate_answer(prompt)
        result["question"] = question
        result["prompt"] = prompt
        
        return result


def demo_llm_layer():
    """Demonstrate the LLM layer capabilities"""
    print("="*80)
    print("Component 3: LLM Layer Demo")
    print("="*80)
    
    # Sample KG results
    sample_kg_results = [
        {
            "generation": "Boomer",
            "avg_satisfaction": 3.2,
            "journey_count": 450
        },
        {
            "generation": "Gen X",
            "avg_satisfaction": 2.8,
            "journey_count": 380
        },
        {
            "generation": "Millennial",
            "avg_satisfaction": 2.5,
            "journey_count": 420
        }
    ]
    
    sample_question = "Which generation of passengers gives the highest food satisfaction ratings?"
    
    print("\nInitializing LLM Handler with Qwen...")
    llm = LLMHandler("Qwen/Qwen2.5-1.5B-Instruct")
    
    print("\nProcessing query...")
    result = llm.process_query(sample_question, sample_kg_results)
    
    print(f"\n{'='*80}")
    print("RESULT")
    print(f"{'='*80}")
    print(f"Question: {result['question']}")
    print(f"Model: {result['model']}")
    print(f"Generation Time: {result['generation_time']:.2f}s")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    demo_llm_layer()
