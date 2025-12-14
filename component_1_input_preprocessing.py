"""
Component 1: Input Preprocessing
- Intent Classification
- Entity Extraction
- Input Embedding
- Error Analysis and Improvement
"""

import re
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
import numpy as np


class InputPreprocessor:
    """
    Handles all input preprocessing tasks:
    1. Intent Classification - determine what the user is asking about
    2. Entity Extraction - extract key entities (airports, classes, generations, etc.)
    3. Input Embedding - convert input to vector representation
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with an embedding model"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Define intent patterns (rule-based classification)
        self.intent_patterns = {
            "flight_delay": [
                r"delay",
                r"late",
                r"on time",
                r"arrival.*delay"
            ],
            "flight_route": [
                r"from .* to",
                r"route",
                r"between .* and"
            ],
            "aircraft_info": [
                r"aircraft",
                r"plane",
                r"fleet",
                r"type.*used"
            ],
            "satisfaction": [
                r"satisfaction",
                r"rating",
                r"score",
                r"happy",
                r"satisfied"
            ],
            "satisfaction_by_demographic": [
                r"generation.*satisfaction",
                r"which generation.*rating",
                r"(millennial|boomer|gen x).*score"
            ],
            "satisfaction_by_loyalty": [
                r"loyalty.*satisfaction",
                r"(elite|premier).*rating",
                r"member.*score"
            ],
            "route_distance": [
                r"longest.*route",
                r"miles",
                r"distance",
                r"far"
            ],
            "popular_airports": [
                r"popular.*airport",
                r"most.*depart",
                r"most.*arrival",
                r"busiest"
            ],
            "passenger_demographics": [
                r"which generation",
                r"who travels",
                r"demographic"
            ],
            "journey_complexity": [
                r"multi-leg",
                r"direct flight",
                r"connection",
                r"layover"
            ],
            "correlation": [
                r"relationship",
                r"correlation",
                r"between .* and"
            ]
        }
        
        # Known entities
        self.known_airports = ["LAX", "IAX", "LHX", "EWX", "DEX", "DFX", "MYX", "SEX", 
                               "SFX", "ORX", "FRX", "SAX", "EDX", "ANX", "FCX", "AUX", 
                               "PIX", "SJX"]
        self.known_classes = ["Economy", "Business", "First"]
        self.known_generations = ["Millennial", "Gen X", "Boomer", "Gen Z"]
        self.known_loyalty_levels = ["non-elite", "premier silver", "premier gold", "premier platinum"]
        self.known_aircraft = ["B777-200", "B787-9", "B737-MAX8", "B737-900", "ERJ-175", 
                               "A320-200", "B777-300", "B767-400", "B787-8", "B737-800"]
    
    def classify_intent(self, question: str) -> str:
        """
        Classify the intent of the user's question using rule-based pattern matching
        Returns the detected intent or 'general' if no specific intent is found
        """
        question_lower = question.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return intent
        
        return "general"
    
    def extract_entities(self, question: str) -> Dict[str, Any]:
        """
        Extract relevant entities from the question
        Returns a dictionary of extracted entities
        """
        entities = {}
        question_upper = question.upper()
        question_lower = question.lower()
        
        # Extract airports
        found_airports = [airport for airport in self.known_airports 
                         if airport in question_upper]
        if found_airports:
            if "from" in question_lower and "to" in question_lower:
                # Try to determine origin and destination
                from_idx = question_lower.index("from")
                to_idx = question_lower.index("to")
                origin_candidates = [a for a in found_airports 
                                    if question_upper.index(a) > from_idx and question_upper.index(a) < to_idx]
                dest_candidates = [a for a in found_airports 
                                  if question_upper.index(a) > to_idx]
                
                if origin_candidates:
                    entities["origin"] = origin_candidates[0]
                if dest_candidates:
                    entities["destination"] = dest_candidates[0]
            else:
                entities["airports"] = found_airports
        
        # Extract passenger class
        for pclass in self.known_classes:
            if pclass.lower() in question_lower:
                entities["passenger_class"] = pclass
                break
        
        # Extract generation
        for gen in self.known_generations:
            if gen.lower() in question_lower:
                entities["generation"] = gen
                break
        
        # Extract loyalty level
        for loyalty in self.known_loyalty_levels:
            if loyalty.lower() in question_lower:
                entities["loyalty_level"] = loyalty
                break
        
        # Extract aircraft type
        for aircraft in self.known_aircraft:
            if aircraft.lower() in question_lower:
                entities["aircraft_type"] = aircraft
                break
        
        # Extract metrics based on keywords
        if any(word in question_lower for word in ["delay", "late", "on time"]):
            entities["metric"] = "arrival_delay_minutes"
        elif any(word in question_lower for word in ["food", "satisfaction", "rating"]):
            entities["metric"] = "food_satisfaction_score"
        elif any(word in question_lower for word in ["miles", "distance"]):
            entities["metric"] = "actual_flown_miles"
        
        # Extract order (for sorting)
        if any(word in question_lower for word in ["longest", "highest", "most", "maximum"]):
            entities["order"] = "DESC"
        elif any(word in question_lower for word in ["shortest", "lowest", "least", "minimum"]):
            entities["order"] = "ASC"
        
        return entities
    
    def embed_input(self, question: str) -> np.ndarray:
        """
        Convert the input question to an embedding vector
        Returns a numpy array of the embedding
        """
        embedding = self.embedding_model.encode(question, convert_to_numpy=True)
        return embedding
    
    def preprocess(self, question: str) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        Returns all preprocessing results
        """
        intent = self.classify_intent(question)
        entities = self.extract_entities(question)
        embedding = self.embed_input(question)
        
        return {
            "original_question": question,
            "intent": intent,
            "entities": entities,
            "embedding": embedding,
            "embedding_shape": embedding.shape
        }
    
def demo_preprocessing():
    """Demonstrate the input preprocessing capabilities"""
    print("="*80)
    print("Component 1: Input Preprocessing Demo")
    print("="*80)
    
    # Initialize preprocessor
    print("\nInitializing Input Preprocessor...")
    preprocessor = InputPreprocessor()
    
    # Test questions
    test_questions = [
        "Which flights have the longest arrival delays?",
        "Show me all flights from LAX to IAX",
        "What is the average food satisfaction score for Economy class?",
        "Which generation travels the most?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        result = preprocessor.preprocess(question)
        print(f"Intent: {result['intent']}")
        print(f"Entities: {result['entities']}")
        print(f"Embedding shape: {result['embedding_shape']}")
        print(f"Embedding (first 5 dims): {result['embedding'][:5]}")


if __name__ == "__main__":
    demo_preprocessing()
