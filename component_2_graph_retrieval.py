"""
Component 2: Graph Retrieval Layer
- Baseline: Cypher queries for retrieving information
- Embeddings: Node embeddings using two different models
"""

from typing import Dict, List, Any, Optional
from utils import Neo4jConnection, get_neo4j_connection, format_kg_results
from sentence_transformers import SentenceTransformer
import numpy as np
from cypher_templates import QUERY_TEMPLATES, get_query_by_intent


class GraphRetriever:
    """
    Handles Knowledge Graph retrieval through two approaches:
    1. Baseline: Direct Cypher queries
    2. Embeddings: Node embeddings for similarity-based retrieval
    """
    
    def __init__(self, neo4j_conn: Optional[Neo4jConnection] = None):
        """Initialize with Neo4j connection"""
        self.conn = neo4j_conn or get_neo4j_connection()
        
    def close(self):
        """Close Neo4j connection"""
        if self.conn:
            self.conn.close()
    
    # ========== BASELINE: Cypher Queries ==========
    
    def query_1_longest_delays(self, limit: int = 10) -> List[Dict]:
        """Query 1: Flights with longest arrival delays"""
        return self.conn.execute_query(QUERY_TEMPLATES["longest_delays"], {"limit": limit})
    
    def query_2_route_search(self, origin: str, destination: str) -> List[Dict]:
        """Query 2: Flights on a specific route"""
        return self.conn.execute_query(QUERY_TEMPLATES["route_search"], 
                                        {"origin": origin, "destination": destination})
    
    def query_3_common_aircraft(self, limit: int = 10) -> List[Dict]:
        """Query 3: Most common aircraft types"""
        return self.conn.execute_query(QUERY_TEMPLATES["common_aircraft"], {"limit": limit})
    
    def query_4_avg_satisfaction_by_class(self, passenger_class: str) -> List[Dict]:
        """Query 4: Average food satisfaction by passenger class"""
        return self.conn.execute_query(QUERY_TEMPLATES["satisfaction_by_class"], 
                                        {"passenger_class": passenger_class})
    
    def query_5_satisfaction_by_generation(self) -> List[Dict]:
        """Query 5: Food satisfaction by generation"""
        return self.conn.execute_query(QUERY_TEMPLATES["satisfaction_by_generation"])
    
    def query_6_satisfaction_by_loyalty(self) -> List[Dict]:
        """Query 6: Satisfaction by loyalty level"""
        return self.conn.execute_query(QUERY_TEMPLATES["satisfaction_by_loyalty"])
    
    def query_7_longest_routes(self, limit: int = 10) -> List[Dict]:
        """Query 7: Longest flight routes by miles"""
        return self.conn.execute_query(QUERY_TEMPLATES["longest_routes"], {"limit": limit})
    
    def query_8_popular_departure_airports(self, limit: int = 10) -> List[Dict]:
        """Query 8: Most popular departure airports"""
        return self.conn.execute_query(QUERY_TEMPLATES["popular_departure_airports"], {"limit": limit})
    
    def query_9_popular_arrival_airports(self, limit: int = 10) -> List[Dict]:
        """Query 9: Most popular arrival airports"""
        return self.conn.execute_query(QUERY_TEMPLATES["popular_arrival_airports"], {"limit": limit})
    
    def query_10_generation_travel_stats(self) -> List[Dict]:
        """Query 10: Which generation travels the most"""
        return self.conn.execute_query(QUERY_TEMPLATES["generation_travel_stats"])
    
    def query_11_journey_complexity(self) -> List[Dict]:
        """Query 11: Multi-leg vs direct flights"""
        return self.conn.execute_query(QUERY_TEMPLATES["journey_complexity"])
    
    def query_12_delay_satisfaction_correlation(self) -> List[Dict]:
        """Query 12: Relationship between delays and satisfaction"""
        return self.conn.execute_query(QUERY_TEMPLATES["delay_satisfaction_correlation"])
    
    def get_query_by_intent(self, intent: str, entities: Dict[str, Any]) -> List[Dict]:
        """Route to appropriate query based on intent and entities"""
        
        try:
            # Prepare parameters based on intent
            params = {}
            
            # Special handling for intents with required parameters
            if intent == "flight_route":
                if "origin" not in entities or "destination" not in entities:
                    return []
                query_template = get_query_by_intent(intent)
                params = {"origin": entities["origin"], "destination": entities["destination"]}
            
            elif intent == "satisfaction":
                # Only use satisfaction_by_class if passenger_class is provided
                if "passenger_class" not in entities:
                    return []
                query_template = get_query_by_intent(intent)
                params = {"passenger_class": entities["passenger_class"]}
            
            elif intent == "popular_airports":
                # Handle special case for arrival vs departure airports
                if entities.get("type") == "arrival":
                    query_template = QUERY_TEMPLATES["popular_arrival_airports"]
                else:
                    query_template = get_query_by_intent(intent)
                params = {"limit": entities.get("limit", 10)}
            
            elif intent in ["flight_delay", "aircraft_info", "route_distance", "popular_airports_departure", 
                          "popular_airports_arrival", "route_performance", "fleet_performance"]:
                # Queries that take limit parameter
                query_template = get_query_by_intent(intent)
                params = {"limit": entities.get("limit", 10)}
            
            else:
                # For other intents (no parameters needed)
                query_template = get_query_by_intent(intent)
            
            # Execute query
            return self.conn.execute_query(query_template, params)
            
        except KeyError:
            # Intent not found, return default statistics
            return self.query_10_generation_travel_stats()


class NodeEmbeddingRetriever:
    """
    Retrieves information using node embeddings for similarity-based search
    Tests two different embedding models
    """
    
    def __init__(self, 
                 neo4j_conn: Optional[Neo4jConnection] = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with Neo4j connection and embedding model"""
        self.conn = neo4j_conn or get_neo4j_connection()
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        
    def create_node_embeddings(self):
        """
        Create embeddings for all nodes in the graph
        This creates a text representation of each node and embeds it
        """
        print(f"Creating node embeddings using {self.model_name}...")
        
        # Get all passengers
        passengers = self.conn.execute_query("""
            MATCH (p:Passenger)
            RETURN p.record_locator as id, 
                   'Passenger' as type,
                   p.loyalty_program_level as loyalty,
                   p.generation as generation
            LIMIT 100
        """)
        
        # Get all journeys
        journeys = self.conn.execute_query("""
            MATCH (j:Journey)
            RETURN j.feedback_ID as id,
                   'Journey' as type,
                   j.passenger_class as passenger_class,
                   j.food_satisfaction_score as satisfaction,
                   j.arrival_delay_minutes as delay
            LIMIT 100
        """)
        
        # Get all flights
        flights = self.conn.execute_query("""
            MATCH (f:Flight)
            RETURN f.flight_number as id,
                   'Flight' as type,
                   f.fleet_type_description as aircraft
            LIMIT 100
        """)
        
        # Get all airports
        airports = self.conn.execute_query("""
            MATCH (a:Airport)
            RETURN a.station_code as id,
                   'Airport' as type
        """)
        
        all_nodes = []
        
        # Create text representations and embeddings
        for p in passengers:
            text = f"Passenger {p.get('loyalty', 'unknown')} loyalty level, {p.get('generation', 'unknown')} generation"
            embedding = self.embedding_model.encode(text)
            all_nodes.append({
                "id": p["id"],
                "type": "Passenger",
                "text": text,
                "embedding": embedding
            })
        
        for j in journeys:
            text = f"Journey in {j.get('passenger_class', 'unknown')} class, satisfaction {j.get('satisfaction', 'N/A')}, delay {j.get('delay', 'N/A')} minutes"
            embedding = self.embedding_model.encode(text)
            all_nodes.append({
                "id": j["id"],
                "type": "Journey",
                "text": text,
                "embedding": embedding
            })
        
        for f in flights:
            text = f"Flight {f['id']} using {f.get('aircraft', 'unknown')} aircraft"
            embedding = self.embedding_model.encode(text)
            all_nodes.append({
                "id": f["id"],
                "type": "Flight",
                "text": text,
                "embedding": embedding
            })
        
        for a in airports:
            text = f"Airport {a['id']}"
            embedding = self.embedding_model.encode(text)
            all_nodes.append({
                "id": a["id"],
                "type": "Airport",
                "text": text,
                "embedding": embedding
            })
        
        print(f"Created {len(all_nodes)} node embeddings")
        return all_nodes
    
    def find_similar_nodes(self, query_embedding: np.ndarray, node_embeddings: List[Dict], 
                          top_k: int = 5) -> List[Dict]:
        """
        Find most similar nodes to the query embedding using cosine similarity
        """
        from numpy.linalg import norm
        
        similarities = []
        for node in node_embeddings:
            # Cosine similarity
            cos_sim = np.dot(query_embedding, node["embedding"]) / (
                norm(query_embedding) * norm(node["embedding"])
            )
            similarities.append({
                "node": node,
                "similarity": float(cos_sim)
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def retrieve_by_similarity(self, question: str, question_embedding: np.ndarray, 
                              top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant information using embedding similarity
        """
        # Create node embeddings (in practice, these would be cached)
        node_embeddings = self.create_node_embeddings()
        
        # Find similar nodes
        similar_nodes = self.find_similar_nodes(question_embedding, node_embeddings, top_k)
        
        # Format results
        results = {
            "question": question,
            "model": self.model_name,
            "similar_nodes": [
                {
                    "node_id": node["node"]["id"],
                    "node_type": node["node"]["type"],
                    "description": node["node"]["text"],
                    "similarity": node["similarity"]
                }
                for node in similar_nodes
            ]
        }
        
        return results


def demo_graph_retrieval():
    """Demonstrate the graph retrieval capabilities"""
    print("="*80)
    print("Component 2: Graph Retrieval Layer Demo")
    print("="*80)
    
    # Initialize retriever
    print("\nInitializing Graph Retriever...")
    retriever = GraphRetriever()
    
    # Test baseline queries
    print("\n" + "="*80)
    print("BASELINE: Cypher Queries")
    print("="*80)
    
    print("\n--- Query 1: Flights with longest delays ---")
    results = retriever.query_1_longest_delays(limit=5)
    print(format_kg_results(results))
    
    print("\n--- Query 5: Satisfaction by generation ---")
    results = retriever.query_5_satisfaction_by_generation()
    print(format_kg_results(results))
    
    retriever.close()


if __name__ == "__main__":
    demo_graph_retrieval()
