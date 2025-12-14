"""
Component 2: Graph Retrieval Layer
- Baseline: Cypher queries for retrieving information
- Embeddings: Node embeddings using two different models
"""

from typing import Dict, List, Any, Optional
from utils import Neo4jConnection, get_neo4j_connection, format_kg_results
from sentence_transformers import SentenceTransformer
import numpy as np


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
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(origin:Airport)
        MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
        WHERE j.arrival_delay_minutes IS NOT NULL
        RETURN f.flight_number as flight_number,
               origin.station_code as origin,
               dest.station_code as destination,
               f.fleet_type_description as aircraft,
               AVG(j.arrival_delay_minutes) as avg_delay_minutes,
               COUNT(j) as journey_count
        ORDER BY avg_delay_minutes DESC
        LIMIT $limit
        """
        return self.conn.execute_query(query, {"limit": limit})
    
    def query_2_route_search(self, origin: str, destination: str) -> List[Dict]:
        """Query 2: Flights on a specific route"""
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(origin:Airport {station_code: $origin})
        MATCH (f)-[:ARRIVES_AT]->(dest:Airport {station_code: $destination})
        MATCH (j:Journey)-[:ON]->(f)
        RETURN f.flight_number as flight_number,
               f.fleet_type_description as aircraft,
               origin.station_code as origin,
               dest.station_code as destination,
               COUNT(j) as number_of_journeys,
               AVG(j.arrival_delay_minutes) as avg_delay
        """
        return self.conn.execute_query(query, {"origin": origin, "destination": destination})
    
    def query_3_common_aircraft(self, limit: int = 10) -> List[Dict]:
        """Query 3: Most common aircraft types"""
        query = """
        MATCH (f:Flight)
        WITH f.fleet_type_description as aircraft_type, COUNT(DISTINCT f) as flight_count
        RETURN aircraft_type, flight_count
        ORDER BY flight_count DESC
        LIMIT $limit
        """
        return self.conn.execute_query(query, {"limit": limit})
    
    def query_4_avg_satisfaction_by_class(self, passenger_class: str) -> List[Dict]:
        """Query 4: Average food satisfaction by passenger class"""
        query = """
        MATCH (j:Journey {passenger_class: $passenger_class})
        WHERE j.food_satisfaction_score IS NOT NULL
        RETURN j.passenger_class as passenger_class,
               AVG(j.food_satisfaction_score) as avg_food_satisfaction,
               COUNT(j) as total_journeys,
               MIN(j.food_satisfaction_score) as min_score,
               MAX(j.food_satisfaction_score) as max_score
        """
        return self.conn.execute_query(query, {"passenger_class": passenger_class})
    
    def query_5_satisfaction_by_generation(self) -> List[Dict]:
        """Query 5: Food satisfaction by generation"""
        query = """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE j.food_satisfaction_score IS NOT NULL AND p.generation IS NOT NULL
        RETURN p.generation as generation,
               AVG(j.food_satisfaction_score) as avg_satisfaction,
               COUNT(j) as journey_count
        ORDER BY avg_satisfaction DESC
        """
        return self.conn.execute_query(query)
    
    def query_6_satisfaction_by_loyalty(self) -> List[Dict]:
        """Query 6: Satisfaction by loyalty level"""
        query = """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE j.food_satisfaction_score IS NOT NULL AND p.loyalty_program_level IS NOT NULL
        RETURN p.loyalty_program_level as loyalty_level,
               AVG(j.food_satisfaction_score) as avg_satisfaction,
               COUNT(j) as journey_count
        ORDER BY avg_satisfaction DESC
        """
        return self.conn.execute_query(query)
    
    def query_7_longest_routes(self, limit: int = 10) -> List[Dict]:
        """Query 7: Longest flight routes by miles"""
        query = """
        MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(origin:Airport)
        MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
        WHERE j.actual_flown_miles IS NOT NULL
        RETURN f.flight_number as flight_number,
               origin.station_code as origin,
               dest.station_code as destination,
               j.actual_flown_miles as miles,
               f.fleet_type_description as aircraft
        ORDER BY j.actual_flown_miles DESC
        LIMIT $limit
        """
        return self.conn.execute_query(query, {"limit": limit})
    
    def query_8_popular_departure_airports(self, limit: int = 10) -> List[Dict]:
        """Query 8: Most popular departure airports"""
        query = """
        MATCH (f:Flight)-[:DEPARTS_FROM]->(airport:Airport)
        WITH airport.station_code as airport_code, COUNT(DISTINCT f) as departure_count
        RETURN airport_code, departure_count
        ORDER BY departure_count DESC
        LIMIT $limit
        """
        return self.conn.execute_query(query, {"limit": limit})
    
    def query_9_popular_arrival_airports(self, limit: int = 10) -> List[Dict]:
        """Query 9: Most popular arrival airports"""
        query = """
        MATCH (f:Flight)-[:ARRIVES_AT]->(airport:Airport)
        WITH airport.station_code as airport_code, COUNT(DISTINCT f) as arrival_count
        RETURN airport_code, arrival_count
        ORDER BY arrival_count DESC
        LIMIT $limit
        """
        return self.conn.execute_query(query, {"limit": limit})
    
    def query_10_generation_travel_stats(self) -> List[Dict]:
        """Query 10: Which generation travels the most"""
        query = """
        MATCH (p:Passenger)-[:TOOK]->(j:Journey)
        WHERE p.generation IS NOT NULL
        RETURN p.generation as generation,
               COUNT(j) as total_journeys,
               AVG(j.actual_flown_miles) as avg_miles_per_journey,
               SUM(j.actual_flown_miles) as total_miles
        ORDER BY total_journeys DESC
        """
        return self.conn.execute_query(query)
    
    def query_11_journey_complexity(self) -> List[Dict]:
        """Query 11: Multi-leg vs direct flights"""
        query = """
        MATCH (j:Journey)
        WHERE j.number_of_legs IS NOT NULL
        WITH CASE 
            WHEN j.number_of_legs = 1 THEN 'Direct'
            ELSE 'Multi-leg'
        END as journey_type,
        j.number_of_legs as legs,
        COUNT(j) as count
        RETURN journey_type, legs, count
        ORDER BY legs
        """
        return self.conn.execute_query(query)
    
    def query_12_delay_satisfaction_correlation(self) -> List[Dict]:
        """Query 12: Relationship between delays and satisfaction"""
        query = """
        MATCH (j:Journey)
        WHERE j.arrival_delay_minutes IS NOT NULL 
          AND j.food_satisfaction_score IS NOT NULL
        WITH CASE
            WHEN j.arrival_delay_minutes <= 0 THEN 'On Time/Early'
            WHEN j.arrival_delay_minutes <= 30 THEN 'Short Delay (1-30 min)'
            WHEN j.arrival_delay_minutes <= 60 THEN 'Medium Delay (31-60 min)'
            ELSE 'Long Delay (60+ min)'
        END as delay_category,
        j.arrival_delay_minutes as delay,
        j.food_satisfaction_score as satisfaction
        RETURN delay_category,
               COUNT(*) as journey_count,
               AVG(satisfaction) as avg_satisfaction,
               AVG(delay) as avg_delay_minutes
        ORDER BY avg_delay_minutes
        """
        return self.conn.execute_query(query)
    
    def get_query_by_intent(self, intent: str, entities: Dict[str, Any]) -> List[Dict]:
        """Route to appropriate query based on intent and entities"""
        
        if intent == "flight_delay":
            return self.query_1_longest_delays()
        
        elif intent == "flight_route":
            if "origin" in entities and "destination" in entities:
                return self.query_2_route_search(entities["origin"], entities["destination"])
            return []
        
        elif intent == "aircraft_info":
            return self.query_3_common_aircraft()
        
        elif intent == "satisfaction":
            if "passenger_class" in entities:
                return self.query_4_avg_satisfaction_by_class(entities["passenger_class"])
            return []
        
        elif intent == "satisfaction_by_demographic":
            return self.query_5_satisfaction_by_generation()
        
        elif intent == "satisfaction_by_loyalty":
            return self.query_6_satisfaction_by_loyalty()
        
        elif intent == "route_distance":
            return self.query_7_longest_routes()
        
        elif intent == "popular_airports":
            if entities.get("type") == "arrival":
                return self.query_9_popular_arrival_airports()
            else:
                return self.query_8_popular_departure_airports()
        
        elif intent == "passenger_demographics":
            return self.query_10_generation_travel_stats()
        
        elif intent == "journey_complexity":
            return self.query_11_journey_complexity()
        
        elif intent == "correlation":
            return self.query_12_delay_satisfaction_correlation()
        
        else:
            # Default: return some general statistics
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
