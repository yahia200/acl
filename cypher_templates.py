"""
Cypher Query Templates Library
Contains all Cypher query templates for the airline knowledge graph
"""

# ========== Flight and Delay Queries ==========

LONGEST_DELAYS = """
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

ROUTE_SEARCH = """
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

LONGEST_ROUTES = """
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

DELAY_SATISFACTION_CORRELATION = """
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

# ========== Aircraft Queries ==========

COMMON_AIRCRAFT = """
MATCH (f:Flight)
WITH f.fleet_type_description as aircraft_type, COUNT(DISTINCT f) as flight_count
RETURN aircraft_type, flight_count
ORDER BY flight_count DESC
LIMIT $limit
"""

# ========== Airport Queries ==========

POPULAR_DEPARTURE_AIRPORTS = """
MATCH (f:Flight)-[:DEPARTS_FROM]->(airport:Airport)
WITH airport.station_code as airport_code, COUNT(DISTINCT f) as departure_count
RETURN airport_code, departure_count
ORDER BY departure_count DESC
LIMIT $limit
"""

POPULAR_ARRIVAL_AIRPORTS = """
MATCH (f:Flight)-[:ARRIVES_AT]->(airport:Airport)
WITH airport.station_code as airport_code, COUNT(DISTINCT f) as arrival_count
RETURN airport_code, arrival_count
ORDER BY arrival_count DESC
LIMIT $limit
"""

# ========== Satisfaction Queries ==========

AVG_SATISFACTION_BY_CLASS = """
MATCH (j:Journey {passenger_class: $passenger_class})
WHERE j.food_satisfaction_score IS NOT NULL
RETURN j.passenger_class as passenger_class,
       AVG(j.food_satisfaction_score) as avg_food_satisfaction,
       COUNT(j) as total_journeys,
       MIN(j.food_satisfaction_score) as min_score,
       MAX(j.food_satisfaction_score) as max_score
"""

SATISFACTION_BY_GENERATION = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE j.food_satisfaction_score IS NOT NULL AND p.generation IS NOT NULL
RETURN p.generation as generation,
       AVG(j.food_satisfaction_score) as avg_satisfaction,
       COUNT(j) as journey_count
ORDER BY avg_satisfaction DESC
"""

SATISFACTION_BY_LOYALTY = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE j.food_satisfaction_score IS NOT NULL AND p.loyalty_program_level IS NOT NULL
RETURN p.loyalty_program_level as loyalty_level,
       AVG(j.food_satisfaction_score) as avg_satisfaction,
       COUNT(j) as journey_count
ORDER BY avg_satisfaction DESC
"""

# ========== Passenger and Journey Queries ==========

GENERATION_TRAVEL_STATS = """
MATCH (p:Passenger)-[:TOOK]->(j:Journey)
WHERE p.generation IS NOT NULL
RETURN p.generation as generation,
       COUNT(j) as total_journeys,
       AVG(j.actual_flown_miles) as avg_miles_per_journey,
       SUM(j.actual_flown_miles) as total_miles
ORDER BY total_journeys DESC
"""

JOURNEY_COMPLEXITY = """
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

# ========== Schema and Graph Structure Queries ==========

GET_SCHEMA = """
CALL db.schema.visualization()
"""

COUNT_NODES_BY_LABEL = """
MATCH (n)
RETURN labels(n) as label, COUNT(n) as count
ORDER BY count DESC
"""

COUNT_RELATIONSHIPS = """
MATCH ()-[r]->()
RETURN type(r) as relationship_type, COUNT(r) as count
ORDER BY count DESC
"""

# ========== Advanced Analytical Queries ==========

ROUTE_PERFORMANCE_ANALYSIS = """
MATCH (j:Journey)-[:ON]->(f:Flight)-[:DEPARTS_FROM]->(origin:Airport)
MATCH (f)-[:ARRIVES_AT]->(dest:Airport)
WHERE j.arrival_delay_minutes IS NOT NULL AND j.food_satisfaction_score IS NOT NULL
RETURN origin.station_code as origin,
       dest.station_code as destination,
       COUNT(j) as total_journeys,
       AVG(j.arrival_delay_minutes) as avg_delay,
       AVG(j.food_satisfaction_score) as avg_satisfaction,
       f.fleet_type_description as aircraft_type
ORDER BY total_journeys DESC
LIMIT $limit
"""

PASSENGER_CLASS_ANALYSIS = """
MATCH (j:Journey)
WHERE j.passenger_class IS NOT NULL
WITH j.passenger_class as class,
     COUNT(j) as journey_count,
     AVG(j.food_satisfaction_score) as avg_satisfaction,
     AVG(j.arrival_delay_minutes) as avg_delay,
     AVG(j.actual_flown_miles) as avg_miles
RETURN class, journey_count, avg_satisfaction, avg_delay, avg_miles
ORDER BY journey_count DESC
"""

FLEET_PERFORMANCE = """
MATCH (j:Journey)-[:ON]->(f:Flight)
WHERE f.fleet_type_description IS NOT NULL
WITH f.fleet_type_description as aircraft,
     COUNT(j) as total_journeys,
     AVG(j.arrival_delay_minutes) as avg_delay,
     AVG(j.food_satisfaction_score) as avg_satisfaction
RETURN aircraft, total_journeys, avg_delay, avg_satisfaction
ORDER BY total_journeys DESC
LIMIT $limit
"""

# ========== Template Dictionary for Easy Access ==========

QUERY_TEMPLATES = {
    "longest_delays": LONGEST_DELAYS,
    "route_search": ROUTE_SEARCH,
    "common_aircraft": COMMON_AIRCRAFT,
    "satisfaction_by_class": AVG_SATISFACTION_BY_CLASS,
    "satisfaction_by_generation": SATISFACTION_BY_GENERATION,
    "satisfaction_by_loyalty": SATISFACTION_BY_LOYALTY,
    "longest_routes": LONGEST_ROUTES,
    "popular_departure_airports": POPULAR_DEPARTURE_AIRPORTS,
    "popular_arrival_airports": POPULAR_ARRIVAL_AIRPORTS,
    "generation_travel_stats": GENERATION_TRAVEL_STATS,
    "journey_complexity": JOURNEY_COMPLEXITY,
    "delay_satisfaction_correlation": DELAY_SATISFACTION_CORRELATION,
    "route_performance": ROUTE_PERFORMANCE_ANALYSIS,
    "passenger_class_analysis": PASSENGER_CLASS_ANALYSIS,
    "fleet_performance": FLEET_PERFORMANCE,
    "get_schema": GET_SCHEMA,
    "count_nodes": COUNT_NODES_BY_LABEL,
    "count_relationships": COUNT_RELATIONSHIPS,
}

# ========== Intent-to-Query Mapping ==========

INTENT_TO_QUERY = {
    "flight_delay": "longest_delays",
    "flight_route": "route_search",
    "aircraft_info": "common_aircraft",
    "satisfaction": "satisfaction_by_class",
    "satisfaction_by_demographic": "satisfaction_by_generation",
    "satisfaction_by_loyalty": "satisfaction_by_loyalty",
    "route_distance": "longest_routes",
    "popular_airports_departure": "popular_departure_airports",
    "popular_airports_arrival": "popular_arrival_airports",
    "popular_airports": "popular_departure_airports",  # default to departure
    "passenger_demographics": "generation_travel_stats",
    "journey_complexity": "journey_complexity",
    "correlation": "delay_satisfaction_correlation",
    "route_performance": "route_performance",
    "passenger_class_analysis": "passenger_class_analysis",
    "fleet_performance": "fleet_performance",
}


def get_query_template(query_name: str) -> str:
    """
    Retrieve a query template by name
    
    Args:
        query_name: Name of the query template
        
    Returns:
        The Cypher query template string
        
    Raises:
        KeyError: If query name not found
    """
    if query_name not in QUERY_TEMPLATES:
        available = ", ".join(QUERY_TEMPLATES.keys())
        raise KeyError(f"Query '{query_name}' not found. Available queries: {available}")
    
    return QUERY_TEMPLATES[query_name]


def get_query_by_intent(intent: str) -> str:
    """
    Retrieve a query template by intent
    
    Args:
        intent: The intent string (e.g., "flight_delay", "satisfaction")
        
    Returns:
        The Cypher query template string
        
    Raises:
        KeyError: If intent not found
    """
    if intent not in INTENT_TO_QUERY:
        available = ", ".join(INTENT_TO_QUERY.keys())
        raise KeyError(f"Intent '{intent}' not found. Available intents: {available}")
    
    query_name = INTENT_TO_QUERY[intent]
    return QUERY_TEMPLATES[query_name]


def list_available_queries() -> list:
    """Return a list of all available query template names"""
    return list(QUERY_TEMPLATES.keys())


def list_available_intents() -> list:
    """Return a list of all available intents"""
    return list(INTENT_TO_QUERY.keys())
