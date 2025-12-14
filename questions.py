"""
Predefined questions for the airline RAG system
Each question is designed to test different aspects of the Knowledge Graph
"""

AIRLINE_QUESTIONS = [
    # Flight-related queries
    {
        "id": 1,
        "question": "Which flights have the longest arrival delays?",
        "intent": "flight_delay",
        "entities": {"metric": "arrival_delay_minutes", "order": "DESC"}
    },
    {
        "id": 2,
        "question": "Show me all flights from LAX to IAX",
        "intent": "flight_route",
        "entities": {"origin": "LAX", "destination": "IAX"}
    },
    {
        "id": 3,
        "question": "What are the most common aircraft types used?",
        "intent": "aircraft_info",
        "entities": {"metric": "fleet_type_description"}
    },
    
    # Passenger satisfaction queries
    {
        "id": 4,
        "question": "What is the average food satisfaction score for Economy class?",
        "intent": "satisfaction",
        "entities": {"metric": "food_satisfaction_score", "class": "Economy"}
    },
    {
        "id": 5,
        "question": "Which generation of passengers gives the highest food satisfaction ratings?",
        "intent": "satisfaction_by_demographic",
        "entities": {"metric": "food_satisfaction_score", "demographic": "generation"}
    },
    {
        "id": 6,
        "question": "How satisfied are premier gold loyalty members compared to non-elite?",
        "intent": "satisfaction_by_loyalty",
        "entities": {"metric": "food_satisfaction_score", "loyalty_levels": ["premier gold", "non-elite"]}
    },
    
    # Route and distance queries
    {
        "id": 7,
        "question": "What are the longest flight routes by miles?",
        "intent": "route_distance",
        "entities": {"metric": "actual_flown_miles", "order": "DESC"}
    },
    {
        "id": 8,
        "question": "Which airports are the most popular departure points?",
        "intent": "popular_airports",
        "entities": {"type": "departure"}
    },
    {
        "id": 9,
        "question": "Which airports are the most popular arrival destinations?",
        "intent": "popular_airports",
        "entities": {"type": "arrival"}
    },
    
    # Passenger demographics
    {
        "id": 10,
        "question": "Which generation travels the most?",
        "intent": "passenger_demographics",
        "entities": {"metric": "generation"}
    },
    
    # Multi-leg journey queries
    {
        "id": 11,
        "question": "How many passengers take multi-leg journeys versus direct flights?",
        "intent": "journey_complexity",
        "entities": {"metric": "number_of_legs"}
    },
    
    # Combined queries
    {
        "id": 12,
        "question": "What is the relationship between flight delays and passenger satisfaction?",
        "intent": "correlation",
        "entities": {"metrics": ["arrival_delay_minutes", "food_satisfaction_score"]}
    },
]


def get_all_questions():
    """Return all predefined questions"""
    return AIRLINE_QUESTIONS


def get_question_text_only():
    """Return list of question texts only"""
    return [q["question"] for q in AIRLINE_QUESTIONS]
