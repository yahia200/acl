import csv
from neo4j import GraphDatabase

def load_config():
    config = {}
    with open("config.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value
    return config

config = load_config()
URI = config["URI"]
USERNAME = config["USERNAME"]
PASSWORD = config["PASSWORD"]

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Passenger) REQUIRE p.record_locator IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (j:Journey) REQUIRE j.feedback_ID IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Flight) REQUIRE (f.flight_number, f.fleet_type_description) IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Airport) REQUIRE a.station_code IS UNIQUE")

def insert_row(tx, row):
    tx.run("""
        MERGE (p:Passenger {record_locator: $record_locator})
        SET p.loyalty_program_level = $loyalty_program_level,
            p.generation = $generation

        MERGE (j:Journey {feedback_ID: $feedback_ID})
        SET j.food_satisfaction_score = toInteger($food_satisfaction_score),
            j.arrival_delay_minutes = toInteger($arrival_delay_minutes),
            j.actual_flown_miles = toInteger($actual_flown_miles),
            j.number_of_legs = toInteger($number_of_legs),
            j.passenger_class = $passenger_class

        MERGE (f:Flight {
            flight_number: toInteger($flight_number),
            fleet_type_description: $fleet_type_description
        })

        MERGE (a1:Airport {station_code: $origin_station_code})
        MERGE (a2:Airport {station_code: $destination_station_code})

        MERGE (p)-[:TOOK]->(j)
        MERGE (j)-[:ON]->(f)
        MERGE (f)-[:DEPARTS_FROM]->(a1)
        MERGE (f)-[:ARRIVES_AT]->(a2)
    """, row)

def load_csv():
    with driver.session() as session:
        session.execute_write(create_constraints)
        with open("Airline_surveys_sample.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session.execute_write(insert_row, row)

if __name__ == "__main__":
    load_csv()
    print("Knowledge Graph successfully created.")
