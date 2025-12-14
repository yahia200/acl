"""
Utility functions shared across all components
"""
from neo4j import GraphDatabase
from typing import Dict, Any, List
import json


def load_config(config_path: str = "config.txt") -> Dict[str, str]:
    """Load configuration from config.txt file"""
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key] = value
    return config


class Neo4jConnection:
    """Manage Neo4j database connection"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        """Close the database connection"""
        if self.driver:
            self.driver.close()
    
    def verify_connectivity(self):
        """Verify connection to Neo4j"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as num")
                return result.single()["num"] == 1
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results as list of dicts"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def execute_write(self, query: str, parameters: Dict = None):
        """Execute a write query"""
        with self.driver.session() as session:
            return session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )


def get_neo4j_connection() -> Neo4jConnection:
    """Get Neo4j connection using config"""
    config = load_config()
    return Neo4jConnection(
        uri=config["URI"],
        username=config["USERNAME"],
        password=config["PASSWORD"]
    )


def format_kg_results(results: List[Dict]) -> str:
    """Format Knowledge Graph query results into readable text"""
    if not results:
        return "No results found in the knowledge graph."
    
    formatted = []
    for i, result in enumerate(results, 1):
        result_str = f"Result {i}:\n"
        for key, value in result.items():
            if value is not None:
                result_str += f"  {key}: {value}\n"
        formatted.append(result_str)
    
    return "\n".join(formatted)
