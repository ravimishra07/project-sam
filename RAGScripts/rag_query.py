#!/usr/bin/env python3
"""
RAG Query Script for Personal Reflection Assistant
Handles natural language queries against cleaned daily logs with embeddings
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

class RAGQueryEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embeddings_file = "cleaned_log_embeddings.jsonl"
        self.logs_dir = "../CleanedDaily"
        
    def load_embeddings(self) -> List[Dict[str, Any]]:
        """Load all embeddings from the JSONL file"""
        embeddings_data = []
        try:
            with open(self.embeddings_file, 'r') as f:
                for line in f:
                    if line.strip():
                        embeddings_data.append(json.loads(line))
            print(f"Loaded {len(embeddings_data)} embeddings")
            return embeddings_data
        except FileNotFoundError:
            print(f"Error: {self.embeddings_file} not found")
            sys.exit(1)
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode the user query using the sentence transformer"""
        return self.model.encode(query)
    
    def find_similar_logs(self, query_embedding: np.ndarray, embeddings_data: List[Dict], top_k: int = 3) -> List[Dict]:
        """Find top-k most similar logs using cosine similarity"""
        similarities = []
        
        for log_data in embeddings_data:
            if 'embedding' not in log_data:
                continue
                
            log_embedding = np.array(log_data['embedding'])
            similarity = np.dot(query_embedding, log_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(log_embedding)
            )
            similarities.append((similarity, log_data))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [log_data for _, log_data in similarities[:top_k]]
    
    def load_full_log(self, date: str) -> Dict[str, Any]:
        """Load the full JSON log for a given date"""
        log_file = Path(self.logs_dir) / f"{date}.json"
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Log file for {date} not found at {log_file}")
            return None
    
    def format_prompt(self, query: str, similar_logs: List[Dict]) -> str:
        """Format the prompt with query and matching logs"""
        prompt = f"User asked: {query}\n\n"
        prompt += "Matching logs:\n"
        
        for log_data in similar_logs:
            date = log_data.get('date', 'Unknown')
            full_log = self.load_full_log(date)
            
            if full_log is None:
                prompt += f"Date: {date}\n"
                prompt += "Summary: [Log file not found]\n"
                prompt += "Mood: [Not available]\n"
                prompt += "Tags: [Not available]\n"
                prompt += "Wins: [Not available]\n"
                prompt += "Losses: [Not available]\n\n"
                continue
            
            # Extract fields from full log, don't hallucinate missing ones
            summary = full_log.get('summary', '[Not available]')
            mood = full_log.get('mood', '[Not available]')
            tags = full_log.get('tags', [])
            wins = full_log.get('wins', '[Not available]')
            losses = full_log.get('losses', '[Not available]')
            
            prompt += f"Date: {date}\n"
            prompt += f"Summary: {summary}\n"
            prompt += f"Mood: {mood}\n"
            prompt += f"Tags: {tags}\n"
            prompt += f"Wins: {wins}\n"
            prompt += f"Losses: {losses}\n\n"
        
        return prompt
    
    def query_openai(self, prompt: str) -> str:
        """Send formatted prompt to OpenAI and get response"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a personal reflection assistant. Analyze the provided daily logs and answer the user's question based on the patterns, emotions, and experiences shown in the logs. Be insightful, empathetic, and provide actionable observations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"
    
    def run_query(self, user_query: str) -> str:
        """Main method to run the complete RAG query pipeline"""
        print("Loading embeddings...")
        embeddings_data = self.load_embeddings()
        
        print("Encoding query...")
        query_embedding = self.encode_query(user_query)
        
        print("Finding similar logs...")
        similar_logs = self.find_similar_logs(query_embedding, embeddings_data)
        
        print("Formatting prompt...")
        prompt = self.format_prompt(user_query, similar_logs)
        
        print("Querying OpenAI...")
        response = self.query_openai(prompt)
        
        return response

def main():
    """Main function to run the RAG query engine"""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Initialize the RAG engine
    rag_engine = RAGQueryEngine()
    
    # Get user query
    print("Personal Reflection Assistant")
    print("=" * 40)
    user_query = input("Enter your question about your daily logs: ").strip()
    
    if not user_query:
        print("No query provided. Exiting.")
        return
    
    # Run the query
    print("\nProcessing your query...")
    response = rag_engine.run_query(user_query)
    
    # Display results
    print("\n" + "=" * 40)
    print("RESPONSE:")
    print("=" * 40)
    print(response)

if __name__ == "__main__":
    main() 