#!/usr/bin/env python3
"""
RAG to LM Studio Bridge
Formats RAG queries and sends them to LM Studio API
Use this to get RAG context in LM Studio GUI
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any
from pathlib import Path

class RAGToLMStudio:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
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
            print(f"✅ Loaded {len(embeddings_data)} embeddings")
            return embeddings_data
        except FileNotFoundError:
            print(f"❌ Error: {self.embeddings_file} not found")
            return []
    
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
            
            # Handle different embedding dimensions
            if len(query_embedding) != len(log_embedding):
                min_len = min(len(query_embedding), len(log_embedding))
                query_embedding_trimmed = query_embedding[:min_len]
                log_embedding_trimmed = log_embedding[:min_len]
            else:
                query_embedding_trimmed = query_embedding
                log_embedding_trimmed = log_embedding
            
            similarity = np.dot(query_embedding_trimmed, log_embedding_trimmed) / (
                np.linalg.norm(query_embedding_trimmed) * np.linalg.norm(log_embedding_trimmed)
            )
            similarities.append((similarity, log_data))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [log_data for _, log_data in similarities[:top_k]]
    
    def load_full_log(self, date: str) -> Dict[str, Any]:
        """Load the full JSON log for a given date"""
        log_file = Path(self.logs_dir) / f"{date}.json"
        try:
            with open(log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def format_rag_context(self, query: str, similar_logs: List[Dict]) -> str:
        """Format the RAG context for LM Studio"""
        context = f"Based on your daily logs, here's the context for your question: '{query}'\n\n"
        context += "Relevant log entries:\n"
        
        for i, log_data in enumerate(similar_logs, 1):
            date = log_data.get('date', 'Unknown')
            full_log = self.load_full_log(date)
            
            if full_log is None:
                context += f"{i}. Date: {date} - [Log file not found]\n"
                continue
            
            summary = full_log.get('summary', '[Not available]')
            mood = full_log.get('mood', '[Not available]')
            tags = full_log.get('tags', [])
            wins = full_log.get('wins', '[Not available]')
            losses = full_log.get('losses', '[Not available]')
            
            context += f"{i}. Date: {date}\n"
            context += f"   Summary: {summary}\n"
            context += f"   Mood: {mood}\n"
            context += f"   Tags: {tags}\n"
            context += f"   Wins: {wins}\n"
            context += f"   Losses: {losses}\n\n"
        
        context += f"Now please answer the user's question: {query}"
        return context
    
    def send_to_lmstudio(self, context: str) -> str:
        """Send formatted context to LM Studio"""
        try:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": "mistralai/mistral-7b-instruct-v0.3",
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are a personal reflection assistant. Analyze the provided daily logs and answer the user's question based on the patterns, emotions, and experiences shown in the logs. Be insightful, empathetic, and provide actionable observations.\n\n{context}"
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to LM Studio. Make sure it's running with a model loaded."
        except requests.exceptions.HTTPError as e:
            return f"❌ HTTP Error: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def process_query(self, query: str) -> str:
        """Process a query through RAG and send to LM Studio"""
        print("🔍 Loading embeddings...")
        embeddings_data = self.load_embeddings()
        
        if not embeddings_data:
            return "❌ No embeddings found. Please check your data files."
        
        print("🧠 Encoding query...")
        query_embedding = self.encode_query(query)
        
        print("🔎 Finding similar logs...")
        similar_logs = self.find_similar_logs(query_embedding, embeddings_data)
        
        print("📝 Formatting context...")
        context = self.format_rag_context(query, similar_logs)
        
        print("🤖 Sending to LM Studio...")
        response = self.send_to_lmstudio(context)
        
        return response

def main():
    """Main function to process a single query"""
    rag_bridge = RAGToLMStudio()
    
    print("🎯 RAG to LM Studio Bridge")
    print("=" * 40)
    print("This script will:")
    print("1. Find relevant logs for your question")
    print("2. Format them as context")
    print("3. Send to LM Studio for analysis")
    print("=" * 40)
    
    query = input("Enter your question about your daily logs: ").strip()
    
    if not query:
        print("❌ No query provided.")
        return
    
    print("\n🔄 Processing...")
    response = rag_bridge.process_query(query)
    
    print("\n" + "=" * 50)
    print("🤖 LM Studio Response:")
    print("=" * 50)
    print(response)

if __name__ == "__main__":
    main() 