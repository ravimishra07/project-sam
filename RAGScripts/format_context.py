#!/usr/bin/env python3
"""
Format RAG context for copy-pasting into LM Studio GUI
No terminal interaction needed - just copy the output into LM Studio
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path

class ContextFormatter:
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
            return embeddings_data
        except FileNotFoundError:
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
    
    def format_context(self, query: str) -> str:
        """Format context for copy-pasting into LM Studio"""
        print("🔍 Loading embeddings...")
        embeddings_data = self.load_embeddings()
        
        if not embeddings_data:
            return "❌ No embeddings found."
        
        print("🧠 Encoding query...")
        query_embedding = self.encode_query(query)
        
        print("🔎 Finding similar logs...")
        similar_logs = self.find_similar_logs(query_embedding, embeddings_data)
        
        print("📝 Formatting context...")
        
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

def main():
    """Format context for copy-pasting"""
    formatter = ContextFormatter()
    
    print("📋 Context Formatter for LM Studio")
    print("=" * 40)
    print("This will format your log context for copy-pasting into LM Studio GUI")
    print("=" * 40)
    
    query = input("Enter your question about your daily logs: ").strip()
    
    if not query:
        print("❌ No query provided.")
        return
    
    context = formatter.format_context(query)
    
    print("\n" + "=" * 60)
    print("📋 COPY THIS CONTEXT INTO LM STUDIO:")
    print("=" * 60)
    print(context)
    print("=" * 60)
    print("\n💡 Instructions:")
    print("1. Copy the text above")
    print("2. Paste it into LM Studio chat")
    print("3. Ask your question normally in LM Studio")
    print("4. The model will have context from your logs!")

if __name__ == "__main__":
    main() 