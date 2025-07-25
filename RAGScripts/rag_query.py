#!/usr/bin/env python3
"""
RAG Query Script for Personal Reflection Assistant
Handles natural language queries against cleaned daily logs with embeddings
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any, Tuple
import sys
from pathlib import Path

class RAGQueryEngine:
    def __init__(self, llm_backend: str = "ollama", model_name: str = "mistral"):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.llm_backend = llm_backend  # "ollama", "lmstudio", or "openai"
        self.model_name = model_name
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
            
            # Handle different embedding dimensions by truncating or padding
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
    
    def query_llm(self, prompt: str) -> str:
        """Send formatted prompt to LLM (Ollama, LM Studio, or OpenAI) and get response"""
        if self.llm_backend == "ollama":
            return self.query_ollama(prompt)
        elif self.llm_backend == "lmstudio":
            return self.query_lmstudio(prompt)
        else:
            return self.query_openai(prompt)
    
    def query_ollama(self, prompt: str) -> str:
        """Send formatted prompt to Ollama and get response"""
        try:
            url = "http://localhost:11434/api/chat"
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a personal reflection assistant. Analyze the provided daily logs and answer the user's question based on the patterns, emotions, and experiences shown in the logs. Be insightful, empathetic, and provide actionable observations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "No response content")
            
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to Ollama. Make sure Ollama is running with: ollama serve"
        except requests.exceptions.RequestException as e:
            return f"Error querying Ollama: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query_lmstudio(self, prompt: str) -> str:
        """Send formatted prompt to LM Studio and get response"""
        try:
            url = "http://localhost:1234/v1/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a personal reflection assistant. Analyze the provided daily logs and answer the user's question based on the patterns, emotions, and experiences shown in the logs. Be insightful, empathetic, and provide actionable observations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
                "stream": False
            }
            
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.ConnectionError:
            return f"Error: Cannot connect to LM Studio. Make sure LM Studio is running and the local server is started (port 1234)"
        except requests.exceptions.RequestException as e:
            return f"Error querying LM Studio: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query_openai(self, prompt: str) -> str:
        """Send formatted prompt to OpenAI and get response"""
        # Check for OpenAI API key only when actually calling the API
        if not os.getenv("OPENAI_API_KEY"):
            return "Error: OPENAI_API_KEY environment variable not set"
            
        try:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a personal reflection assistant. Analyze the provided daily logs and answer the user's question based on the patterns, emotions, and experiences shown in the logs. Be insightful, empathetic, and provide actionable observations."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"
    
    def run_query(self, user_query: str, debug_mode: bool = False) -> str:
        """Main method to run the complete RAG query pipeline"""
        print("Loading embeddings...")
        embeddings_data = self.load_embeddings()
        
        print("Encoding query...")
        query_embedding = self.encode_query(user_query)
        
        print("Finding similar logs...")
        similar_logs = self.find_similar_logs(query_embedding, embeddings_data)
        
        print("Formatting prompt...")
        prompt = self.format_prompt(user_query, similar_logs)
        
        if debug_mode:
            print("\n" + "=" * 60)
            print("DEBUG MODE - RAW PROMPT (would be sent to OpenAI):")
            print("=" * 60)
            return prompt
        
        print("Querying LLM...")
        response = self.query_llm(prompt)
        
        return response

def main():
    """Main function to run the RAG query engine"""
    # Get LLM choice
    print("Personal Reflection Assistant")
    print("=" * 40)
    print("Choose LLM backend:")
    print("1. Ollama (local, no API key needed)")
    print("2. LM Studio (local, no API key needed)")
    print("3. OpenAI (cloud, requires API key)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        backend = "ollama"
        model = input("Enter Ollama model name (default: mistral): ").strip() or "mistral"
        print(f"Using Ollama with model: {model}")
    elif choice == "2":
        backend = "lmstudio"
        model = input("Enter LM Studio model name (default: local-model): ").strip() or "local-model"
        print(f"Using LM Studio with model: {model}")
    elif choice == "3":
        backend = "openai"
        model = "gpt-4"
        print("Using OpenAI")
    else:
        print("Invalid choice. Using Ollama with mistral.")
        backend = "ollama"
        model = "mistral"
    
    # Initialize the RAG engine
    rag_engine = RAGQueryEngine(llm_backend=backend, model_name=model)
    
    print(f"\n🤖 Chatbot ready! Type 'quit' or 'exit' to end the conversation.")
    print(f"💡 You can ask questions about your daily logs, mood patterns, etc.")
    print("-" * 50)
    
    # Chat loop
    while True:
        try:
            # Get user query
            user_query = input("\nYou: ").strip()
            
            if not user_query:
                continue
                
            if user_query.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Thanks for chatting with your reflection assistant.")
                break
            
            # Check for debug mode command
            debug_mode = False
            if user_query.startswith('/debug'):
                debug_mode = True
                user_query = user_query[7:].strip()  # Remove '/debug' prefix
                if not user_query:
                    print("❌ Please provide a question after /debug")
                    continue
            
            # Run the query
            print("🤔 Thinking...")
            response = rag_engine.run_query(user_query, debug_mode=debug_mode)
            
            # Display results
            print("\n" + "=" * 50)
            if debug_mode:
                print("🔍 DEBUG MODE - RAW PROMPT:")
            else:
                print("🤖 Assistant:")
            print("=" * 50)
            print(response)
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 