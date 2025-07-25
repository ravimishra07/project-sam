#!/usr/bin/env python3
"""
RAG UI - Streamlit Web App
Beautiful GUI for querying your personal logs with LM Studio
"""

import streamlit as st
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict, Any
from pathlib import Path
import time

class RAGEngine:
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
    
    def query_lmstudio(self, context: str) -> str:
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
    
    def process_query(self, query: str) -> tuple[str, List[Dict]]:
        """Process a query and return response with used logs"""
        embeddings_data = self.load_embeddings()
        
        if not embeddings_data:
            return "❌ No embeddings found. Please check your data files.", []
        
        query_embedding = self.encode_query(query)
        similar_logs = self.find_similar_logs(query_embedding, embeddings_data)
        
        # Format context
        context = f"Based on your daily logs, here's the context for your question: '{query}'\n\n"
        context += "Relevant log entries:\n"
        
        used_logs = []
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
            
            used_logs.append({
                'date': date,
                'summary': summary,
                'mood': mood,
                'tags': tags,
                'wins': wins,
                'losses': losses
            })
        
        context += f"Now please answer the user's question: {query}"
        
        response = self.query_lmstudio(context)
        return response, used_logs

def main():
    st.set_page_config(
        page_title="Personal Reflection Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .log-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .response-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤖 Personal Reflection Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your daily logs and get insights from your personal data</p>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = RAGEngine()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Check LM Studio connection
        try:
            response = requests.get("http://localhost:1234/v1/models", timeout=5)
            if response.status_code == 200:
                st.success("✅ LM Studio Connected")
                models = response.json()
                if 'data' in models and models['data']:
                    st.info(f"📋 Available models: {len(models['data'])}")
            else:
                st.error("❌ LM Studio not responding")
        except:
            st.error("❌ Cannot connect to LM Studio")
            st.info("💡 Make sure LM Studio is running with a model loaded")
        
        st.divider()
        
        # Example questions
        st.header("💡 Example Questions")
        examples = [
            "When was I most happy?",
            "What are my mood patterns?",
            "Tell me about my relationship with Tanvi",
            "What were my biggest wins recently?",
            "When did I feel stressed or anxious?"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{example}"):
                st.session_state.query = example
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask About Your Logs")
        
        # Query input
        query = st.text_input(
            "What would you like to know about your daily logs?",
            value=st.session_state.get('query', ''),
            placeholder="e.g., When was I most happy? What are my mood patterns?"
        )
        
        # Process button
        if st.button("🔍 Analyze My Logs", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("🔍 Searching your logs..."):
                    time.sleep(1)
                
                with st.spinner("🧠 Analyzing with AI..."):
                    response, used_logs = st.session_state.rag_engine.process_query(query)
                
                # Display response
                st.markdown('<div class="response-box">', unsafe_allow_html=True)
                st.markdown("### 🤖 AI Response")
                st.write(response)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store in session state for the other column
                st.session_state.last_response = response
                st.session_state.used_logs = used_logs
            else:
                st.warning("Please enter a question.")
    
    with col2:
        st.header("📋 Used Logs")
        
        if 'used_logs' in st.session_state and st.session_state.used_logs:
            for i, log in enumerate(st.session_state.used_logs, 1):
                with st.expander(f"📅 {log['date']}", expanded=True):
                    st.markdown(f"**Summary:** {log['summary']}")
                    st.markdown(f"**Mood:** {log['mood']}")
                    st.markdown(f"**Tags:** {', '.join(log['tags']) if log['tags'] else 'None'}")
                    if log['wins'] != '[Not available]':
                        st.markdown(f"**Wins:** {log['wins']}")
                    if log['losses'] != '[Not available]':
                        st.markdown(f"**Losses:** {log['losses']}")
        else:
            st.info("Ask a question to see which logs were used for the analysis.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by LM Studio + RAG | Your personal reflection assistant</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 