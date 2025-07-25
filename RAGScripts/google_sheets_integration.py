#!/usr/bin/env python3
"""
Google Sheets Integration for RAG System
Reads, cleans, and integrates Google Sheets data into the RAG pipeline
"""

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import re

class GoogleSheetsRAG:
    def __init__(self):
        self.sheet_id = "1Ur9cMT9nIu5DbhefgUBC7RRJ2ejx4MdVg_oW2cBjEfo"
        self.scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        self.credentials = None
        self.client = None
        
    def setup_credentials(self, credentials_file: str = None):
        """Setup Google Sheets credentials"""
        try:
            if credentials_file and os.path.exists(credentials_file):
                # Use service account credentials
                self.credentials = Credentials.from_service_account_file(
                    credentials_file, scopes=self.scope
                )
            else:
                # Try to use default credentials (if you're logged in)
                print("⚠️  No credentials file found. You'll need to authenticate manually.")
                print("💡 To use service account:")
                print("   1. Go to Google Cloud Console")
                print("   2. Create a service account")
                print("   3. Download JSON credentials")
                print("   4. Pass the file path to setup_credentials()")
                return False
                
            self.client = gspread.authorize(self.credentials)
            return True
        except Exception as e:
            print(f"❌ Error setting up credentials: {str(e)}")
            return False
    
    def read_sheet_tab(self, tab_name: str) -> pd.DataFrame:
        """Read a specific tab from the Google Sheet"""
        try:
            sheet = self.client.open_by_key(self.sheet_id)
            worksheet = sheet.worksheet(tab_name)
            
            # Get all values
            data = worksheet.get_all_records()
            
            if not data:
                print(f"⚠️  No data found in tab: {tab_name}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            print(f"✅ Loaded {len(df)} rows from tab: {tab_name}")
            return df
            
        except Exception as e:
            print(f"❌ Error reading tab {tab_name}: {str(e)}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text
    
    def format_samdaily_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format samdaily tab data for RAG"""
        formatted_entries = []
        
        for idx, row in df.iterrows():
            # Clean and extract data
            entry = {
                'date': self.clean_text(row.get('Date', '')),
                'summary': self.clean_text(row.get('Summary', '')),
                'mood': self.clean_text(row.get('Mood', '')),
                'energy': self.clean_text(row.get('Energy', '')),
                'tags': self.clean_text(row.get('Tags', '')),
                'wins': self.clean_text(row.get('Wins', '')),
                'losses': self.clean_text(row.get('Losses', '')),
                'notes': self.clean_text(row.get('Notes', '')),
                'source': 'google_sheets_samdaily'
            }
            
            # Only add if there's meaningful content
            if entry['summary'] or entry['notes']:
                formatted_entries.append(entry)
        
        return formatted_entries
    
    def format_energy_modes_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format energy modes data for RAG"""
        formatted_entries = []
        
        for idx, row in df.iterrows():
            entry = {
                'date': self.clean_text(row.get('Date', '')),
                'energy_mode': self.clean_text(row.get('Energy Mode', '')),
                'description': self.clean_text(row.get('Description', '')),
                'triggers': self.clean_text(row.get('Triggers', '')),
                'coping_strategies': self.clean_text(row.get('Coping Strategies', '')),
                'source': 'google_sheets_energy_modes'
            }
            
            if entry['energy_mode'] or entry['description']:
                formatted_entries.append(entry)
        
        return formatted_entries
    
    def format_prompts_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format prompts data for RAG"""
        formatted_entries = []
        
        for idx, row in df.iterrows():
            entry = {
                'prompt_type': self.clean_text(row.get('Prompt Type', '')),
                'prompt_text': self.clean_text(row.get('Prompt Text', '')),
                'response': self.clean_text(row.get('Response', '')),
                'date': self.clean_text(row.get('Date', '')),
                'source': 'google_sheets_prompts'
            }
            
            if entry['prompt_text'] or entry['response']:
                formatted_entries.append(entry)
        
        return formatted_entries
    
    def format_generic_data(self, df: pd.DataFrame, tab_name: str) -> List[Dict[str, Any]]:
        """Format generic tab data for RAG"""
        formatted_entries = []
        
        for idx, row in df.iterrows():
            # Create a generic entry with all non-empty columns
            entry = {
                'source': f'google_sheets_{tab_name.lower().replace(" ", "_")}',
                'row_id': idx + 1
            }
            
            for col in df.columns:
                value = self.clean_text(row.get(col, ''))
                if value:
                    entry[col.lower().replace(' ', '_')] = value
            
            if len(entry) > 2:  # More than just source and row_id
                formatted_entries.append(entry)
        
        return formatted_entries
    
    def process_all_tabs(self) -> List[Dict[str, Any]]:
        """Process all tabs and return formatted data"""
        all_entries = []
        
        # Define tab processors
        tab_processors = {
            'samdaily': self.format_samdaily_data,
            'SAM- Energy Modes': self.format_energy_modes_data,
            'promptsSam with schema': self.format_prompts_data,
            'Sheet6': self.format_generic_data,
            'Video': self.format_generic_data
        }
        
        for tab_name, processor in tab_processors.items():
            print(f"📊 Processing tab: {tab_name}")
            df = self.read_sheet_tab(tab_name)
            
            if not df.empty:
                if tab_name in ['Sheet6', 'Video']:
                    entries = processor(df, tab_name)
                else:
                    entries = processor(df)
                
                all_entries.extend(entries)
                print(f"✅ Added {len(entries)} entries from {tab_name}")
        
        return all_entries
    
    def save_to_json(self, entries: List[Dict[str, Any]], filename: str = "google_sheets_data.json"):
        """Save formatted data to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(entries)} entries to {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving to JSON: {str(e)}")
            return None
    
    def create_embeddings_file(self, entries: List[Dict[str, Any]], filename: str = "google_sheets_embeddings.jsonl"):
        """Create embeddings file for RAG integration"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            
            with open(filename, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(entries):
                    # Create a text representation for embedding
                    text_parts = []
                    
                    if 'summary' in entry:
                        text_parts.append(f"Summary: {entry['summary']}")
                    if 'notes' in entry:
                        text_parts.append(f"Notes: {entry['notes']}")
                    if 'description' in entry:
                        text_parts.append(f"Description: {entry['description']}")
                    if 'prompt_text' in entry:
                        text_parts.append(f"Prompt: {entry['prompt_text']}")
                    if 'response' in entry:
                        text_parts.append(f"Response: {entry['response']}")
                    
                    # Add all other fields
                    for key, value in entry.items():
                        if key not in ['summary', 'notes', 'description', 'prompt_text', 'response', 'source', 'row_id']:
                            text_parts.append(f"{key}: {value}")
                    
                    text = " | ".join(text_parts)
                    
                    if text.strip():
                        # Generate embedding
                        embedding = model.encode(text).tolist()
                        
                        # Create entry for embeddings file
                        embedding_entry = {
                            'id': f"gs_{i}",
                            'text': text,
                            'embedding': embedding,
                            'metadata': entry
                        }
                        
                        f.write(json.dumps(embedding_entry, ensure_ascii=False) + '\n')
            
            print(f"✅ Created embeddings file: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            return None

def main():
    """Main function to process Google Sheets data"""
    print("🔗 Google Sheets RAG Integration")
    print("=" * 50)
    
    # Initialize
    gs_rag = GoogleSheetsRAG()
    
    # Setup credentials (you'll need to provide the credentials file)
    credentials_file = input("Enter path to Google service account JSON file (or press Enter to skip): ").strip()
    
    if credentials_file and os.path.exists(credentials_file):
        if gs_rag.setup_credentials(credentials_file):
            print("✅ Credentials loaded successfully")
        else:
            print("❌ Failed to load credentials")
            return
    else:
        print("⚠️  No credentials file provided. You'll need to set up authentication manually.")
        print("💡 See the setup_credentials() method for instructions.")
        return
    
    # Process all tabs
    print("\n🔄 Processing Google Sheets data...")
    entries = gs_rag.process_all_tabs()
    
    if not entries:
        print("❌ No data found or processed")
        return
    
    print(f"\n✅ Processed {len(entries)} total entries")
    
    # Save to JSON
    json_file = gs_rag.save_to_json(entries)
    
    # Create embeddings
    embeddings_file = gs_rag.create_embeddings_file(entries)
    
    print(f"\n🎉 Integration complete!")
    print(f"📁 Data saved to: {json_file}")
    print(f"🧠 Embeddings saved to: {embeddings_file}")
    print(f"\n💡 Next steps:")
    print(f"   1. Copy the embeddings file to your RAGScripts folder")
    print(f"   2. Update your RAG scripts to include Google Sheets data")
    print(f"   3. Test the integration with your Streamlit app")

if __name__ == "__main__":
    main() 