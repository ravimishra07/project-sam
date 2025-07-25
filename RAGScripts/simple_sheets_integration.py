#!/usr/bin/env python3
"""
Simple Google Sheets Integration for RAG System
Works with CSV exports from Google Sheets
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any
import re
from pathlib import Path

class SimpleSheetsRAG:
    def __init__(self):
        self.data_dir = "google_sheets_data"
        self.output_dir = "RAGScripts"
        
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
    
    def process_csv_file(self, filepath: str, tab_name: str) -> List[Dict[str, Any]]:
        """Process a CSV file and format for RAG"""
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Loaded {len(df)} rows from {tab_name}")
            
            formatted_entries = []
            
            for idx, row in df.iterrows():
                entry = {
                    'source': f'google_sheets_{tab_name.lower().replace(" ", "_")}',
                    'row_id': idx + 1
                }
                
                # Add all non-empty columns
                for col in df.columns:
                    value = self.clean_text(row.get(col, ''))
                    if value:
                        entry[col.lower().replace(' ', '_')] = value
                
                # Only add if there's meaningful content
                if len(entry) > 2:  # More than just source and row_id
                    formatted_entries.append(entry)
            
            return formatted_entries
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {str(e)}")
            return []
    
    def process_all_csv_files(self) -> List[Dict[str, Any]]:
        """Process all CSV files in the data directory"""
        all_entries = []
        
        if not os.path.exists(self.data_dir):
            print(f"📁 Creating data directory: {self.data_dir}")
            os.makedirs(self.data_dir)
            print(f"💡 Please export your Google Sheets tabs as CSV files to: {self.data_dir}")
            print(f"   Expected files:")
            print(f"   - {self.data_dir}/samdaily.csv")
            print(f"   - {self.data_dir}/energy_modes.csv")
            print(f"   - {self.data_dir}/prompts.csv")
            print(f"   - {self.data_dir}/sheet6.csv")
            print(f"   - {self.data_dir}/video.csv")
            return []
        
        # Process each CSV file
        csv_files = {
            'samdaily': 'samdaily.csv',
            'energy_modes': 'energy_modes.csv', 
            'prompts': 'prompts.csv',
            'sheet6': 'sheet6.csv',
            'video': 'video.csv'
        }
        
        for tab_name, filename in csv_files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                print(f"📊 Processing: {filename}")
                entries = self.process_csv_file(filepath, tab_name)
                all_entries.extend(entries)
                print(f"✅ Added {len(entries)} entries from {tab_name}")
            else:
                print(f"⚠️  File not found: {filepath}")
        
        return all_entries
    
    def save_to_json(self, entries: List[Dict[str, Any]], filename: str = "google_sheets_data.json"):
        """Save formatted data to JSON file"""
        output_path = os.path.join(self.output_dir, filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved {len(entries)} entries to {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving to JSON: {str(e)}")
            return None
    
    def create_embeddings_file(self, entries: List[Dict[str, Any]], filename: str = "google_sheets_embeddings.jsonl"):
        """Create embeddings file for RAG integration"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            
            output_path = os.path.join(self.output_dir, filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(entries):
                    # Create a text representation for embedding
                    text_parts = []
                    
                    # Add all fields except metadata
                    for key, value in entry.items():
                        if key not in ['source', 'row_id']:
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
            
            print(f"✅ Created embeddings file: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error creating embeddings: {str(e)}")
            return None
    
    def merge_with_existing_embeddings(self, new_embeddings_file: str):
        """Merge new embeddings with existing ones"""
        existing_file = os.path.join(self.output_dir, "cleaned_log_embeddings.jsonl")
        merged_file = os.path.join(self.output_dir, "combined_embeddings.jsonl")
        
        try:
            all_embeddings = []
            
            # Load existing embeddings
            if os.path.exists(existing_file):
                with open(existing_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_embeddings.append(json.loads(line))
                print(f"✅ Loaded {len(all_embeddings)} existing embeddings")
            
            # Load new embeddings
            if os.path.exists(new_embeddings_file):
                with open(new_embeddings_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            all_embeddings.append(json.loads(line))
                print(f"✅ Added Google Sheets embeddings")
            
            # Save combined embeddings
            with open(merged_file, 'w') as f:
                for entry in all_embeddings:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"✅ Created combined embeddings file: {merged_file}")
            print(f"📊 Total embeddings: {len(all_embeddings)}")
            
            return merged_file
            
        except Exception as e:
            print(f"❌ Error merging embeddings: {str(e)}")
            return None

def main():
    """Main function to process Google Sheets CSV data"""
    print("🔗 Simple Google Sheets RAG Integration")
    print("=" * 50)
    
    # Initialize
    sheets_rag = SimpleSheetsRAG()
    
    # Process all CSV files
    print("🔄 Processing CSV files...")
    entries = sheets_rag.process_all_csv_files()
    
    if not entries:
        print("❌ No data found. Please export your Google Sheets as CSV files.")
        print("\n💡 How to export from Google Sheets:")
        print("   1. Open your Google Sheet")
        print("   2. Go to File > Download > CSV")
        print("   3. Save each tab as a separate CSV file")
        print("   4. Place them in the 'google_sheets_data' folder")
        return
    
    print(f"\n✅ Processed {len(entries)} total entries")
    
    # Save to JSON
    json_file = sheets_rag.save_to_json(entries)
    
    # Create embeddings
    embeddings_file = sheets_rag.create_embeddings_file(entries)
    
    if embeddings_file:
        # Merge with existing embeddings
        combined_file = sheets_rag.merge_with_existing_embeddings(embeddings_file)
        
        print(f"\n🎉 Integration complete!")
        print(f"📁 Data saved to: {json_file}")
        print(f"🧠 Embeddings saved to: {embeddings_file}")
        print(f"🔗 Combined embeddings: {combined_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Update your RAG scripts to use the combined embeddings")
        print(f"   2. Test the integration with your Streamlit app")
        print(f"   3. Your Google Sheets data is now part of your RAG system!")

if __name__ == "__main__":
    main() 