# Google Sheets Integration Setup

This guide will help you integrate your Google Sheets data into the RAG system.

## 📋 What this does:

- **Reads your Google Sheets data** (unstructured format)
- **Cleans and formats** the data for RAG
- **Creates embeddings** for semantic search
- **Merges with existing logs** for comprehensive analysis

## 🚀 Quick Setup (Recommended):

### Step 1: Export Google Sheets as CSV

1. **Open your Google Sheet**: https://docs.google.com/spreadsheets/d/1Ur9cMT9nIu5DbhefgUBC7RRJ2ejx4MdVg_oW2cBjEfo/edit

2. **Export each tab as CSV**:
   - Click on each tab (samdaily, SAM- Energy Modes, etc.)
   - Go to **File > Download > CSV**
   - Save with these names:
     - `samdaily.csv`
     - `energy_modes.csv`
     - `prompts.csv`
     - `sheet6.csv`
     - `video.csv`

3. **Place CSV files** in the `google_sheets_data` folder (will be created automatically)

### Step 2: Run the Integration

```bash
cd RAGScripts
python3 simple_sheets_integration.py
```

### Step 3: Update Your RAG Scripts

The script will create `combined_embeddings.jsonl` that includes both your daily logs and Google Sheets data.

## 🔧 Advanced Setup (Direct API Access):

If you want real-time access to Google Sheets:

1. **Create Google Cloud Project**:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable Google Sheets API

2. **Create Service Account**:
   - Go to IAM & Admin > Service Accounts
   - Create new service account
   - Download JSON credentials file

3. **Share your Google Sheet**:
   - Share with the service account email (found in JSON file)
   - Give "Editor" permissions

4. **Run the advanced script**:
   ```bash
   python3 google_sheets_integration.py
   ```

## 📊 What Data Gets Processed:

### samdaily Tab:
- Date, Summary, Mood, Energy, Tags, Wins, Losses, Notes

### SAM- Energy Modes Tab:
- Energy modes, descriptions, triggers, coping strategies

### promptsSam with schema Tab:
- Prompt types, text, responses, dates

### Sheet6 & Video Tabs:
- All columns processed generically

## 🎯 Benefits:

✅ **Unified Search**: Search across all your data sources
✅ **Personal Insights**: Get AI analysis of your complete data
✅ **Real-time Updates**: Keep your RAG system current
✅ **Structured Analysis**: Clean, searchable format

## 🔍 Example Queries After Integration:

- "What are my energy patterns?"
- "Show me all my wins from Google Sheets"
- "What prompts have I used recently?"
- "Tell me about my mood and energy correlation"

## 🛠️ Troubleshooting:

**No CSV files found**:
- Make sure you exported each tab separately
- Check file names match exactly
- Ensure files are in the `google_sheets_data` folder

**Embedding errors**:
- Make sure sentence-transformers is installed
- Check that CSV files have readable content

**Permission errors**:
- For API access, ensure service account has access to the sheet
- For CSV files, check file permissions

## 📈 Next Steps:

1. **Test the integration** with your Streamlit app
2. **Ask questions** about your combined data
3. **Update regularly** by re-running the integration script
4. **Customize** the data processing for your specific needs

Your Google Sheets data will now be part of your personal reflection assistant! 🎉 