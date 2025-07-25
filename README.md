**Product Requirements Document (PRD)**

---

**Project Title:** Personal Self-Awareness App with RAG-Enhanced Reflection

**Version:** 1.0
**Owner:** Ravi (Self-use Only)
**Goal:** To create a mobile-first app that automatically logs behavioral data (phone usage, step count, sleep) and merges it with manual daily emotional logs. The app should support natural language queries via a Retrieval-Augmented Generation (RAG) system, backed by OpenAI or a local model.

---

### 1. **Problem Statement**

People experiencing mood instability (e.g. bipolar, ADHD, burnout) often lack a reliable internal compass to track their state over time. Reflection is hard during low or high states. Manual journaling helps but isn't enough. Behavioral data exists in phones but is rarely interpreted meaningfully.

**Need:** A single place that mirrors emotional + behavioral patterns back to the user.

---

### 2. **Goals**

* Auto-log physical & digital activity (steps, screen time, app usage)
* Merge it with structured daily emotional logs
* Query past patterns using plain language (e.g., "What happened on July 7?" or "Is my Instagram usage going up?")
* Build a RAG system that retrieves actual log data and lets a model generate responses based on real, grounded context
* Prioritize privacy (everything stored locally unless explicitly using OpenAI API)

---

### 3. **Core Features**

#### 3.1 Manual Daily Logging

* Summary (free text)
* Status block:

  * Mood (0–10)
  * Sleep Duration (hours)
  * Sleep Quality (0–10)
  * Energy (0–10)
  * Stability Score (0–10)
* Tags (user-defined)
* Insights (wins, losses, ideas)
* Trigger Events
* Symptom Checklist

#### 3.2 Automatic Data Capture

* Step Count (via Google Fit or Android Sensors)
* Screen Time (total + per-app breakdown)
* App Usage (time spent per app, number of opens)
* Heart Points / Physical Activity
* Optional: Sleep data from health APIs

#### 3.3 Unified Daily Record

Each day becomes one JSON record combining both manual and automatic data:

```json
{
  "date": "2025-07-25",
  "summary": "...",
  "status": { ... },
  "deviceMetrics": {
    "steps": 3421,
    "screenTime": 6.2,
    "appUsage": {
      "Instagram": 1.5,
      "YouTube": 2.3
    },
    "sleep": 6.5,
    "heartPoints": 14
  },
  "tags": [...],
  "insights": { ... },
  ...
}
```

---

### 4. **Query Engine (RAG)**

#### 4.1 Input Types

* Date-based: "What happened on July 7?"
* Tag-based: "Show logs with burnout"
* Pattern: "Is my mood improving this month?"
* Behaviorally reflective: "Is my Instagram use increasing?"
* Emotional diagnosis: "When did I feel suicidal or extremely numb?"

#### 4.2 Retrieval

* On-device search by:

  * Date
  * Tags
  * Mood/energy thresholds
  * Keyword matches in summary/insights
  * Semantic similarity (if using local embeddings)

#### 4.3 Generation

* Retrieved logs are turned into structured prompt
* Sent to OpenAI (or local model like Mistral)
* Returns a natural language response using real data only (no hallucination)

---

### 5. **Architecture Overview**

**Frontend (Android app):**

* Jetpack Compose UI
* Log entry interface
* Behavior graph display (step trends, screen time etc.)
* Query bar + chat-style response

**Backend (on-device):**

* Room DB or flat JSON for logs
* Scheduler to auto-log metrics daily
* Retrieval engine (basic at first, semantic later)
* Prompt builder for model queries

**LLM Layer:**

* OpenAI API for generation (fallback)
* Future option: Local Mistral/Gemma via server or Android-compatible inference

---

### 6. **Privacy + Security**

* All data stays local unless user explicitly uses cloud model
* OpenAI calls will only send retrieved, small context
* No data is uploaded or synced externally

---

### 7. **Non-Goals**

* No Play Store deployment (personal use only)
* Not a habit tracker or therapy app
* No multiplayer or account system

---

### 8. **Roadmap (Phases)**

#### Phase 1: MVP

* Manual logging UI
* Auto metric logger (steps, screen time, app usage)
* JSON file merger per day
* OpenAI RAG: answer "What happened on X day?"

#### Phase 2:

* Smart queries: mood patterns, behavior reflection
* Insights generation from logs
* Local embedding-based semantic search

#### Phase 3:

* Full offline RAG using local model
* Timeline visualizations
* Risk flagging (e.g. mood drops, social isolation)

---

### 9. **Impact**

* Real-time self-awareness
* Grounded reflection in periods of instability
* Emotional validation without external therapy
* Continuity of self across mood episodes

---

*Built not for the world. Built for one person who needs a mirror that doesn’t lie.*

