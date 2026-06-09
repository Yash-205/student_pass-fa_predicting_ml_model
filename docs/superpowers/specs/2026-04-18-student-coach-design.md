# Student Pass/Fail Predictor + AI Study Coach — Design Spec
Date: 2026-04-18

## Overview
A Streamlit web app where a student enters their academic details, an ML model predicts Pass/Fail, and a LangGraph AI agent acts as a personalized study coach.

## M1 — ML Pipeline

### Features (all numeric)
`parental_education_level`, `daily_study_hours`, `attendance_rate`, `sleep_hours`, `stress_level`, `motivation_score`, `math_score`, `reading_score`, `writing_score`

### Target
`pass_fail` column → Pass=1, Fail=0

### Preprocessing Steps
1. Fill nulls with column median
2. IQR outlier removal on all 9 features
3. Clip at 1st/99th percentile
4. MinMaxScaler
5. SMOTE for class balancing
6. Train/test split (80/20, stratified)

### Models
- LogisticRegression (GridSearchCV on C)
- XGBoost (GridSearchCV on n_estimators, max_depth, learning_rate)
- Best model selected by F1 weighted
- KMeans (k=3) for student segmentation

### Artifacts saved to `models/`
`model.pkl`, `scaler.pkl`, `kmeans.pkl`, `meta.pkl`, `input_feature_cols.pkl`

## M2 — Streamlit App

### Pages (via st.navigation)
1. **Dashboard** (`pages/dashboard.py`) — Student enters 9 values → prediction card + charts + "Chat with Coach" button
2. **Study Coach** (`pages/chat_interface.py`) — LangGraph chat agent

### LangGraph Agent (Approach A: Linear + general chat router)

```
user_message → router → [diagnose → plan → retrieve →] respond → memory → END
                   ↑ general chat skips diagnose/plan/retrieve
```

**Nodes (agents/nodes.py):**
- `RouterNode` — classifies intent via Groq LLM
- `DiagnosisNode` — finds weak areas from student data + prediction
- `PlannerNode` — generates weekly study plan via Groq LLM
- `ResourceRetrieverNode` — ChromaDB RAG + Tavily web search
- `ResponseGeneratorNode` — final conversational reply via Groq LLM
- `MemoryNode` — saves turn to st.session_state

**Tools:**
- `tools/web_search_tool.py` — Tavily client
- `tools/rag_tool.py` — ChromaDB + HuggingFace sentence-transformers, seeded with ~25 study strategy docs

**State (agents/state.py):** AgentState TypedDict with messages, student_data, learning_gaps, study_plan, retrieved_resources, web_links, session_history, is_study_query

**Memory (memory/session_memory.py):** SessionMemory keyed by student_id, persisted in st.session_state

## File Structure
```
agents/state.py, nodes.py, study_coach_agent.py
tools/web_search_tool.py, rag_tool.py
memory/session_memory.py
src/config.py, data_processor.py, model_trainer.py, inference.py
pages/dashboard.py, chat_interface.py
app.py
train_model.py
requirements.txt
.env
```

## API Keys
- Groq: via GROQ_API_KEY env var
- Tavily: via TAVILY_API_KEY env var
- HuggingFace: via HUGGINGFACE_TOKEN env var (for embeddings download)
