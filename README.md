# Student Pass/Fail Predictor with AI Study Coach

**Live Demo:** [https://studentpass-fapredictingmlmodel-cr9r6kmp3swsgnsmqn6dzk.streamlit.app/](https://studentpass-fapredictingmlmodel-cr9r6kmp3swsgnsmqn6dzk.streamlit.app/)

A two-page Streamlit application that predicts whether a student will pass or fail using a pre-trained ML pipeline, then connects them to a personalised AI Study Coach powered by LangGraph, Groq LLM, ChromaDB RAG, and Tavily web search.

---

## Features

**Dashboard (Page 1)**
- Enter 9 academic and wellbeing metrics via sliders
- Instant Pass / Fail prediction with a pass-probability bar
- Subject score breakdown with visual progress bars
- Personalised actionable tips based on detected weak areas
- One-click navigation to the Study Coach

**AI Study Coach (Page 2)**
- Conversational chat powered by `llama-3.1-8b-instant` (Groq)
- LangGraph agent pipeline: route → diagnose → plan → retrieve → respond → memory
- Generates a personalised 7-day study plan based on the student's profile
- RAG retrieval from a ChromaDB knowledge base of 25 curated study tips
- Live web search via Tavily for up-to-date resources and links
- Restricted to study-related topics — off-topic questions are politely declined
- Chat session resets automatically when a new prediction is made

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.54, custom CSS (glassmorphism dark theme) |
| ML — Classification | XGBoost + Logistic Regression selected via GridSearchCV |
| ML — Clustering | KMeans (scikit-learn) |
| ML — Preprocessing | MinMaxScaler, SMOTE (imbalanced-learn), IQR outlier removal |
| Agent Orchestration | LangGraph `StateGraph` |
| LLM | Groq API — `llama-3.1-8b-instant` via `langchain-groq` |
| RAG Vector Store | ChromaDB via `langchain-chroma` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` |
| Web Search | Tavily Python client |
| Serialisation | joblib |

---

## Project Structure

```
student_pass-fail_predicting_ml_model/
│
├── app.py                        # Streamlit entry point (multi-page navigation)
├── train_model.py                # Training pipeline script
│
├── data/
│   └── student_data.csv          # Training dataset (9 features + pass_fail label)
│
├── models/                       # Saved artifacts after training (git-ignored)
│   ├── model.pkl                 # Best classifier
│   ├── scaler.pkl                # Fitted MinMaxScaler
│   ├── kmeans.pkl                # Fitted KMeans
│   └── meta.pkl                  # Training metadata (scores, feature names)
│
├── src/
│   ├── config.py                 # Paths and feature column list
│   ├── data_processor.py         # Load, clean, scale, split, SMOTE
│   ├── model_trainer.py          # Train, evaluate, save classifiers + KMeans
│   ├── inference.py              # StudentPredictor — predict + recommendations
│   └── ui_components.py          # UIBuilder — CSS, cards, score bars, tip cards
│
├── agents/
│   ├── state.py                  # AgentState TypedDict (LangGraph state schema)
│   ├── nodes.py                  # 6 LangGraph nodes (Router → Diagnosis → Planner → Retriever → Response → Memory)
│   └── study_coach_agent.py      # StudyCoachAgent — builds and runs the StateGraph
│
├── tools/
│   ├── rag_tool.py               # RAGTool — ChromaDB vector store seeded with 25 study tips
│   └── web_search_tool.py        # WebSearchTool — Tavily live web search
│
├── memory/
│   └── session_memory.py         # SessionMemory helper
│
├── pages/
│   ├── dashboard.py              # Page 1 — prediction form and results
│   └── chat_interface.py         # Page 2 — AI Study Coach chat
│
├── .streamlit/
│   └── config.toml               # Disables file watcher (suppresses torch warnings)
│
├── requirements.txt
└── .env                          # API keys (git-ignored)
```

---

## Dataset Features

| Feature | Description |
|---|---|
| `parental_education_level` | 1 (no formal education) → 7 (postgraduate) |
| `daily_study_hours` | Hours studied per day |
| `attendance_rate` | 0.0 → 1.0 (proportion of classes attended) |
| `sleep_hours` | Hours of sleep per night |
| `stress_level` | 1 (very low) → 10 (extremely high) |
| `motivation_score` | 0 → 100 |
| `math_score` | 0 → 100 |
| `reading_score` | 0 → 100 |
| `writing_score` | 0 → 100 |
| `pass_fail` | Target label — "Pass" or "Fail" |

---

## LangGraph Agent Pipeline

```
User message
     │
     ▼
 RouterNode          classifies query: STUDY or GENERAL
     │
     ├─ GENERAL ──► ResponseGeneratorNode   (politely declines off-topic questions)
     │
     └─ STUDY ───► DiagnosisNode            (identifies weak areas from student profile)
                        │
                        ▼
                   PlannerNode              (generates personalised 7-day study plan via Groq)
                        │
                        ▼
                   ResourceRetrieverNode    (RAG from ChromaDB + live Tavily web search)
                        │
                        ▼
                   ResponseGeneratorNode    (assembles final reply with plan + resource links)
                        │
                        ▼
                   MemoryNode              (appends turn to session_history)
                        │
                        ▼
                      END
```

### LangChain / LangGraph usage by file

| File | What it uses |
|---|---|
| `agents/state.py` | `BaseMessage`, `Annotated` list with `operator.add` reducer |
| `agents/nodes.py` | `ChatGroq` (langchain-groq), `HumanMessage`, `AIMessage`, `SystemMessage` |
| `agents/study_coach_agent.py` | `StateGraph`, `add_edge`, `add_conditional_edges`, `compile()` |
| `tools/rag_tool.py` | `Chroma` (langchain-chroma), `HuggingFaceEmbeddings`, `Document` |

---

## Model Performance

Best model selected by GridSearchCV: **Logistic Regression**

| Metric | Score |
|---|---|
| Accuracy | 1.00 (100%) |
| Precision | 1.00 (100%) |
| Recall | 1.00 (100%) |
| F1 Score | 1.00 (100%) |

> **Why are all metrics 1.0?**
> The dataset used is synthetic — the `pass_fail` label is deterministically derived from the average of `math_score`, `reading_score`, and `writing_score`. Because the target is a direct function of input features, the model learns a perfect rule and achieves 100% on all metrics. On a real-world dataset with noisy, independently collected labels, these scores would be lower and more representative.

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd student_pass-fail_predicting_ml_model
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- Free Groq key: [console.groq.com](https://console.groq.com)
- Free Tavily key: [tavily.com](https://tavily.com)

### 3. Train the model

```bash
python train_model.py
```

Runs the 8-step pipeline: load → fill nulls → remove outliers → scale → SMOTE → train classifiers → KMeans → save artifacts to `models/`.

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## ML Training Pipeline

Run via `python train_model.py`. Eight sequential steps:

### Step 1 — Load data
Reads `data/student_data.csv` using pandas and prints the row count and column list.

### Step 2 — Fill null values (`fill_nulls`)
For each of the 9 feature columns, missing values are filled with that column's **median**. Median is used instead of mean to be robust against skewed distributions.

### Step 3 — Remove outliers (`remove_outliers`)
Applies the **IQR method** to each feature column:
- Computes Q1 (25th percentile) and Q3 (75th percentile)
- Removes rows where a value falls outside `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`
- Skips columns with IQR = 0 to avoid dropping all rows for constant-value columns

### Step 4 — Preprocess features and target (`preprocess`)
- Encodes the target: `pass_fail` → 1 (Pass) or 0 (Fail)
- Prints class distribution (Pass % vs Fail %)
- Applies a **1%–99% percentile clip** on each feature to suppress extreme values before scaling
- Scales all 9 features to [0, 1] using **MinMaxScaler** (`fit_transform`)

### Step 5 — Split and balance (`split_and_balance`)
- Splits data into **80% train / 20% test** with `stratify=y` to preserve class proportions
- Applies **SMOTE** (Synthetic Minority Over-sampling Technique) to the training set only, so the minority class (Fail or Pass) is synthetically oversampled to balance the training distribution
- The test set is never touched by SMOTE — it stays as real data

### Step 6 — Train and evaluate classifiers (`train_and_evaluate`)
Two candidate models are trained using **GridSearchCV** with 3-fold Stratified K-Fold cross-validation, scored by weighted F1:

| Model | Hyperparameters searched |
|---|---|
| Logistic Regression | `C ∈ {0.1, 1.0, 10.0}` |
| XGBoost | `n_estimators ∈ {50, 100}`, `max_depth ∈ {3, 6}`, `learning_rate ∈ {0.05, 0.1}` |

The model with the highest weighted F1 on the test set is selected as the best model.

### Step 7 — KMeans clustering (`train_kmeans`)
Fits a **KMeans** model with 3 clusters on the scaled training data. Clusters group students into performance bands (e.g. high achievers, average, at-risk) and are used in the dashboard to show which group a student belongs to.

### Step 8 — Save artifacts (`save_artifacts`)
Saves all trained objects to the `models/` directory using joblib:

| File | Contents |
|---|---|
| `model.pkl` | Best trained classifier |
| `scaler.pkl` | Fitted MinMaxScaler |
| `kmeans.pkl` | Fitted KMeans |
| `meta.pkl` | Best model name + accuracy, precision, recall, F1 scores |
| `feature_names.pkl` | Scaled feature names array |
| `input_feature_cols.pkl` | List of input feature column names |

---

## Author

**Yash Agarwal**  
Machine Learning · AI Agents · Python Development
