# Architectural Data Pipeline
### Student Pass/Fail Prediction System — End-to-End Data Flow

---

```
┌─────────────────────────────────────────────────────────────┐
│                     📂 RAW INPUT DATA                       │
│              data/student_data.csv  (~1,000,000 rows)       │
│                                                             │
│  Features (9):                                              │
│    parental_education_level  (1–7, ordinal)                 │
│    daily_study_hours         (0–12 h/day)                   │
│    attendance_rate           (0.0–1.0)                      │
│    sleep_hours               (3–12 h/night)                 │
│    stress_level              (1–10)                         │
│    motivation_score          (0–100)                        │
│    math_score                (0–100)                        │
│    reading_score             (0–100)                        │
│    writing_score             (0–100)                        │
│                                                             │
│  Target: pass_fail  ("Pass" | "Fail")                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  🧹 STEP 1: NULL HANDLING                   │
│                 DataProcessor.fill_nulls()                  │
│                                                             │
│  For each of the 9 feature columns:                         │
│    df[col].fillna(df[col].median())                         │
│  → Missing values replaced with per-column median           │
│  → Result: 0 null values remain                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               📊 STEP 2: OUTLIER REMOVAL (IQR)              │
│              DataProcessor.remove_outliers()                │
│                                                             │
│  For each of the 9 feature columns:                         │
│    Lower Bound = Q1 − (1.5 × IQR)                           │
│    Upper Bound = Q3 + (1.5 × IQR)                           │
│  → Rows outside bounds are dropped                          │
│  → Columns with IQR = 0 are skipped                         │
│  → Reduces dataset by ~1–2%                                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              🏷️  STEP 3: TARGET ENCODING                    │
│               DataProcessor.preprocess()                    │
│                                                             │
│  df["result"] = 1 if pass_fail == "Pass" else 0             │
│                                                             │
│  Class distribution (approximate):                          │
│    Pass (1) ≈ 50–70%   |   Fail (0) ≈ 30–50%               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              📏 STEP 4: PERCENTILE CLIPPING + SCALING        │
│               DataProcessor.preprocess()                    │
│                                                             │
│  1. Clip each feature to [1st percentile, 99th percentile]  │
│     → Suppresses remaining extreme values before scaling    │
│                                                             │
│  2. MinMaxScaler().fit_transform(X)                         │
│     → All 9 features normalised to [0.0 → 1.0]              │
│     → scaler.pkl saved for inference                        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               ✂️  STEP 5: TRAIN / TEST SPLIT                │
│             DataProcessor.split_and_balance()               │
│                                                             │
│  train_test_split(X_scaled, y,                              │
│                   test_size=0.2, random_state=42,           │
│                   stratify=y)                               │
│                                                             │
│  Training Set:  80%  (stratified — same Pass/Fail ratio)    │
│  Test Set:      20%  (held out, never touched by SMOTE)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              🔁 STEP 6: SMOTE OVERSAMPLING                  │
│             DataProcessor.split_and_balance()               │
│                  (applied to TRAIN SET only)                │
│                                                             │
│  SMOTE(random_state=42).fit_resample(X_train, y_train)      │
│  → Generates synthetic minority-class samples               │
│  → Balances Fail class to match Pass count in training set  │
│  → Prevents model from always predicting the majority class │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          🏆 STEP 7: MODEL SELECTION via GridSearchCV        │
│             ModelTrainer.train_and_evaluate()               │
│                                                             │
│  Two candidates, each tuned with 3-fold StratifiedKFold,   │
│  scored by weighted F1:                                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Logistic Regression  C ∈ {0.1, 1.0, 10.0}           │   │
│  │ XGBoost              n_estimators ∈ {50, 100}        │   │
│  │                      max_depth    ∈ {3, 6}           │   │
│  │                      learning_rate∈ {0.05, 0.1}      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Winner: model with highest weighted F1 on test set         │
│  → Saved as model.pkl                                       │
│  → Metrics (accuracy/precision/recall/F1) saved in meta.pkl │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            🔵 STEP 8: K-MEANS CLUSTERING                    │
│              ModelTrainer.train_kmeans()                    │
│                  (Unsupervised — no labels)                 │
│                                                             │
│  KMeans(n_clusters=3, random_state=42).fit(X_scaled)        │
│  → Segments students into 3 performance clusters:           │
│      Cluster 0: High Achievers                              │
│      Cluster 1: Average Performers                          │
│      Cluster 2: Struggling / At-Risk Students               │
│  → Saved as: kmeans.pkl                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               💾 STEP 9: ARTIFACT SERIALISATION             │
│              ModelTrainer.save_artifacts()                  │
│                                                             │
│  All artifacts saved to /models/ using joblib:              │
│                                                             │
│  ┌──────────────────────────┬───────────────────────────┐  │
│  │ File                     │ Purpose                   │  │
│  ├──────────────────────────┼───────────────────────────┤  │
│  │ model.pkl                │ Best trained classifier   │  │
│  │ scaler.pkl               │ Fitted MinMaxScaler       │  │
│  │ kmeans.pkl               │ Behavioural cluster model │  │
│  │ meta.pkl                 │ Model name + F1/acc/etc   │  │
│  │ feature_names.pkl        │ Ordered feature name list │  │
│  │ input_feature_cols.pkl   │ Feature column list       │  │
│  └──────────────────────────┴───────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│        🖥️  STEP 10: STREAMLIT DASHBOARD (app.py)            │
│                                                             │
│  PAGE 1 — Dashboard (pages/dashboard.py):                   │
│    1. User sets 9 sliders (scores, habits, wellbeing)       │
│    2. Input → DataFrame → scaler.transform()                │
│    3. model.predict_proba() → Pass/Fail + probability %     │
│    4. kmeans.predict() → Behavioural cluster (0, 1, 2)      │
│    5. Rule-based tips generated from threshold checks       │
│    6. CTA button → navigates to Study Coach chat            │
│                                                             │
│  PAGE 2 — AI Study Coach (pages/chat_interface.py):         │
│    1. LangGraph agent pipeline invoked per message:         │
│       Router → Diagnose → Plan → Retrieve → Respond → Memory│
│    2. Groq LLM (llama-3.1-8b-instant) generates responses   │
│    3. ChromaDB RAG retrieves relevant study tips            │
│    4. Tavily live web search adds current resource links    │
│    5. SessionMemory persists conversation history           │
│    6. Off-topic questions are politely declined             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     📤 FINAL OUTPUT                         │
│                                                             │
│  Dashboard:  Pass ✅ / Fail ❌ + Probability %              │
│              Behavioural Cluster (High / Average / At-Risk) │
│              Score breakdown bars (Math / Reading / Writing)│
│              Personalised actionable improvement tips       │
│                                                             │
│  Study Coach: Personalised 7-day study plan                 │
│               RAG-retrieved study tips                      │
│               Live web resource links (Tavily)              │
│               Conversational coaching (multi-turn)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI Study Coach Pipeline
### `pages/chat_interface.py` — LangGraph Agent Flow

```
┌─────────────────────────────────────────────────────────────┐
│              👤 USER INPUT (chat_interface.py)              │
│                                                             │
│  Prerequisites (enforced by guard clause):                  │
│    • student_data must exist in st.session_state            │
│    • prediction_done flag must be True                      │
│    → If missing: redirect to Dashboard page                 │
│                                                             │
│  Inputs passed to agent.chat():                             │
│    • user_message     — current chat prompt                 │
│    • student_data     — 9 features + prediction + cluster   │
│    • session_history  — loaded from SessionMemory           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            🧠 LANGGRAPH STATE INITIALISATION                │
│             StudyCoachAgent.chat()  →  AgentState           │
│                                                             │
│  AgentState fields:                                         │
│  ┌────────────────────────┬──────────────────────────────┐  │
│  │ Field                  │ Initial value                │  │
│  ├────────────────────────┼──────────────────────────────┤  │
│  │ messages               │ [HumanMessage(user_message)] │  │
│  │ student_data           │ dict from st.session_state   │  │
│  │ learning_gaps          │ []                           │  │
│  │ study_plan             │ None                         │  │
│  │ retrieved_resources    │ []                           │  │
│  │ web_links              │ []                           │  │
│  │ session_history        │ SessionMemory.get_history()  │  │
│  │ is_study_query         │ False  (set by RouterNode)   │  │
│  └────────────────────────┴──────────────────────────────┘  │
│                                                             │
│  messages uses operator.add reducer → auto-appended         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   🔀 NODE 1: RouterNode                     │
│                    agents/nodes.py                          │
│                                                             │
│  Calls Groq LLM (llama-3.1-8b-instant):                    │
│    System: "Classify query — STUDY or GENERAL"              │
│    Input:  last user message                                │
│                                                             │
│  → STUDY  : academic topics (scores, plans, study tips)     │
│  → GENERAL: greetings, off-topic, casual chat              │
│                                                             │
│  Sets state["is_study_query"] = True | False                │
│  Fallback on LLM error → defaults to True (STUDY path)     │
└────────────┬────────────────────────────────────┬───────────┘
             │ is_study_query == True              │ is_study_query == False
             ▼                                    ▼
┌────────────────────────┐          ┌─────────────────────────┐
│  🔬 NODE 2:            │          │  ⚡ SHORT PATH           │
│  DiagnosisNode         │          │  Skip directly to        │
│                        │          │  ResponseGeneratorNode   │
│  Rule-based thresholds │          │  (no plan, no RAG)       │
│  on student_data:      │          └────────────┬────────────┘
│                        │                       │
│  math_score    < 50 →  │                       │
│  reading_score < 50 →  │                       │
│  writing_score < 50 →  │                       │
│  attendance    < 0.75→ │                       │
│  study_hours   < 2h  → │                       │
│  stress_level  > 7   → │                       │
│  sleep_hours   < 6h  → │                       │
│  motivation    < 30  → │                       │
│                        │                       │
│  → learning_gaps list  │                       │
│  → ["no major gaps"]   │                       │
│    if all pass         │                       │
└────────────┬───────────┘                       │
             │                                   │
             ▼                                   │
┌─────────────────────────────────────────────────────────────┐
│                   📅 NODE 3: PlannerNode                    │
│                    agents/nodes.py                          │
│                                                             │
│  Builds a structured prompt with:                           │
│    • Predicted outcome (Pass / Fail)                        │
│    • learning_gaps from DiagnosisNode                       │
│    • Full student profile (all 9 metrics)                   │
│                                                             │
│  Calls Groq LLM → generates a 7-day study plan             │
│    Day 1 through Day 7, task-specific per weak area         │
│                                                             │
│  → state["study_plan"] = plan text                          │
│  Fallback on error → placeholder message                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              📚 NODE 4: ResourceRetrieverNode               │
│                    agents/nodes.py                          │
│                                                             │
│  Query = "study tips for {top 2 learning gaps}"             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  RAGTool  (tools/rag_tool.py)                        │   │
│  │    • ChromaDB vector store  (BASE_DIR/chroma_db/)    │   │
│  │    • Embeddings: all-MiniLM-L6-v2 (HuggingFace)     │   │
│  │    • 25 curated study-tip documents (auto-seeded)    │   │
│  │    • similarity_search(query, k=3)                   │   │
│  │    → retrieved_resources: List[str]  (3 tips)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  WebSearchTool  (tools/web_search_tool.py)           │   │
│  │    • TavilyClient (TAVILY_API_KEY)                   │   │
│  │    • Live web search — returns top 2 results         │   │
│  │    → web_links: List[str]  ("Title — URL" format)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Both results stored in AgentState                          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼ (joins GENERAL path here)
┌─────────────────────────────────────────────────────────────┐
│             💬 NODE 5: ResponseGeneratorNode                │
│                    agents/nodes.py                          │
│                                                             │
│  ── STUDY path system prompt contains: ─────────────────── │
│    • Predicted outcome + full student profile               │
│    • Identified weak areas (learning_gaps)                  │
│    • RAG study tips (retrieved_resources)                   │
│    • Web resource links with full URLs                      │
│    • Generated 7-day study plan                             │
│    • Last 6 turns of conversation history                   │
│    • Instructions to be encouraging + reference scores      │
│                                                             │
│  ── GENERAL path system prompt: ──────────────────────────  │
│    • Instructs agent to decline off-topic questions         │
│    • Redirects to academic topics only                      │
│    • Includes last 6 turns of conversation history          │
│                                                             │
│  Calls Groq LLM → final AI reply                           │
│  state["messages"] = [AIMessage(content=reply)]             │
│  Fallback on error → connection error message               │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  🗂️  NODE 6: MemoryNode                     │
│                    agents/nodes.py                          │
│                                                             │
│  Extracts last HumanMessage + last AIMessage from state     │
│  Appends both as turns to session_history:                  │
│    {"role": "user",      "content": "..."}                  │
│    {"role": "assistant", "content": "..."}                  │
│                                                             │
│  state["session_history"] = updated list                    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         END                                 │
│            StateGraph compilation terminates                │
│                                                             │
│  agent.chat() extracts from final state:                    │
│    ai_reply        = result["messages"][-1].content         │
│    updated_history = result["session_history"]              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           💾 SESSION MEMORY SYNC  (chat_interface.py)       │
│                                                             │
│  session_mem.set_history("active", updated_history)         │
│  → SessionMemory stores history in st.session_state         │
│  → Persists across Streamlit reruns for this user           │
│  → Loaded back via session_mem.get_history("active")        │
│    on the next message                                       │
│                                                             │
│  On new prediction (dashboard.py):                          │
│    st.session_state.pop("session_memory")                   │
│    → Full reset — fresh coaching session begins             │
│                                                             │
│  On "Clear Chat" button:                                    │
│    session_mem.clear("active")                              │
│    → History wiped, chat_history reset, page reruns         │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 📤 STUDY COACH FINAL OUTPUT                 │
│                                                             │
│  Conversational AI reply rendered via st.chat_message()     │
│                                                             │
│  STUDY query reply includes:                                │
│    ✅ Personalised 7-day study plan (Day 1–7)               │
│    ✅ Up to 3 RAG-retrieved study tips                      │
│    ✅ Up to 2 live web resource links (clickable URLs)      │
│    ✅ References to specific student scores & weak areas    │
│    ✅ Encouraging tone (especially for Fail predictions)    │
│                                                             │
│  GENERAL query reply:                                       │
│    ℹ️  Polite decline + redirect to academic topics         │
└─────────────────────────────────────────────────────────────┘
```

---

**⚠️ Note on model metrics:** All evaluation metrics (accuracy, precision, recall, F1) reach
1.00 on this dataset because the `pass_fail` label in the synthetic CSV is a deterministic
function of the input feature scores (specifically the average of `math_score`,
`reading_score`, and `writing_score`). The classifier trivially learns this rule. On a
real-world dataset with independently collected labels and natural noise, metrics would be
lower and more representative.

---

**Team:** Yash Agarwal
