# Student Pass/Fail Predictor with AI Study Coach

### A Machine Learning + Generative AI Approach to Personalised Academic Intervention

* **Yash Agarwal** – Full Stack ML, Agent Architecture & Deployment

**Project Details:**

* **Course:** Intro to GenAI Capstone Project
* **Batch:** 2024
* **Date:** April 2026
* **GitHub Repository:** https://github.com/Yash-205/student_pass-fa_predicting_ml_model
* **Hosted Application:** https://student-pass-fail-predicting-ml-model.streamlit.app/

---

## 1. Problem Statement

### Context and Background

In academic environments, student performance evaluation is traditionally reactive — educators only become aware of a student's academic risk after grades have already declined. By the time lagging indicators like exam scores become visible, the critical window for early intervention has often passed.

This project goes a step further than a standard prediction system: it not only predicts whether a student will pass or fail, but also provides each student with a **personalised AI-powered coaching session** that diagnoses their weak areas, generates a targeted 7-day study plan, and retrieves live resources — all in a conversational interface.

### The Core Challenge

Two challenges are addressed simultaneously:

1. **Prediction:** Accurately predict student pass/fail outcomes from 9 behavioral and academic features, handling class imbalance and selecting the best model objectively.
2. **Intervention:** Translate a raw prediction into actionable, personalised guidance through a conversational AI agent — making the system useful to students, not just to administrators.

### Solution Overview

The system is a two-page Streamlit application:

**Page 1 — Dashboard:** The student enters 9 metrics via sliders and receives an instant Pass/Fail prediction with probability, a behavioural cluster assignment, score breakdown, and personalised tips.

**Page 2 — AI Study Coach:** The student chats with a LangGraph-powered AI agent that diagnoses their weak areas, generates a 7-day study plan via Groq LLM, retrieves study tips from a ChromaDB knowledge base, and fetches live web resources via Tavily search.

---

## 2. Data Description

### Source Data

The dataset `data/student_data.csv` contains approximately **1,000,000 synthetic student records** simulating large-scale institutional data across a diverse student population.

### Dataset Features (9 Input Features + 1 Target)

| Feature                      | Type    | Description                                               |
| :--------------------------- | :------ | :-------------------------------------------------------- |
| `parental_education_level` | Integer | 1 (no formal education) → 7 (postgraduate)               |
| `daily_study_hours`        | Float   | Hours studied independently per day (0–12)               |
| `attendance_rate`          | Float   | Proportion of classes attended (0.0–1.0)                 |
| `sleep_hours`              | Float   | Hours of sleep per night (3–12)                          |
| `stress_level`             | Integer | Self-reported stress level (1 = very low, 10 = very high) |
| `motivation_score`         | Integer | Motivation score (0–100)                                 |
| `math_score`               | Integer | Math subject score (0–100)                               |
| `reading_score`            | Integer | Reading subject score (0–100)                            |
| `writing_score`            | Integer | Writing subject score (0–100)                            |
| `pass_fail`                | String  | Target label —`"Pass"` or `"Fail"`                   |

> **Note on the synthetic dataset:** The `pass_fail` label in this dataset is deterministically derived from the average of `math_score`, `reading_score`, and `writing_score`. This causes the trained classifier to learn a perfect rule and achieve 1.00 on all evaluation metrics. This is expected behaviour for this synthetic dataset; real-world labels, collected independently of features, would produce lower and more informative metrics.

### Feature Engineering Decisions

All 9 features are used directly as model inputs after cleaning and scaling. No engineered features are added. `pass_fail` is the only target — there is no separate grade prediction model in this version.

---

## 3. EDA & Preprocessing

### 3.1 Null Value Handling

For each of the 9 feature columns, missing values are filled with the **column median** — chosen over mean for robustness against skewed distributions:

```python
df[col].fillna(df[col].median())
```

After this step, 0 null values remain in any feature column.

### 3.2 Outlier Removal (IQR Method)

Applied to each feature column independently:

```
Lower Bound = Q1 − 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Rows outside these bounds are dropped. Columns with IQR = 0 (no variation) are skipped. This step removes approximately 1–2% of records.

### 3.3 Target Encoding

```python
df["result"] = 1 if pass_fail == "Pass" else 0
```

Class distribution is printed at this stage to confirm approximate balance between Pass and Fail classes.

### 3.4 Percentile Clipping + MinMaxScaling

Before scaling, each feature is clipped to its 1st–99th percentile range to suppress remaining extreme values. MinMaxScaler is then applied:

```
X_scaled = (X − X_min) / (X_max − X_min)   →   all values in [0.0, 1.0]
```

The fitted scaler is saved as `models/scaler.pkl` for use at inference time.

### 3.5 Train/Test Split

```python
train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
```

- **Training set:** 80% (stratified — same Pass/Fail ratio)
- **Test set:** 20% (held out permanently, never seen during training or SMOTE)

### 3.6 SMOTE Oversampling

**Synthetic Minority Over-sampling Technique** is applied exclusively to the training split:

```python
SMOTE(random_state=42).fit_resample(X_train, y_train)
```

Synthetic minority-class samples are generated by interpolating between real minority neighbours. This balances the training distribution without corrupting the test set's real-world proportions.

---

## 4. Methodology

### 4.1 Technical Stack

| Category             | Libraries / Tools                                                                |
| :------------------- | :------------------------------------------------------------------------------- |
| Data Processing      | `pandas`, `numpy`                                                            |
| Preprocessing        | `scikit-learn` (MinMaxScaler, train_test_split)                                |
| Class Balancing      | `imbalanced-learn` (SMOTE)                                                     |
| ML — Classification | `scikit-learn` (LogisticRegression, GridSearchCV), `xgboost` (XGBClassifier) |
| ML — Clustering     | `scikit-learn` (KMeans)                                                        |
| Model Persistence    | `joblib`                                                                       |
| Streamlit UI         | `streamlit`                                                                    |
| Agent Orchestration  | `langgraph` (StateGraph)                                                       |
| LLM                  | `langchain-groq` → Groq API (`llama-3.1-8b-instant`)                        |
| RAG Vector Store     | `langchain-chroma` → ChromaDB                                                 |
| Embeddings           | `langchain-huggingface` → `sentence-transformers/all-MiniLM-L6-v2`          |
| Web Search           | `tavily-python` (TavilyClient)                                                 |

### 4.2 Model Selection via GridSearchCV

Two candidate classifiers are trained and compared using **3-fold Stratified K-Fold cross-validation**, scored by weighted F1:

| Model               | Hyperparameters Searched                                                                 |
| :------------------ | :--------------------------------------------------------------------------------------- |
| Logistic Regression | `C ∈ {0.1, 1.0, 10.0}`                                                                |
| XGBoost Classifier  | `n_estimators ∈ {50, 100}`, `max_depth ∈ {3, 6}`, `learning_rate ∈ {0.05, 0.1}` |

The model with the highest **weighted F1** on the held-out test set is selected as `model.pkl`. This objective selection process ensures the best-performing model is used regardless of algorithm family.

### 4.3 K-Means Behavioral Clustering

A `KMeans(n_clusters=3, random_state=42)` model is fitted on the scaled training data. It segments students into 3 performance clusters:

- **Cluster 0:** High Achievers
- **Cluster 1:** Average Performers
- **Cluster 2:** Struggling / At-Risk Students

The cluster assignment is shown on the dashboard and passed to the AI Study Coach to contextualise advice.

### 4.4 Artifact Serialisation

All trained objects are saved to `models/` using `joblib`:

| File                       | Purpose                                        |
| :------------------------- | :--------------------------------------------- |
| `model.pkl`              | Best trained classifier (LR or XGBoost)        |
| `scaler.pkl`             | Fitted MinMaxScaler                            |
| `kmeans.pkl`             | Fitted KMeans (3 clusters)                     |
| `meta.pkl`               | Best model name + accuracy/precision/recall/F1 |
| `feature_names.pkl`      | Ordered feature name array                     |
| `input_feature_cols.pkl` | Feature column list                            |

### 4.5 LangGraph AI Study Coach

The AI Study Coach is implemented as a **LangGraph `StateGraph`** with 6 nodes wired in sequence:

```
RouterNode → (STUDY) → DiagnosisNode → PlannerNode → ResourceRetrieverNode → ResponseGeneratorNode → MemoryNode → END
           → (GENERAL) ────────────────────────────────────────────────────→ ResponseGeneratorNode → MemoryNode → END
```

| Node                      | Responsibility                                                              |
| :------------------------ | :-------------------------------------------------------------------------- |
| `RouterNode`            | Groq LLM classifies the query as STUDY or GENERAL                           |
| `DiagnosisNode`         | Rule-based thresholds on 8 student metrics →`learning_gaps` list         |
| `PlannerNode`           | Groq LLM generates a personalised 7-day study plan                          |
| `ResourceRetrieverNode` | ChromaDB RAG (k=3 tips) + Tavily live web search (k=2 links)                |
| `ResponseGeneratorNode` | Assembles final reply with all context; politely declines off-topic queries |
| `MemoryNode`            | Appends the current turn to `session_history`                             |

**AgentState** (LangGraph TypedDict) carries all shared data across nodes:

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    student_data: dict
    learning_gaps: List[str]
    study_plan: Optional[str]
    retrieved_resources: List[str]
    web_links: List[str]
    session_history: List[dict]
    is_study_query: bool
```

**Session persistence** is handled by `SessionMemory` (stored in `st.session_state`), which acts as the backing store for conversation history. It is reset automatically when a new prediction is made.

---

## 5. Evaluation

### 5.1 Performance Metrics

Models are evaluated on the isolated 20% held-out test set with stratified class proportions.

| Metric              | Score |
| :------------------ | :---- |
| **Accuracy**  | 1.00  |
| **Precision** | 1.00  |
| **Recall**    | 1.00  |
| **F1 Score**  | 1.00  |

As noted in Section 2, these perfect scores are expected given the synthetic, deterministic nature of the pass/fail labels. On a noisy real-world dataset, all metrics would be meaningfully lower.

### 5.2 Preprocessing Results Summary

| Step                    | Result                                           |
| :---------------------- | :----------------------------------------------- |
| Null value imputation   | 0 nulls remaining after median fill              |
| IQR outlier removal     | ~1–2% of rows removed                           |
| SMOTE on training split | Fail class up-sampled to match Pass count        |
| MinMaxScaler            | All 9 features normalised to [0.0, 1.0]          |
| Model selection         | Best of LR vs XGBoost by weighted F1 on test set |

---

## 6. Optimisation

### 6.1 Objective Model Selection

Rather than committing to a single algorithm, **GridSearchCV** is used to tune both Logistic Regression and XGBoost, and the winning model is selected automatically by F1 score. This prevents algorithm bias and produces a model that is provably the best option explored.

### 6.2 Class Imbalance (SMOTE)

SMOTE is applied **only to the training set** to prevent information leakage into the test evaluation. The test set is evaluated on its original, real-world distribution.

### 6.3 Leakage-Free Preprocessing

The scaler, KMeans model, and all artifacts are **fitted on training data only** and saved as `.pkl` files. Inference loads these pre-fitted objects and applies them to new inputs — ensuring the exact same transformation is used at prediction time as during training.

### 6.4 ChromaDB Absolute Path

The ChromaDB vector store is persisted at an **absolute path** (`BASE_DIR/chroma_db/`) resolved from `src/config.py`. This prevents the `./chroma_db` relative-path issue where the persist directory would change depending on the working directory or deployment environment.

### 6.5 Auto-Training on First Deploy

`app.py` checks for the existence of all required `.pkl` files at startup. If any are missing (e.g. on a fresh Streamlit Cloud deploy), the full training pipeline runs automatically before the UI loads — making the app self-contained without requiring a manual training step.

### 6.6 Per-User Session Isolation

`SessionMemory` is stored inside `st.session_state` (not as a module-level singleton), ensuring each Streamlit user session has its own isolated conversation history. Resetting a prediction clears the memory via `st.session_state.pop("session_memory")`.

---

## 7. Team Contribution

### Yash Agarwal – Full Stack ML, Agent Architecture & Deployment

* Led data acquisition, loading, and initial exploration of the ~1M-record student dataset.
* Designed and implemented the full `DataProcessor` class: IQR outlier removal, median null imputation, percentile clipping, MinMaxScaling, stratified train/test split, and SMOTE oversampling.
* Ensured all transformers were fitted only on training data and saved correctly for leakage-free inference.
* Designed and implemented the `ModelTrainer` class with GridSearchCV over Logistic Regression and XGBoost.
* Configured the 3-fold StratifiedKFold evaluation strategy and weighted F1 scoring.
* Implemented KMeans clustering (k=3) for student behavioural segmentation.
* Evaluated model performance on the held-out test set and managed artifact serialisation.
* Built the complete two-page Streamlit application (`app.py`, `pages/dashboard.py`, `pages/chat_interface.py`).
* Implemented `StudentPredictor` inference class and all prediction + recommendation logic.
* Created the `UIBuilder` class with the full dark glassmorphism CSS design system, prediction cards, score bars, stat chips, and tip cards.
* Managed auto-training on first deploy and Streamlit Cloud secret key integration.
* Designed and implemented the complete LangGraph `StudyCoachAgent` with 6 nodes (Router, Diagnosis, Planner, ResourceRetriever, ResponseGenerator, Memory).
* Built `RAGTool` (ChromaDB + HuggingFace embeddings, 25 curated study tips) and `WebSearchTool` (Tavily live search).
* Implemented `SessionMemory` class and wired it as the per-user backing store for conversation history.
* Authored all technical documentation including `ARCHITECTURE_PIPELINE.md` and `FINAL_PROJECT_REPORT.md`.

### Contribution Summary

| Task Area                       | Primary Contributor |
| :------------------------------ | :------------------ |
| Data Cleaning & Preprocessing   | Yash Agarwal        |
| ML Modelling & GridSearchCV     | Yash Agarwal        |
| Streamlit UI & Inference Engine | Yash Agarwal        |
| LangGraph Agent & AI Tools      | Yash Agarwal        |

---

*This project demonstrates how a combination of classical machine learning and modern generative AI can transform a raw pass/fail prediction into a fully personalised, conversational academic coaching experience — available to any student, at any time, at zero marginal cost.*
