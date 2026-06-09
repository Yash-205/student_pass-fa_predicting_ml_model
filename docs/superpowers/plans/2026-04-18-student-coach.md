# Student Pass/Fail Predictor + AI Study Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain M1 ML pipeline on new dataset, then build a 2-page Streamlit app where a student enters their data, gets a Pass/Fail prediction with dashboard, and chats with a LangGraph AI study coach powered by Groq + Tavily + ChromaDB RAG.

**Architecture:** Linear LangGraph agent with a router node that sends study queries through diagnose→plan→retrieve→respond and general chat straight to respond→memory. Student data from the dashboard is passed into the agent via session state.

**Tech Stack:** scikit-learn, XGBoost, SMOTE, MinMaxScaler, LangGraph, LangChain-Groq (llama-3.1-8b-instant), Tavily, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Streamlit

---

## File Map

**Modified:**
- `requirements.txt` — add LangGraph, LangChain, Groq, Tavily, ChromaDB, sentence-transformers, plotly, python-dotenv
- `src/config.py` — update DATA_PATH and FEATURES for new dataset
- `src/data_processor.py` — rewrite preprocessing for new dataset (MinMaxScaler, no poly, null fill, new target col)
- `src/model_trainer.py` — fix XGBoost deprecation, update save_artifacts signature (remove poly)
- `src/inference.py` — remove poly transform, update feature cols and recommendations
- `train_model.py` — add fill_nulls step, update save_artifacts call
- `app.py` — replace with st.navigation multi-page entry point

**Created:**
- `.env` — GROQ_API_KEY, TAVILY_API_KEY
- `agents/__init__.py`
- `agents/state.py` — AgentState TypedDict
- `agents/nodes.py` — RouterNode, DiagnosisNode, PlannerNode, ResourceRetrieverNode, ResponseGeneratorNode, MemoryNode
- `agents/study_coach_agent.py` — StudyCoachAgent (compiles LangGraph graph)
- `tools/__init__.py`
- `tools/web_search_tool.py` — Tavily wrapper
- `tools/rag_tool.py` — ChromaDB + HuggingFace embeddings, seeded knowledge base
- `memory/__init__.py`
- `memory/session_memory.py` — SessionMemory class
- `pages/__init__.py`
- `pages/dashboard.py` — student input form + prediction + charts
- `pages/chat_interface.py` — LangGraph chat UI

---

## Task 1: Update requirements.txt and create .env

**Files:**
- Modify: `requirements.txt`
- Create: `.env`

- [ ] **Step 1: Rewrite requirements.txt**

```text
imbalanced-learn==0.14.1
joblib==1.5.3
matplotlib==3.10.8
numpy==2.4.2
pandas==2.3.3
scikit-learn==1.8.0
streamlit==1.54.0
xgboost==3.2.0
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-groq>=0.2.0
langchain-huggingface>=0.1.0
chromadb>=0.5.0
tavily-python>=0.5.0
sentence-transformers>=3.0.0
plotly>=5.20.0
python-dotenv>=1.0.0
```

- [ ] **Step 2: Create .env with API keys**

```text
GROQ_API_KEY=<your-groq-api-key>
TAVILY_API_KEY=<your-tavily-api-key>
```

- [ ] **Step 3: Ensure .env is in .gitignore**

Add `.env` to `.gitignore` if not already present.

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors. `sentence-transformers` download may take a minute.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add M2 dependencies to requirements"
```

---

## Task 2: Update src/config.py

**Files:**
- Modify: `src/config.py`

- [ ] **Step 1: Rewrite config.py**

```python
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student_data.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURES = [
    "parental_education_level",
    "daily_study_hours",
    "attendance_rate",
    "sleep_hours",
    "stress_level",
    "motivation_score",
    "math_score",
    "reading_score",
    "writing_score",
]
```

- [ ] **Step 2: Verify file saved correctly**

```bash
python -c "from src.config import DATA_PATH, FEATURES; print(DATA_PATH); print(FEATURES)"
```

Expected output:
```
/Users/.../data/student_data.csv
['parental_education_level', 'daily_study_hours', 'attendance_rate', 'sleep_hours', 'stress_level', 'motivation_score', 'math_score', 'reading_score', 'writing_score']
```

- [ ] **Step 3: Commit**

```bash
git add src/config.py
git commit -m "feat: update config for new dataset and features"
```

---

## Task 3: Rewrite src/data_processor.py

**Files:**
- Modify: `src/data_processor.py`

- [ ] **Step 1: Rewrite data_processor.py**

```python
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

from src.config import DATA_PATH, FEATURES


class DataProcessor:
    def __init__(self, data_path: str = str(DATA_PATH)) -> None:
        self.data_path = data_path
        self.feature_cols = FEATURES
        self.scaler = MinMaxScaler()
        self.feature_names: np.ndarray = np.array(FEATURES)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        print(f"Loaded: {df.shape[0]:,} rows | Columns: {df.columns.tolist()}")
        return df

    def fill_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        print(f"Null values filled with column medians.")
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.feature_cols:
            if col in df.columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        print(f"After IQR outlier removal: {df.shape[0]:,} rows")
        return df

    def preprocess(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, pd.DataFrame]:
        df = df.copy()
        df["result"] = df["pass_fail"].apply(
            lambda x: 1 if str(x).strip().lower() == "pass" else 0
        )
        vc = df["result"].value_counts()
        print(
            f"Class distribution → Pass: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)  "
            f"Fail: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)"
        )

        X_raw = df[self.feature_cols].copy()
        y = df["result"].copy()

        for col in self.feature_cols:
            X_raw[col] = X_raw[col].clip(
                X_raw[col].quantile(0.01), X_raw[col].quantile(0.99)
            )

        X_scaled = self.scaler.fit_transform(X_raw)
        return X_scaled, y, df

    def split_and_balance(
        self, X: np.ndarray, y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

        vc_sm = pd.Series(y_train_sm).value_counts()
        print(f"After SMOTE → Pass: {vc_sm.get(1,0):,}  Fail: {vc_sm.get(0,0):,}")
        return X_train_sm, X_test, y_train_sm, y_test
```

- [ ] **Step 2: Verify preprocessing runs on the dataset**

```bash
python -c "
from src.data_processor import DataProcessor
dp = DataProcessor()
df = dp.load_data()
df = dp.fill_nulls(df)
df = dp.remove_outliers(df)
X, y, df2 = dp.preprocess(df)
print('X shape:', X.shape, '| y shape:', y.shape)
print('X min/max:', X.min(), X.max())
"
```

Expected: X shape with 9 columns, min≥0.0, max≤1.0 (MinMaxScaler bounds).

- [ ] **Step 3: Commit**

```bash
git add src/data_processor.py
git commit -m "feat: rewrite data_processor for new dataset with MinMaxScaler"
```

---

## Task 4: Update src/model_trainer.py

**Files:**
- Modify: `src/model_trainer.py`

- [ ] **Step 1: Update save_artifacts signature and fix XGBoost deprecation**

Replace the full file:

```python
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

from src.config import MODELS_DIR


class ModelTrainer:
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        self.models_dir = models_dir
        self.best_model: Any = None
        self.best_name: str = ""
        self.kmeans: Optional[KMeans] = None
        self.metrics: Dict[str, float] = {}

    def train_and_evaluate(
        self,
        X_train_sm: np.ndarray,
        y_train_sm: pd.Series,
        X_test: np.ndarray,
        y_test: pd.Series,
    ) -> None:
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        candidates = {
            "LogisticRegression": {
                "model": LogisticRegression(
                    class_weight="balanced", random_state=42, max_iter=2000
                ),
                "params": {"C": [0.1, 1.0, 10.0]},
            },
            "XGBoost": {
                "model": XGBClassifier(
                    eval_metric="logloss", random_state=42, n_jobs=-1
                ),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 6],
                    "learning_rate": [0.05, 0.1],
                },
            },
        }

        print(
            "\n{:<20} {:>10} {:>12} {:>10} {:>12}".format(
                "Model", "Accuracy", "Precision", "Recall", "F1 (weighted)"
            )
        )
        print("-" * 66)

        best_f1 = -1.0
        best_params: Dict[str, Any] = {}

        for name, config in candidates.items():
            print(f"\nRunning GridSearchCV for {name}...")
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                scoring="f1_weighted",
                cv=cv_strategy,
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train_sm, y_train_sm)
            best_estimator = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")

            y_pred = best_estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            print(
                "{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>12.4f}".format(
                    name, acc, prec, rec, f1
                )
            )

            if f1 > best_f1:
                best_f1, self.best_name, self.best_model, best_params = (
                    f1,
                    name,
                    best_estimator,
                    grid_search.best_params_,
                )

        print(f"\n🏆 Best overall model: {self.best_name}  (F1 = {best_f1:.4f})")
        print(f"Winning hyperparameters: {best_params}")

        y_pred_final = self.best_model.predict(X_test)
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred_final)),
            "precision": float(precision_score(y_test, y_pred_final, average="weighted")),
            "recall": float(recall_score(y_test, y_pred_final, average="weighted")),
            "f1": float(f1_score(y_test, y_pred_final, average="weighted")),
        }

        print(f"\n── Final Metrics ({self.best_name}) ──────────────────")
        for k, v in self.metrics.items():
            print(f"  {k.capitalize():<10}: {v:.4f}")

        print("\n── Classification Report ──────────────────────────────")
        print(classification_report(y_test, y_pred_final, target_names=["Fail", "Pass"]))

        cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1])
        print("── Confusion Matrix ───────────────────────────────────")
        print(f"  TN: {cm[0,0]:>8,}  FP: {cm[0,1]:>8,}")
        print(f"  FN: {cm[1,0]:>8,}  TP: {cm[1,1]:>8,}")

    def train_kmeans(
        self, df: pd.DataFrame, feature_cols: List[str], scaler: Any
    ) -> None:
        self.kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
        self.kmeans.fit(scaler.transform(df[feature_cols]))

    def save_artifacts(
        self, scaler: Any, feature_names: np.ndarray, feature_cols: List[str]
    ) -> None:
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.best_model, os.path.join(self.models_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(self.models_dir, "scaler.pkl"))
        joblib.dump(self.kmeans, os.path.join(self.models_dir, "kmeans.pkl"))
        joblib.dump(
            feature_names.tolist(),
            os.path.join(self.models_dir, "feature_names.pkl"),
        )
        joblib.dump(
            feature_cols, os.path.join(self.models_dir, "input_feature_cols.pkl")
        )
        joblib.dump(
            {"best_model_name": self.best_name, **self.metrics},
            os.path.join(self.models_dir, "meta.pkl"),
        )
        print("\n✅ All artifacts saved to models/")
```

- [ ] **Step 2: Commit**

```bash
git add src/model_trainer.py
git commit -m "feat: update model_trainer - fix XGBoost deprecation, remove poly"
```

---

## Task 5: Update src/inference.py

**Files:**
- Modify: `src/inference.py`

- [ ] **Step 1: Rewrite inference.py**

```python
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.config import MODELS_DIR, FEATURES


class StudentPredictor:
    def __init__(self, models_dir: str = str(MODELS_DIR)) -> None:
        self.models_dir = models_dir
        self.feature_cols = FEATURES
        self.model_files = {
            "model":  os.path.join(self.models_dir, "model.pkl"),
            "scaler": os.path.join(self.models_dir, "scaler.pkl"),
            "kmeans": os.path.join(self.models_dir, "kmeans.pkl"),
            "meta":   os.path.join(self.models_dir, "meta.pkl"),
        }
        self.bundle = self._load_pretrained_model()

    def _load_pretrained_model(self) -> Optional[Dict[str, Any]]:
        if not all(os.path.exists(v) for v in self.model_files.values()):
            return None
        return {
            "model":  joblib.load(self.model_files["model"]),
            "scaler": joblib.load(self.model_files["scaler"]),
            "kmeans": joblib.load(self.model_files["kmeans"]),
            "meta":   joblib.load(self.model_files["meta"]),
        }

    def predict_bundle(
        self, X_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.bundle:
            raise RuntimeError("Models not loaded. Run train_model.py first.")
        Xc = X_df[self.feature_cols].copy()
        for col in self.feature_cols:
            Xc[col] = pd.to_numeric(Xc[col], errors="coerce").fillna(0.0)
        X_s = self.bundle["scaler"].transform(Xc)
        probas = self.bundle["model"].predict_proba(X_s)[:, 1]
        preds = (probas >= 0.5).astype(int)
        clusters = self.bundle["kmeans"].predict(X_s)
        return probas, preds, clusters

    def get_student_recommendations(self, row: pd.Series) -> List[str]:
        recs = []
        if float(row.get("daily_study_hours", 5)) < 2:
            recs.append("📚 Increase daily study to at least 2–3 hours.")
        if float(row.get("attendance_rate", 1.0)) < 0.75:
            recs.append("🏫 Aim for 75%+ attendance to reduce knowledge gaps.")
        if float(row.get("math_score", 100)) < 50:
            recs.append("🔢 Focus on math — practice problem sets daily.")
        if float(row.get("reading_score", 100)) < 50:
            recs.append("📖 Improve reading — summarize one article daily.")
        if float(row.get("writing_score", 100)) < 50:
            recs.append("✍️ Work on writing structure and grammar daily.")
        if float(row.get("stress_level", 5)) > 7:
            recs.append("🧘 High stress detected — try the Pomodoro technique.")
        if float(row.get("sleep_hours", 8)) < 6:
            recs.append("😴 Aim for 7–8 hours of sleep for better retention.")
        if float(row.get("motivation_score", 50)) < 30:
            recs.append("🎯 Set small SMART goals to rebuild motivation daily.")
        if not recs:
            recs.append("🌟 Great habits! Keep consistency and add spaced revision.")
        return recs

    def get_meta(self) -> Dict[str, Any]:
        return self.bundle["meta"] if self.bundle else {}

    def is_ready(self) -> bool:
        return self.bundle is not None
```

- [ ] **Step 2: Commit**

```bash
git add src/inference.py
git commit -m "feat: update inference for new features, remove poly dependency"
```

---

## Task 6: Update train_model.py and retrain

**Files:**
- Modify: `train_model.py`

- [ ] **Step 1: Rewrite train_model.py**

```python
from dotenv import load_dotenv
load_dotenv()

from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer


def main() -> None:
    print("🚀 Starting training pipeline...")

    processor = DataProcessor()
    trainer = ModelTrainer()

    print("\n[1] Loading data...")
    df = processor.load_data()

    print("\n[2] Filling null values...")
    df = processor.fill_nulls(df)

    print("\n[3] Removing outliers...")
    df_clean = processor.remove_outliers(df)

    print("\n[4] Preprocessing features and target...")
    X_scaled, y, df_clean = processor.preprocess(df_clean)

    print("\n[5] Splitting and balancing with SMOTE...")
    X_train_sm, X_test, y_train_sm, y_test = processor.split_and_balance(X_scaled, y)

    print("\n[6] Training and evaluating models...")
    trainer.train_and_evaluate(X_train_sm, y_train_sm, X_test, y_test)

    print("\n[7] Training KMeans clustering...")
    trainer.train_kmeans(df_clean, processor.feature_cols, processor.scaler)

    print("\n[8] Saving artifacts...")
    trainer.save_artifacts(
        processor.scaler, processor.feature_names, processor.feature_cols
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training**

```bash
python train_model.py
```

Expected: Training completes, final metrics printed, `models/` directory created with `model.pkl`, `scaler.pkl`, `kmeans.pkl`, `meta.pkl`, `feature_names.pkl`, `input_feature_cols.pkl`.

- [ ] **Step 3: Verify artifacts exist**

```bash
ls -lh models/
```

Expected: 6 `.pkl` files present.

- [ ] **Step 4: Verify inference works**

```bash
python -c "
from src.inference import StudentPredictor
import pandas as pd
p = StudentPredictor()
print('Ready:', p.is_ready())
df = pd.DataFrame([{
    'parental_education_level': 3,
    'daily_study_hours': 3.5,
    'attendance_rate': 0.85,
    'sleep_hours': 7.0,
    'stress_level': 5,
    'motivation_score': 60,
    'math_score': 65,
    'reading_score': 70,
    'writing_score': 68,
}])
prob, pred, cluster = p.predict_bundle(df)
print('Prob:', prob, 'Pred:', pred, 'Cluster:', cluster)
"
```

Expected: Ready: True, numeric probability, pred 0 or 1, cluster 0/1/2.

- [ ] **Step 5: Commit**

```bash
git add train_model.py models/
git commit -m "feat: retrain M1 pipeline on new dataset, add fill_nulls step"
```

---

## Task 7: Create agent state

**Files:**
- Create: `agents/__init__.py`
- Create: `agents/state.py`

- [ ] **Step 1: Create agents/__init__.py**

Empty file: `agents/__init__.py`

- [ ] **Step 2: Create agents/state.py**

```python
from typing import TypedDict, List, Optional, Annotated
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    student_data: dict
    learning_gaps: List[str]
    study_plan: Optional[str]
    retrieved_resources: List[str]
    session_history: List[dict]
    web_links: List[str]
    is_study_query: bool
```

- [ ] **Step 3: Verify import**

```bash
python -c "from agents.state import AgentState; print('AgentState OK')"
```

Expected: `AgentState OK`

- [ ] **Step 4: Commit**

```bash
git add agents/
git commit -m "feat: add agent state definition"
```

---

## Task 8: Create tools

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/web_search_tool.py`
- Create: `tools/rag_tool.py`

- [ ] **Step 1: Create tools/__init__.py**

Empty file.

- [ ] **Step 2: Create tools/web_search_tool.py**

```python
import os
from typing import List
from tavily import TavilyClient


class WebSearchTool:
    def __init__(self, api_key: str = None):
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 3) -> List[str]:
        try:
            results = self.client.search(query)
            return [r["url"] for r in results["results"][:max_results]]
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
```

- [ ] **Step 3: Create tools/rag_tool.py**

```python
import os
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

STUDY_TIPS = [
    "For improving math scores: practice daily with problem sets, focus on weak areas first, use spaced repetition for formulas.",
    "For improving reading comprehension: read actively by summarizing each paragraph, build vocabulary daily, practice with timed passages.",
    "For improving writing scores: outline before writing, practice essay structures daily, read model essays and analyze structure.",
    "Managing stress while studying: use the Pomodoro technique (25 min study, 5 min break), exercise regularly, maintain consistent sleep.",
    "Improving attendance: set morning routines, use accountability partners, track attendance goals weekly.",
    "Increasing study hours: block-schedule dedicated study time, eliminate phone distractions, use active recall instead of passive reading.",
    "Improving motivation: set SMART goals, track progress visually with charts, reward small wins, join study groups.",
    "Sleep and academic performance: 7-9 hours of sleep improves memory consolidation, avoid screens 1 hour before bed.",
    "Parental education and support: discuss academic goals with parents weekly, create a quiet dedicated study space at home.",
    "Online resources for math: Khan Academy, Wolfram Alpha, PatrickJMT on YouTube are free and effective.",
    "Online resources for reading and writing: Purdue OWL for writing guides, ReadWorks for reading comprehension practice.",
    "Feynman technique: explain a concept in simple terms as if teaching a child — gaps in explanation reveal gaps in understanding.",
    "Time management for students: use weekly planners, prioritize high-value tasks using Eisenhower matrix, review plans every Sunday.",
    "Dealing with test anxiety: practice past papers under timed conditions, use box breathing (4-4-4-4) before exams.",
    "Low income and education: use free resources — public libraries, Khan Academy, MIT OpenCourseWare, and edX free courses.",
    "Daily study routine: study at the same time each day to build strong habit, start with the hardest subject first.",
    "Attendance and learning: missing class creates cumulative gaps, review missed content on the same day.",
    "Motivation improvement: find personal relevance in each subject, connect academic work to career goals you care about.",
    "Internet for self-learning: YouTube, Coursera (audit mode), and edX (audit mode) provide free high-quality courses.",
    "Active learning strategies: teach others what you learned, create mind maps, use Anki flashcards for key concepts.",
    "Study groups: meet 2-3 times per week, assign rotating teacher role, review past exam papers together.",
    "Exam preparation: start 2 weeks early, create one-page summary sheets, do full timed practice tests 3 days before.",
    "Writing improvement: journal daily for 10 minutes, practice summarizing news articles in 3 sentences.",
    "Math improvement: never skip steps when solving problems, show all working, review every error to understand the mistake.",
    "Reading speed and retention: practice reading with a pointer, stop at end of each page to recall main ideas.",
]


class RAGTool:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = Chroma(
            collection_name="study_knowledge",
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        self._seed_if_empty()

    def _seed_if_empty(self):
        if self.vectorstore._collection.count() == 0:
            docs = [Document(page_content=tip) for tip in STUDY_TIPS]
            self.vectorstore.add_documents(docs)
            print(f"RAG: seeded {len(docs)} study tip documents.")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
```

- [ ] **Step 4: Verify RAG tool seeds and retrieves**

```bash
python -c "
from tools.rag_tool import RAGTool
rag = RAGTool()
results = rag.retrieve('how to improve math score', k=2)
for r in results:
    print('-', r[:80])
"
```

Expected: 2 relevant study tips printed (math-related).

- [ ] **Step 5: Verify web search tool**

```bash
python -c "
from tools.web_search_tool import WebSearchTool
w = WebSearchTool()
urls = w.search('how to improve math score for students', max_results=2)
print(urls)
"
```

Expected: list of 2 URLs (or empty list if rate limited).

- [ ] **Step 6: Commit**

```bash
git add tools/
git commit -m "feat: add Tavily web search tool and ChromaDB RAG tool with seeded knowledge base"
```

---

## Task 9: Create memory/session_memory.py

**Files:**
- Create: `memory/__init__.py`
- Create: `memory/session_memory.py`

- [ ] **Step 1: Create memory/__init__.py**

Empty file.

- [ ] **Step 2: Create memory/session_memory.py**

```python
from typing import Dict, List


class SessionMemory:
    def __init__(self):
        self._store: Dict[str, dict] = {}

    def save(self, student_id: str, data: dict):
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id].update({k: v for k, v in data.items() if k != "history"})

    def load(self, student_id: str) -> dict:
        return self._store.get(student_id, {})

    def get_history(self, student_id: str) -> List[dict]:
        return self._store.get(student_id, {}).get("history", [])

    def add_turn(self, student_id: str, role: str, content: str):
        if student_id not in self._store:
            self._store[student_id] = {"history": []}
        self._store[student_id]["history"].append({"role": role, "content": content})
```

- [ ] **Step 3: Verify**

```bash
python -c "
from memory.session_memory import SessionMemory
m = SessionMemory()
m.save('s1', {'goal': 'pass math'})
m.add_turn('s1', 'user', 'hello')
m.add_turn('s1', 'assistant', 'hi there')
print(m.load('s1'))
print(m.get_history('s1'))
"
```

Expected: dict with goal and history list with 2 turns.

- [ ] **Step 4: Commit**

```bash
git add memory/
git commit -m "feat: add session memory for cross-turn context"
```

---

## Task 10: Create agents/nodes.py

**Files:**
- Create: `agents/nodes.py`

- [ ] **Step 1: Create agents/nodes.py**

```python
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.state import AgentState
from tools.web_search_tool import WebSearchTool
from tools.rag_tool import RAGTool

_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)


class RouterNode:
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        system = (
            "Classify this student query. Reply with exactly one word only: "
            "STUDY if it's about academic performance, study plans, scores, learning gaps, or study advice. "
            "GENERAL if it's casual conversation, greetings, or unrelated to studying."
        )
        result = _llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=last_msg)]
        )
        state["is_study_query"] = "STUDY" in result.content.upper()
        return state


class DiagnosisNode:
    def __call__(self, state: AgentState) -> AgentState:
        data = state["student_data"]
        gaps = []
        if float(data.get("math_score", 100)) < 50:
            gaps.append("math (score below 50)")
        if float(data.get("reading_score", 100)) < 50:
            gaps.append("reading (score below 50)")
        if float(data.get("writing_score", 100)) < 50:
            gaps.append("writing (score below 50)")
        if float(data.get("attendance_rate", 1.0)) < 0.75:
            gaps.append("attendance (below 75%)")
        if float(data.get("daily_study_hours", 5)) < 2:
            gaps.append("study time (less than 2 hours/day)")
        if float(data.get("stress_level", 5)) > 7:
            gaps.append("stress management (high stress level)")
        if float(data.get("sleep_hours", 8)) < 6:
            gaps.append("sleep (less than 6 hours/night)")
        if float(data.get("motivation_score", 50)) < 30:
            gaps.append("motivation (low score)")
        state["learning_gaps"] = gaps if gaps else ["no major gaps detected"]
        return state


class PlannerNode:
    def __call__(self, state: AgentState) -> AgentState:
        data = state["student_data"]
        gaps = state["learning_gaps"]
        prediction = data.get("prediction", "unknown")
        prompt = (
            f"You are a study coach. A student's predicted outcome is '{prediction}'.\n"
            f"Their weak areas: {', '.join(gaps)}.\n"
            f"Their profile: daily study hours={data.get('daily_study_hours')}, "
            f"attendance rate={data.get('attendance_rate')}, "
            f"stress level={data.get('stress_level')}, "
            f"sleep hours={data.get('sleep_hours')}, "
            f"math score={data.get('math_score')}, "
            f"reading score={data.get('reading_score')}, "
            f"writing score={data.get('writing_score')}.\n"
            "Create a concise, actionable 7-day study plan targeting these weak areas. "
            "Be specific with daily tasks. Format as Day 1 through Day 7."
        )
        result = _llm.invoke([HumanMessage(content=prompt)])
        state["study_plan"] = result.content
        return state


class ResourceRetrieverNode:
    def __init__(self):
        self.rag = RAGTool()
        self.search = WebSearchTool()

    def __call__(self, state: AgentState) -> AgentState:
        gaps = state["learning_gaps"]
        query = f"study tips and resources for {', '.join(gaps[:2])}"
        rag_results = self.rag.retrieve(query, k=3)
        web_results = self.search.search(query, max_results=2)
        state["retrieved_resources"] = rag_results + web_results
        return state


class ResponseGeneratorNode:
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        history = state.get("session_history", [])[-6:]
        history_text = "\n".join(
            [f"{t['role'].capitalize()}: {t['content']}" for t in history]
        )

        if state.get("is_study_query"):
            resources_preview = "\n".join(
                [f"- {r[:120]}" for r in state.get("retrieved_resources", [])[:3]]
            )
            system = (
                "You are a friendly, encouraging AI study coach.\n"
                f"Student data: {state['student_data']}\n"
                f"Learning gaps identified: {', '.join(state.get('learning_gaps', []))}\n"
                f"Generated study plan:\n{state.get('study_plan', 'Not generated')}\n"
                f"Relevant resources:\n{resources_preview}\n"
                f"Conversation so far:\n{history_text}\n\n"
                "Respond helpfully and conversationally. Reference the student's specific data "
                "when relevant. Keep responses concise and encouraging."
            )
        else:
            system = (
                "You are a friendly AI study coach having a general conversation with a student.\n"
                f"Conversation so far:\n{history_text}\n\n"
                "Respond naturally, warmly, and helpfully."
            )

        result = _llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=last_msg)]
        )
        state["messages"] = [AIMessage(content=result.content)]
        return state


class MemoryNode:
    def __call__(self, state: AgentState) -> AgentState:
        msgs = state["messages"]
        last_human = next(
            (m.content for m in reversed(msgs) if isinstance(m, HumanMessage)), ""
        )
        last_ai = next(
            (m.content for m in reversed(msgs) if isinstance(m, AIMessage)), ""
        )
        history = list(state.get("session_history", []))
        if last_human:
            history.append({"role": "user", "content": last_human})
        if last_ai:
            history.append({"role": "assistant", "content": last_ai})
        state["session_history"] = history
        return state
```

- [ ] **Step 2: Verify import**

```bash
python -c "from agents.nodes import RouterNode, DiagnosisNode, PlannerNode, ResourceRetrieverNode, ResponseGeneratorNode, MemoryNode; print('nodes OK')"
```

Expected: `nodes OK`

- [ ] **Step 3: Commit**

```bash
git add agents/nodes.py
git commit -m "feat: add LangGraph agent nodes (router, diagnose, plan, retrieve, respond, memory)"
```

---

## Task 11: Create agents/study_coach_agent.py

**Files:**
- Create: `agents/study_coach_agent.py`

- [ ] **Step 1: Create agents/study_coach_agent.py**

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.state import AgentState
from agents.nodes import (
    RouterNode,
    DiagnosisNode,
    PlannerNode,
    ResourceRetrieverNode,
    ResponseGeneratorNode,
    MemoryNode,
)


def _route(state: AgentState) -> str:
    return "diagnose" if state.get("is_study_query") else "respond"


class StudyCoachAgent:
    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("router", RouterNode())
        g.add_node("diagnose", DiagnosisNode())
        g.add_node("plan", PlannerNode())
        g.add_node("retrieve", ResourceRetrieverNode())
        g.add_node("respond", ResponseGeneratorNode())
        g.add_node("memory", MemoryNode())

        g.set_entry_point("router")
        g.add_conditional_edges(
            "router",
            _route,
            {"diagnose": "diagnose", "respond": "respond"},
        )
        g.add_edge("diagnose", "plan")
        g.add_edge("plan", "retrieve")
        g.add_edge("retrieve", "respond")
        g.add_edge("respond", "memory")
        g.add_edge("memory", END)

        return g.compile()

    def run(self, state: AgentState) -> AgentState:
        return self.graph.invoke(state)

    def chat(
        self,
        user_message: str,
        student_data: dict,
        session_history: list,
    ) -> tuple[str, list]:
        state = AgentState(
            messages=[HumanMessage(content=user_message)],
            student_data=student_data,
            learning_gaps=[],
            study_plan=None,
            retrieved_resources=[],
            session_history=session_history,
            current_goal=None,
            is_study_query=False,
        )
        result = self.run(state)
        ai_reply = result["messages"][-1].content
        updated_history = result["session_history"]
        return ai_reply, updated_history
```

- [ ] **Step 2: Smoke-test the agent (uses Groq API — needs internet)**

```bash
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
from agents.study_coach_agent import StudyCoachAgent
agent = StudyCoachAgent()
reply, history = agent.chat(
    'Hello! Can you help me?',
    student_data={'math_score': 45, 'prediction': 'Fail', 'daily_study_hours': 1.5},
    session_history=[]
)
print('Reply:', reply[:200])
print('History turns:', len(history))
"
```

Expected: A natural response from the agent, history has 2 entries (user + assistant).

- [ ] **Step 3: Commit**

```bash
git add agents/study_coach_agent.py
git commit -m "feat: wire LangGraph study coach agent with conditional routing"
```

---

## Task 12: Create pages/dashboard.py

**Files:**
- Create: `pages/__init__.py`
- Create: `pages/dashboard.py`

- [ ] **Step 1: Create pages/__init__.py**

Empty file.

- [ ] **Step 2: Create pages/dashboard.py**

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import StudentPredictor
from src.ui_components import UIBuilder

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

@st.cache_resource
def load_predictor():
    return StudentPredictor()

predictor = load_predictor()
ui = UIBuilder()
ui.load_css()

if not predictor.is_ready():
    st.error("⚠️ Models not found. Run `python train_model.py` first.")
    st.stop()

st.title("🎓 Student Performance Predictor")
st.markdown("Enter your academic details below to get a Pass/Fail prediction.")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        parental_education = st.slider("Parental Education Level", 1, 6, 3,
            help="1=No formal education, 6=Postgraduate")
        daily_study = st.slider("Daily Study Hours", 0.0, 12.0, 3.0, step=0.5)
        attendance = st.slider("Attendance Rate", 0.0, 1.0, 0.85, step=0.01,
            help="0.0 = 0%, 1.0 = 100%")

    with col2:
        sleep_hours = st.slider("Sleep Hours per Night", 3.0, 12.0, 7.0, step=0.5)
        stress_level = st.slider("Stress Level", 1, 10, 5,
            help="1 = very low stress, 10 = extremely high stress")
        motivation = st.slider("Motivation Score", 0, 100, 60)

    with col3:
        math_score = st.slider("Math Score", 0, 100, 65)
        reading_score = st.slider("Reading Score", 0, 100, 65)
        writing_score = st.slider("Writing Score", 0, 100, 65)

    submitted = st.form_submit_button("🔍 Predict My Outcome", type="primary")

if submitted:
    input_data = {
        "parental_education_level": parental_education,
        "daily_study_hours": daily_study,
        "attendance_rate": attendance,
        "sleep_hours": sleep_hours,
        "stress_level": stress_level,
        "motivation_score": motivation,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }
    df = pd.DataFrame([input_data])
    prob, pred, cluster = predictor.predict_bundle(df)

    st.session_state.student_data = {
        **input_data,
        "prediction": "Pass" if pred[0] == 1 else "Fail",
        "probability": float(prob[0]),
        "cluster": int(cluster[0]),
    }
    st.session_state.prediction_done = True

    st.markdown("---")
    ui.render_prediction_card(float(prob[0]), int(pred[0]), int(cluster[0]))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Score Breakdown")
        fig, ax = plt.subplots(figsize=(6, 3))
        scores = [math_score, reading_score, writing_score]
        labels = ["Math", "Reading", "Writing"]
        colors = ["#6366f1" if s >= 50 else "#ef4444" for s in scores]
        ui.mpl_bar(ax, labels, scores, "Subject Scores", colors)
        ax.axhline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("📋 Actionable Tips")
        recs = predictor.get_student_recommendations(pd.Series(input_data))
        for rec in recs:
            st.markdown(f"- {rec}")

    st.markdown("---")
    st.markdown("### Ready to improve? Chat with your AI Study Coach below 👇")
    if st.button("💬 Open Study Coach Chat", type="primary"):
        st.switch_page("pages/chat_interface.py")
```

- [ ] **Step 3: Commit**

```bash
git add pages/
git commit -m "feat: add student dashboard page with prediction form and charts"
```

---

## Task 13: Create pages/chat_interface.py

**Files:**
- Create: `pages/chat_interface.py`

- [ ] **Step 1: Create pages/chat_interface.py**

```python
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from agents.study_coach_agent import StudyCoachAgent
from src.ui_components import UIBuilder

st.set_page_config(page_title="AI Study Coach", layout="wide")

ui = UIBuilder()
ui.load_css()

@st.cache_resource
def load_agent():
    return StudyCoachAgent()

agent = load_agent()

st.title("🤖 AI Study Coach")

if "student_data" not in st.session_state or not st.session_state.get("prediction_done"):
    st.info("Please go to the **Dashboard** page first and run a prediction to start your coaching session.")
    if st.button("← Go to Dashboard"):
        st.switch_page("pages/dashboard.py")
    st.stop()

student_data = st.session_state.student_data
prediction = student_data.get("prediction", "Unknown")
prob = student_data.get("probability", 0.0)
cluster = student_data.get("cluster", 0)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Outcome", prediction)
col2.metric("Pass Probability", f"{prob*100:.1f}%")
col3.metric("Student Group", f"Group {cluster + 1}")

st.markdown("---")
st.markdown("Ask me anything — study tips, a weekly plan, subject help, or just chat!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_history" not in st.session_state:
    st.session_state.session_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your study coach..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply, updated_history = agent.chat(
                user_message=prompt,
                student_data=student_data,
                session_history=st.session_state.session_history,
            )
        st.write(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.session_history = updated_history

if st.session_state.chat_history:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.session_history = []
        st.rerun()
```

- [ ] **Step 2: Commit**

```bash
git add pages/chat_interface.py
git commit -m "feat: add LangGraph study coach chat interface"
```

---

## Task 14: Update app.py

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Rewrite app.py as multi-page navigation entry point**

```python
import streamlit as st

dashboard = st.Page("pages/dashboard.py", title="Dashboard", icon="🎓", default=True)
chat = st.Page("pages/chat_interface.py", title="Study Coach", icon="🤖")

pg = st.navigation([dashboard, chat])
pg.run()
```

- [ ] **Step 2: Run the app and verify both pages load**

```bash
streamlit run app.py
```

Expected: App opens at localhost:8501, shows Dashboard page with sliders. Sidebar (or top nav) shows "Dashboard" and "Study Coach" links. Submitting the form shows prediction card. "Open Study Coach Chat" button navigates to chat page. Chat page shows coach interface.

- [ ] **Step 3: Test full flow manually**
  - Enter student values → click Predict → verify prediction card appears
  - Adjust sliders to low scores → verify tips appear
  - Click "Open Study Coach Chat" → verify student metrics shown at top
  - Type "What should I study this week?" → verify agent responds with study advice referencing the student's scores
  - Type "What's 2+2?" → verify agent responds conversationally without triggering study pipeline unnecessarily
  - Clear chat → verify history resets

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: update app.py to multi-page navigation with Dashboard and Study Coach"
```

---

## Task 15: Add .gitignore entries and final cleanup

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Ensure these are in .gitignore**

```
.env
chroma_db/
models/
__pycache__/
*.pyc
.DS_Store
```

- [ ] **Step 2: Final verification — run full smoke test**

```bash
python -c "
from src.inference import StudentPredictor
from agents.study_coach_agent import StudyCoachAgent
p = StudentPredictor()
print('Predictor ready:', p.is_ready())
a = StudyCoachAgent()
print('Agent built OK')
"
```

Expected: Both print statements succeed.

- [ ] **Step 3: Final commit**

```bash
git add .gitignore
git commit -m "chore: finalize gitignore for env, models, chroma_db"
```
