"""
RAG (Retrieval-Augmented Generation) Tool for the Study Coach.
This tool manages a local vector database (Chroma) containing study tips and knowledge.
It allows the agent to retrieve relevant advice based on user queries.
"""

import os
from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import BASE_DIR

# Default directory for persistent vector storage
_DEFAULT_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# Static list of study tips used to seed the vector database
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
    """
    A tool for retrieving domain-specific knowledge using semantic search.
    Initializes a Chroma vector store with HuggingFace embeddings.
    """
    def __init__(self, persist_dir: str = _DEFAULT_PERSIST_DIR):
        """Initializes embeddings and loads/creates the vector store."""
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
        """Populates the vector store with initial data if it contains no documents."""
        try:
            existing = self.vectorstore.get()
            already_seeded = len(existing["ids"]) > 0
        except Exception:
            already_seeded = False
        if not already_seeded:
            docs = [Document(page_content=tip) for tip in STUDY_TIPS]
            self.vectorstore.add_documents(docs)
            print(f"RAG: seeded {len(docs)} study tip documents.")

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieves the top k most relevant documents for a given query.
        
        Args:
            query: The search query (natural language).
            k: Number of documents to return.
            
        Returns:
            List of document content strings.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

