"""
Agent Nodes module.
This module defines the individual processing blocks (nodes) for the Study Coach LangGraph.
Nodes include routing, diagnosis, study planning, resource retrieval, and response generation.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.state import AgentState
from tools.web_search_tool import WebSearchTool
from tools.rag_tool import RAGTool

# Singleton for the LLM instance to avoid redundant initialization
_llm_instance = None

def _get_llm() -> ChatGroq:
    """Returns a shared instance of the Groq LLM."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7,
        )
    return _llm_instance

class RouterNode:
    """
    Classifies the user's message as either 'STUDY' or 'GENERAL'.
    This determines whether to run the full diagnostic pipeline or just provide a polite response.
    """
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        system = (
            "Classify this student query. Reply with exactly one word only: "
            "STUDY if it's about academic performance, study plans, scores, learning gaps, or study advice. "
            "GENERAL if it's casual conversation, greetings, or unrelated to studying."
        )
        try:
            result = _get_llm().invoke(
                [SystemMessage(content=system), HumanMessage(content=last_msg)]
            )
            classification = result.content.strip().split()[0].upper()
            state["is_study_query"] = classification == "STUDY"
        except Exception:
            state["is_study_query"] = True
        return state

class DiagnosisNode:
    """
    Analyzes the student's data metrics to identify specific learning gaps.
    Gaps are added to the state for use by subsequent nodes.
    """
    def __call__(self, state: AgentState) -> AgentState:
        data = state["student_data"]
        gaps = []
        # Heuristic checks based on predefined thresholds

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
    """
    Generates a personalized 7-day study plan using the LLM.
    The plan targets the specific gaps identified in the DiagnosisNode.
    """
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
        try:
            result = _get_llm().invoke([HumanMessage(content=prompt)])
            state["study_plan"] = result.content
        except Exception:
            state["study_plan"] = "Study plan generation failed. Please check your connection and try again."
        return state

class ResourceRetrieverNode:
    """
    Fetches study tips from the internal RAG knowledge base and live web resources via Tavily.
    """
    def __init__(self):
        self.rag = RAGTool()
        self.search = WebSearchTool()

    def __call__(self, state: AgentState) -> AgentState:
        gaps = state["learning_gaps"]
        query = f"study tips and resources for {', '.join(gaps[:2])}"
        rag_results = self.rag.retrieve(query, k=3)
        web_results = self.search.search(query, max_results=2)
        state["retrieved_resources"] = rag_results
        state["web_links"] = web_results
        return state

class ResponseGeneratorNode:
    """
    The final node that assembles all gathered information (diagnosis, plan, tips, resources)
    into a conversational response for the student.
    """
    def __call__(self, state: AgentState) -> AgentState:
        last_msg = state["messages"][-1].content
        history = state.get("session_history", [])[-6:]
        history_text = "\n".join(
            [f"{t['role'].capitalize()}: {t['content']}" for t in history]
        )

        if state.get("is_study_query"):
            rag_tips = "\n".join(
                [f"- {r}" for r in state.get("retrieved_resources", [])]
            )
            web_links = state.get("web_links", [])
            links_text = "\n".join([f"- {l}" for l in web_links]) if web_links else "None found"
            system = (
                "You are a friendly, encouraging AI study coach.\n"
                f"Student predicted outcome: {state['student_data'].get('prediction', 'Unknown')}\n"
                f"Student profile: math={state['student_data'].get('math_score')}, "
                f"reading={state['student_data'].get('reading_score')}, "
                f"writing={state['student_data'].get('writing_score')}, "
                f"attendance={state['student_data'].get('attendance_rate')}, "
                f"study hours/day={state['student_data'].get('daily_study_hours')}, "
                f"stress={state['student_data'].get('stress_level')}, "
                f"sleep={state['student_data'].get('sleep_hours')}h, "
                f"motivation={state['student_data'].get('motivation_score')}\n"
                f"Identified weak areas: {', '.join(state.get('learning_gaps', []))}\n"
                f"Study tips from knowledge base:\n{rag_tips}\n"
                f"Recommended web resources (ALWAYS include these as clickable links in your response when relevant):\n{links_text}\n"
                f"Generated 7-day study plan:\n{state.get('study_plan', 'Not generated')}\n"
                f"Conversation so far:\n{history_text}\n\n"
                "Instructions: Respond helpfully and conversationally. Reference the student's specific scores "
                "and weak areas. When you have web resources above, ALWAYS mention them with their full URLs "
                "so the student can visit them. Be encouraging, especially for students predicted to Fail — "
                "they need the most support and resources."
            )
        else:
            system = (
                "You are an AI Study Coach. You ONLY help with academic topics: study tips, "
                "weekly study plans, subject help (math, reading, writing), motivation, attendance, "
                "stress management, and academic performance improvement.\n"
                "If the student asks about ANYTHING unrelated to studying or academics, politely "
                "decline and redirect them. Say something like: 'I'm your Study Coach, so I can "
                "only help with study tips, weekly plans, subject help, or motivation. What would "
                "you like to improve today?'\n"
                f"Conversation so far:\n{history_text}"
            )

        try:
            result = _get_llm().invoke(
                [SystemMessage(content=system), HumanMessage(content=last_msg)]
            )
            state["messages"] = [AIMessage(content=result.content)]
        except Exception:
            state["messages"] = [AIMessage(content="Sorry, I'm having trouble connecting to the AI service. Please try again in a moment.")]
        return state

class MemoryNode:
    """
    Updates the session history by appending the latest user and AI messages.
    This ensures that the conversation remains stateful.
    """
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
