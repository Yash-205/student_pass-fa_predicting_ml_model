"""
Study Coach Agent module.
This module defines the StudyCoachAgent, which orchestrates a multi-node LangGraph
to process student queries, diagnose gaps, plan study sessions, and retrieve resources.
"""

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
    """Routing function to decide if the query requires a full diagnosis or just a response."""
    return "diagnose" if state.get("is_study_query") else "respond"

class StudyCoachAgent:
    """
    Orchestrator for the AI Study Coach.
    Builds and manages a stateful computation graph (LangGraph) for student support.
    """
    def __init__(self):
        """Initializes the agent by building the internal graph."""
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Defines the structure of the LangGraph.
        Nodes: router -> diagnose -> plan -> retrieve -> respond -> memory.
        """
        g = StateGraph(AgentState)

        # Register nodes
        g.add_node("router", RouterNode())
        g.add_node("diagnose", DiagnosisNode())
        g.add_node("plan", PlannerNode())
        g.add_node("retrieve", ResourceRetrieverNode())
        g.add_node("respond", ResponseGeneratorNode())
        g.add_node("memory", MemoryNode())

        # Define transitions
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
        """Executes the graph for a given state."""
        return self.graph.invoke(state)

    def chat(
        self,
        user_message: str,
        student_data: dict,
        session_history: list,
    ) -> tuple[str, list]:
        """
        High-level interface for the chat UI.
        Wraps message into state, runs the graph, and returns the response and updated history.
        """
        state = AgentState(
            messages=[HumanMessage(content=user_message)],
            student_data=student_data,
            learning_gaps=[],
            study_plan=None,
            retrieved_resources=[],
            web_links=[],
            session_history=session_history,
            is_study_query=False,
        )
        result = self.run(state)
        ai_reply = result["messages"][-1].content
        updated_history = result["session_history"]
        return ai_reply, updated_history

