import os
import re
from dotenv import load_dotenv

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Load variables from the .env file into the environment
load_dotenv()

# Verify the key is loaded (optional debug line)
if not os.getenv("GOOGLE_API_KEY"):
  raise ValueError("GOOGLE_API_KEY not found in .env file!")


# 1. Define the Shared State structure passed between agents
class AgentState(TypedDict):
  problem_statement: str
  solution_code: str
  manager_feedback: str
  accuracy_score: float
  iteration_count: int


# Initialize the LLM (using Gemini 3.8 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)


# 2. Define the Developer Agent Node
def developer_node(state: AgentState):
  print(f"\n--- [Developer] Working on iteration {state.get('iteration_count', 1)} ---")

  problem = state["problem_statement"]
  feedback = state.get(
      "manager_feedback", "None. This is the first implementation attempt."
  )

  prompt = f"""
    You are an expert software developer. 
    Problem Statement: {problem}
    
    Manager's Feedback from last iteration: {feedback}
    
    Write or refine the complete, working code/demo solution to address the problem and incorporate the feedback.
    Provide clear, functional code.
    """

  response = llm.invoke([HumanMessage(content=prompt)])
  current_iteration = state.get("iteration_count", 0) + 1

  return {
      "solution_code": response.content,
      "iteration_count": current_iteration,
  }


# 3. Define the Manager Agent Node (Evaluator)
def manager_node(state: AgentState):
  print("\n--- [Manager] Evaluating solution ---")

  code = state["solution_code"]
  problem = state["problem_statement"]

  prompt = f"""
    You are a strict engineering manager. Evaluate the following solution against the problem statement.
    
    Problem Statement: {problem}
    Solution/Code: {code}
    
    Task:
    1. Critically analyze if the solution fully works and meets requirements.
    2. Estimate an accuracy percentage score from 0 to 100 based on code correctness, completeness, and edge cases.
    3. Provide constructive feedback on what needs to be fixed if the score is below 97.
    
    Format your response EXACTLY like this:
    SCORE: [integer between 0 and 100]
    FEEDBACK: [your detailed feedback for the developer]
    """

  response = llm.invoke([HumanMessage(content=prompt)])
  content = response.content
  content = response.content if isinstance(response.content, str) else str(response.content)

  # Simple parsing of the manager's response
  match = re.search(r"SCORE:\s*(\d+)", content, re.IGNORECASE)
  score = float(match.group(1)) if match else 80.0

  print(f"Manager evaluated score: {score}%")

  return {"manager_feedback": content, "accuracy_score": score}


# 4. Define the Router / Conditional Edge
def should_continue(state: AgentState):
  score = state["accuracy_score"]
  iterations = state["iteration_count"]

  # Stop if accuracy hits target or if it exceeds max iterations (safety circuit breaker)
  if score >= 97.0 or iterations >= 5:
    print(f"\nTarget reached or max iterations hit! Final Accuracy: {score}%")
    return "end"
  else:
    print(
        f"\nAccuracy is {score}% (< 97%). Sending back to Developer for refinement..."
    )
    return "retry"


# 5. Build and Compile the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("developer", developer_node)
workflow.add_node("manager", manager_node)

# Set entry point
workflow.set_entry_point("developer")

# Add edges (Developer -> Manager -> Conditional Router)
workflow.add_edge("developer", "manager")

workflow.add_conditional_edges(
    "manager",
    should_continue,
    {
        "retry": "developer",
        "end": END,
    },
)

app = workflow.compile()


# 6. Execution Block
if __name__ == "__main__":
  initial_state = {
      "problem_statement": (
        "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation."
      ),
      "solution_code": "",
      "manager_feedback": "",
      "accuracy_score": 0.0,
      "iteration_count": 0,
  }

  print("Starting Multi-Agent Workflow...")
  final_state = app.invoke(initial_state)

  print("\n========== FINAL RESULT ==========")
  print(f"Total Iterations: {final_state['iteration_count']}")
  print(f"Final Accuracy: {final_state['accuracy_score']}%")
  print("\nFinal Solution Code:\n", final_state["solution_code"])