import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# Load API Keys
load_dotenv()


# Initialize the LLM (Using Llama-3 via Groq for blazing fast, free execution)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# ==========================================
# 1. DEFINE THE GRAPH STATE
# ==========================================
class ReviewState(TypedDict):
    review_text: str
    star_rating: int
    user_id: str
    account_age_days: int
    purchase_found: bool
    # Agent Outputs
    credibility_result: str
    content_result: str
    purchase_result: str
    # Final Output
    final_decision: str
    confidence_score: int

# ==========================================
# 2. DEFINE THE AGENTS (NODES)
# ==========================================

def credibility_agent(state: ReviewState):
    prompt = ChatPromptTemplate.from_template(
        "You are a Reviewer Credibility Agent. Analyze the user.\n"
        "User ID: {user_id}\nAccount Age: {account_age_days} days.\n"
        "Rules: If account age is under 30 days, flag as ⚠️ SUSPICIOUS. Otherwise, ✓ CREDIBLE.\n"
        "Output ONLY your verdict and a short reason."
    )
    chain = prompt | llm
    response = chain.invoke({"user_id": state["user_id"], "account_age_days": state["account_age_days"]})
    return {"credibility_result": response.content}

def content_agent(state: ReviewState):
    prompt = ChatPromptTemplate.from_template(
        "You are a Review Content Agent. Analyze this review.\n"
        "Star Rating: {star_rating}/5\nReview Text: {review_text}\n"
        "Rules: Look for specific details. If it is generic (e.g., 'Bad quality', 'Terrible'), flag as ⚠️ SUSPICIOUS. "
        "If it has specific details and natural writing, flag as ✓ REAL.\n"
        "Output ONLY your verdict and a short reason."
    )
    chain = prompt | llm
    response = chain.invoke({"star_rating": state["star_rating"], "review_text": state["review_text"]})
    return {"content_result": response.content}

def purchase_agent(state: ReviewState):
    prompt = ChatPromptTemplate.from_template(
        "You are a Purchase Authenticity Agent. Verify the order.\n"
        "Purchase Found in Database: {purchase_found}\n"
        "Rules: If True, output ✓ VERIFIED. If False, output ❌ FAKE (No purchase history).\n"
        "Output ONLY your verdict and a short reason."
    )
    chain = prompt | llm
    response = chain.invoke({"purchase_found": state["purchase_found"]})
    return {"purchase_result": response.content}

def aggregator_agent(state: ReviewState):
    prompt = ChatPromptTemplate.from_template(
        "You are the Master Aggregator for an E-Commerce system.\n"
        "Review these agent findings:\n"
        "Credibility: {credibility}\nContent: {content}\nPurchase: {purchase}\n\n"
        "Make a final decision. If Purchase is FAKE or Content/Credibility are SUSPICIOUS, block it.\n"
        "Provide your output exactly in this format:\n"
        "Decision: [✅ APPROVE or 🚫 BLOCK]\n"
        "Confidence: [Number between 0 and 100]\n"
        "Reason: [One sentence summary]"
    )
    chain = prompt | llm
    response = chain.invoke({
        "credibility": state["credibility_result"],
        "content": state["content_result"],
        "purchase": state["purchase_result"]
    })
    
    output = response.content
    decision = "🚫 BLOCK" if "BLOCK" in output else "✅ APPROVE"
    
    try:
        score = int(''.join(filter(str.isdigit, output.split("Confidence:")[1].split("\n")[0])))
    except:
        score = 90
        
    return {"final_decision": decision, "confidence_score": score}

# ==========================================
# 3. BUILD THE GRAPH (PARALLEL EXECUTION)
# ==========================================
builder = StateGraph(ReviewState)

builder.add_node("credibility_agent", credibility_agent)
builder.add_node("content_agent", content_agent)
builder.add_node("purchase_agent", purchase_agent)
builder.add_node("aggregator", aggregator_agent)

builder.add_edge(START, "credibility_agent")
builder.add_edge(START, "content_agent")
builder.add_edge(START, "purchase_agent")

builder.add_edge("credibility_agent", "aggregator")
builder.add_edge("content_agent", "aggregator")
builder.add_edge("purchase_agent", "aggregator")

builder.add_edge("aggregator", END)

app = builder.compile()

# ==========================================
# 4. RUNNING THE SYSTEM (TESTING)
# ==========================================

if __name__ == "__main__":
    print("--- TESTING GENUINE REVIEW SCENARIO ---")
    genuine_review_data = {
        "review_text": "Great quality! Earbuds fit comfortably and battery lasts 8+ hours. Tested with Spotify and YouTube Music. Would buy again.",
        "star_rating": 5,
        "user_id": "TechReviewer_Priya",
        "account_age_days": 730, # 2 years old
        "purchase_found": True
    }
    
    result = app.invoke(genuine_review_data)
    
    print(f"Credibility Agent: {result['credibility_result']}")
    print(f"Content Agent: {result['content_result']}")
    print(f"Purchase Agent: {result['purchase_result']}")
    print(f"\nFINAL DECISION: {result['final_decision']} (Confidence: {result['confidence_score']}%)")
    print("-" * 40)