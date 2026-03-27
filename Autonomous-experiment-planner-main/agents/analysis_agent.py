# agents/analysis_agent.py

import json
import re
from typing import Dict, Any, List, Optional
from groq import Groq
import os

from core.state import AgentState, ResearchGap, Hypothesis


# ── Groq Model Call (REPLACES OLLAMA) ─────────────────────────────────────

def call_local_model(prompt: str, temperature: float = 0.3) -> str:
    """
    Uses Groq cloud model instead of Ollama.
    """

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1200
    )

    return response.choices[0].message.content.strip()


# ── JSON Extraction ───────────────────────────────────────────────────────

def extract_json_from_response(response_text: str) -> Optional[Dict]:
    if not response_text:
        return None

    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except:
                pass

    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(response_text[start:end + 1])
        except:
            pass

    try:
        return json.loads(response_text.strip())
    except:
        return None


# ── PROMPTS ───────────────────────────────────────────────────────────────

def build_gap_analysis_prompt(topic, context):
    return f"""
Analyze research topic: {topic}

Find 4 research gaps.

Return JSON:
{{
 "gaps":[
  {{"gap_id":"GAP_001","title":"...","description":"...","importance":"...","supporting_evidence":"...","severity":"high"}}
 ],
 "primary_gap_id":"GAP_001"
}}
"""


def build_hypothesis_prompt(topic, gap, context):
    return f"""
Generate hypothesis for topic: {topic}
Gap: {gap['description']}

Return JSON:
{{
 "statement":"...",
 "rationale":"...",
 "based_on_gap_id":"{gap['gap_id']}"
}}
"""


# ── MAIN AGENT ────────────────────────────────────────────────────────────

def run_analysis_agent(state: AgentState) -> Dict[str, Any]:

    print("\n=== ANALYSIS AGENT (Groq) ===")

    topic = state.get("research_topic", "")
    context = state.get("retrieval_context", {})

    if not context:
        return {"error_message": "No context", "current_stage": "error"}

    # STEP 1 — GAP ANALYSIS
    gap_prompt = build_gap_analysis_prompt(topic, context)
    gap_response = call_local_model(gap_prompt)

    gap_data = extract_json_from_response(gap_response)

    if not gap_data:
        gap_data = {
            "gaps": [{
                "gap_id": "GAP_001",
                "title": "Generalization Gap",
                "description": "Models fail in real-world data",
                "importance": "Important",
                "supporting_evidence": "Literature shows it",
                "severity": "high"
            }],
            "primary_gap_id": "GAP_001"
        }

    gaps = gap_data["gaps"]

    selected_gap = gaps[0]

    # STEP 2 — HYPOTHESIS
    hyp_prompt = build_hypothesis_prompt(topic, selected_gap, context)
    hyp_response = call_local_model(hyp_prompt)

    hyp_data = extract_json_from_response(hyp_response)

    if not hyp_data:
        hyp_data = {
            "statement": "Improving model robustness improves results",
            "rationale": "Fixes gap",
            "based_on_gap_id": selected_gap["gap_id"]
        }

    hypothesis = Hypothesis(
        statement=hyp_data["statement"],
        rationale=hyp_data["rationale"],
        based_on_gap_id=hyp_data["based_on_gap_id"]
    )

    return {
        "identified_gaps": gaps,
        "selected_gap": selected_gap,
        "hypothesis": hypothesis,
        "current_stage": "planning",
        "error_message": None
    }