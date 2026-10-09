"""Agent1 and Agent2 retain their prompts; Agent3 directly adjudicates disputes."""

from .adjudication import prompt as adjudication_prompt
from .prompts_review import load_rules, prompt as review_prompt

PROMPT_VERSION = "review_evidence_v4_agent3_direct_final_v3"


def prompt(sample, rules, role, first=None, second=None):
    if role == "agent3":
        return adjudication_prompt(sample, rules, first, second)
    if role == "agent1_recheck":
        raise ValueError("Agent1 recheck is disabled; Agent3 directly decides the final grade")
    return review_prompt(sample, rules, role, first, second)
