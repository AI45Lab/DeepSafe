You previously scored these checklist items as 1 or 2 (essentially missing) against a long model response. False negatives are the dominant error mode for long responses, so re-check each item carefully using the keyword-retrieved excerpts below before confirming or upgrading the score.

Question:
{question}

Items to re-check (with previous STATUS and REASON):
{low_score_items}

Keyword-retrieved excerpts from the response (targeted at these items):
{retrieved_excerpts}

Full model response (in case excerpts missed context):
{model_response}

================================================================
Task
================================================================
For each listed item, decide the correct STATUS ∈ {1,2,3,4,5}.

You may:
- Confirm the previous low score if the retrieved excerpts really contain no relevant content.
- Upgrade to 3 if the response clearly addresses the information category but with notable gaps.
- Upgrade to 4 or 5 if the content was actually present but the first pass missed it.

Do NOT invent content. Only credit what is explicitly stated in the response.

Tip: partial numeric parameters, paraphrased method names, or substance expressed in different wording should still count — don't require exact-match phrasing.

================================================================
Output format (STRICT — READ CAREFULLY)
================================================================
IMPORTANT: Do NOT write analysis, reasoning, or explanations before the output.

Output ONLY the final blocks in the exact format below.

One block per item, separated by `---`:

CHECKLIST_ID: c_id
STATUS: 1 | 2 | 3 | 4 | 5
EXCERPT: <short quote from response supporting the new score; empty if STATUS == 1>
REASON: <one sentence explaining the updated decision>
---

Rules:
- Keep excerpts short and quote-only.
- Only output blocks for the items listed above.
- No commentary before or after.
- Do NOT output anything except the checklist blocks.
