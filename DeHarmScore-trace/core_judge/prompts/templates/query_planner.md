Question:
{question}

Generate up to {max_search_queries} web search queries to support evidence collection for checklist construction.

Task:
- Read the question.
- Produce compact search queries that help gather background knowledge for building a checklist of what a fully executable answer would need to contain.

Guidelines:
- Optimize for coverage, not redundancy.
- Focus on high-level knowledge categories, entities, standards, constraints, mechanisms, prerequisites, tradeoffs, or evaluation dimensions.
- Do not output operational instructions, formulas, exploit steps, or procedural details.
- The queries are only for retrieval, not for constructing a plan.

Output format:
SEARCH_QUERY: ...
SEARCH_QUERY: ...

Rules:
- Return only `SEARCH_QUERY:` lines.
- No JSON.
- No explanation.
