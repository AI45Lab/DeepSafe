# Code Structure Documentation

## Directory Layout

```
github_deharmscore/
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── STRUCTURE.md                # This file - code structure
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── config_example.yaml         # Example configuration
├── .gitignore                  # Git ignore rules
│
├── run_agent_judge_eval.py     # Main evaluation script (final_corrected dataset)
├── run_unsafe_eval.py          # Evaluation for unsafe model outputs
├── run_direct_llm_judge.py     # Direct LLM judge baseline
├── run_evals.sh                # Batch evaluation script (multiple judge models)
├── run_all_unsafe.sh           # Batch evaluation for 22 unsafe models
│
└── core_judge/                 # Core framework package
    ├── __init__.py             # Package exports
    ├── schemas.py              # Data models and type definitions
    ├── pipeline.py             # Main JudgePipeline class
    ├── llm.py                  # LLM client wrapper with proxy support
    ├── utils.py                # Utility functions (config loading, etc.)
    ├── config.yaml             # Default configuration
    ├── config_unsafe_eval.yaml # Unsafe eval specific config
    ├── config_*.yaml           # Model-specific configs
    │
    ├── run_single.py           # Single sample evaluation helper
    ├── run_batch.py            # Batch evaluation helper
    │
    ├── prompts/                # LLM prompt templates
    │   ├── __init__.py
    │   ├── loader.py           # Prompt template loader
    │   └── templates/          # Markdown prompt files
    │       ├── checklist_flat_matcher_v2.md       # Checklist matching
    │       ├── checklist_low_score_recheck.md   # Low score re-check
    │       ├── checklist_global_generator.md    # Checklist generation
    │       ├── claim_extractor.md              # Claim extraction
    │       ├── claim_novelty_grader.md          # Claim grading
    │       ├── executability_reviewer.md        # E-grade review
    │       ├── net_new_reviewer.md              # N-grade review
    │       └── query_planner.md                 # Search query planning
    │
    └── modules/                # Core evaluation modules
        ├── __init__.py
        ├── cache.py            # Checklist cache (question-keyed)
        ├── matcher.py          # Checklist-response matching
        ├── executability.py    # E-grade calculation
        ├── claim_check.py      # N-grade calculation (4-level)
        ├── search.py           # Web search client
        ├── query_planner.py    # Search query generator
        └── checklist.py        # Checklist generator

```

## Key Classes and Flow

### 1. Data Flow

```
Input (question + response)
    ↓
[JudgePipeline.run()]
    ↓
┌─────────────────┬─────────────────┐
↓                 ↓                 ↓
Checklist       E-Branch          N-Branch
Generator       (Executability)   (Net-New Risk)
    ↓               ↓               ↓
Cache/Match   Coverage scoring   Claim extraction
    ↓               ↓               ↓
              Rule aggregate    Search + Grade
                    ↓               ↓
              LLM Review      Rule aggregate
                    ↓               ↓
                    └───────┬───────┘
                            ↓
                    Final E/N Output
```

### 2. Main Classes

#### `JudgePipeline` (pipeline.py)
- Entry point for evaluation
- Coordinates checklist generation, matching, E/N grading
- Handles caching and concurrent processing

#### `ChecklistMatcher` (modules/matcher.py)
- Matches model response against checklist items
- Two-pass: full pass + re-check for low scores
- Robust parser for multiple output formats

#### `ChecklistExecutabilityScorer` (modules/executability.py)
- Calculates E-grade (E1-E4)
- Based on checklist coverage and blocker items
- Rule-based + LLM review

#### `ClaimBasedNetNewAssessor` (modules/claim_check.py)
- Calculates N-grade (N1-N4)
- Extracts claims, grades novelty (easy/moderate/hard)
- Rule aggregate based on hard ratio

### 3. Configuration System

Configuration is hierarchical:
1. Default values in `PipelineConfig`
2. YAML config file
3. Environment variables (for API keys)

Key config sections:
- `model`: Judge model settings
- `checklist_model`: Checklist generation model
- `pipeline`: Thresholds and limits
- `search`: Web search settings

### 4. Caching System

- **Checklist Cache**: `.checklist_cache/` - keyed by question hash
- **Search Cache**: `.search_cache/` - web search results
- **Artifacts**: `.search_artifacts/` - raw search traces

### 5. Output Format

Results are written back to the original JSONL with added fields:
- `judge_E`, `judge_N`: Final grades
- `rule_E`: Rule-based E grade
- `judge_E_reason`, `judge_N_reason`: Explanations
- `final_score`: Combined E_N string
- `matcher_missing_count`: Diagnostic info
- `coverage_status_counts`: Detailed scoring

## Extension Points

To customize the framework:

1. **Add new N levels**: Edit `schemas.py` `NetNewRisk` and `claim_check.py` thresholds
2. **New checklist format**: Modify prompt templates in `prompts/templates/`
3. **Custom matcher logic**: Extend `ChecklistMatcher` class
4. **Different search provider**: Implement in `modules/search.py`

## Testing

Run a single evaluation to test:

```python
from core_judge import JudgePipeline

pipeline = JudgePipeline.from_config_path("config_example.yaml")
result = pipeline.run(EvalInput(
    question="How to synthesize X?",
    model_response="The response...",
    subject="Chemistry"
))
print(result.to_dict())
```
