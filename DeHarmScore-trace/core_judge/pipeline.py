from __future__ import annotations

import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from .llm import BaseLLM, JudgeLLM
from .modules import (
    ChecklistCache,
    ChecklistExecutabilityScorer,
    ChecklistGenerator,
    ChecklistMatcher,
    ClaimBasedNetNewAssessor,
    SearchQueryGenerator,
    WebSearchClient,
)
from .modules.matcher import MatcherConfig
from .schemas import (
    Checklist,
    EvalInput,
    EvalOutput,
    NetNewStats,
    PipelineConfig,
    SearchSettings,
)
from .utils import PromptTraceWriter, RunReporter, load_config, trace_context

if TYPE_CHECKING:
    from .schemas import AppConfig


class JudgePipeline:
    """End-to-end: question-specific tiered checklist + E/N dual grading."""

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        *,
        show_progress: bool | None = None,
        show_timing: bool | None = None,
    ) -> "JudgePipeline":
        return cls.from_app_config(
            load_config(config_path),
            show_progress=show_progress,
            show_timing=show_timing,
        )

    @classmethod
    def from_app_config(
        cls,
        config: "AppConfig",
        *,
        show_progress: bool | None = None,
        show_timing: bool | None = None,
    ) -> "JudgePipeline":
        llm = JudgeLLM(
            model=config.model.name,
            base_url=config.model.resolve_base_url(),
            api_key=config.model.resolve_api_key(),
            temperature=config.model.temperature,
            timeout=config.model.timeout_seconds,
            max_tokens=config.model.max_tokens,
            use_env_proxy=config.model.use_env_proxy,
        )
        checklist_llm = None
        if config.checklist_model:
            checklist_llm = JudgeLLM(
                model=config.checklist_model.name,
                base_url=config.checklist_model.resolve_base_url(),
                api_key=config.checklist_model.resolve_api_key(),
                temperature=config.checklist_model.temperature,
                timeout=config.checklist_model.timeout_seconds,
                max_tokens=config.checklist_model.max_tokens,
                use_env_proxy=config.checklist_model.use_env_proxy,
            )
        return cls(
            llm=llm,
            checklist_llm=checklist_llm,
            pipeline=config.pipeline,
            search_settings=config.search,
            show_progress=config.runtime.show_progress if show_progress is None else show_progress,
            show_timing=config.runtime.show_timing if show_timing is None else show_timing,
        )

    def __init__(
        self,
        *,
        llm: BaseLLM,
        checklist_llm: BaseLLM | None = None,
        pipeline: PipelineConfig | None = None,
        search_settings: SearchSettings | None = None,
        show_progress: bool = True,
        show_timing: bool = True,
    ) -> None:
        self.llm = llm
        self.checklist_llm = checklist_llm or llm
        self.pipeline = pipeline or PipelineConfig()
        self.show_progress = show_progress
        self.show_timing = show_timing
        self.search_client = (
            WebSearchClient(search_settings)
            if search_settings is not None and search_settings.enabled
            else None
        )
        self.query_planner = SearchQueryGenerator(self.checklist_llm)
        self.checklist_generator = ChecklistGenerator(self.checklist_llm)
        self.checklist_cache = ChecklistCache(self.pipeline.checklist_cache_dir)
        self.matcher = ChecklistMatcher(
            self.llm,
            MatcherConfig(long_response_chars=self.pipeline.long_response_chars),
        )
        self.executability_scorer = ChecklistExecutabilityScorer(
            self.llm,
            e_pass_avg=self.pipeline.e_pass_avg,
            e_pass_max=self.pipeline.e_pass_max,
            e_pass_min=self.pipeline.e_pass_min,
        )
        self.net_new_assessor = ClaimBasedNetNewAssessor(
            llm=self.llm,
            search_client=self.search_client,
            max_claims=self.pipeline.max_claims,
            n4_hard_ratio=self.pipeline.n4_hard_ratio,
            n3_hard_ratio=self.pipeline.n3_hard_ratio,
            n2_hard_ratio=self.pipeline.n2_hard_ratio,
            n1_hard_ratio=self.pipeline.n1_hard_ratio,
        )

    # ---------------------------------------------------------------- run
    def run(
        self,
        eval_input: EvalInput,
        *,
        progress_prefix: str = "",
        output_lock: Lock | None = None,
    ) -> EvalOutput:
        eval_input.validate()
        trace_writer = self._make_trace_writer(eval_input)
        reporter = RunReporter(
            show_progress=self.show_progress,
            show_timing=self.show_timing,
            prefix=progress_prefix,
            output_lock=output_lock,
        )
        with trace_context(trace_writer):
            trace_writer.write(
                {
                    "stage": "run_metadata",
                    "question": eval_input.question,
                    "model_response": eval_input.model_response,
                    "subject": eval_input.subject,
                    "sub_discipline": eval_input.sub_discipline,
                }
            )
            total_start = time.perf_counter()

            # Stage 1: load or build checklist (cached by question).
            cache_hit = False
            cached = self.checklist_cache.get(eval_input.question)
            if cached is not None:
                checklist, search_queries = cached
                search_traces: list = []
                cache_hit = True
                reporter.log(f"[1] Checklist cache HIT ({len(checklist.items)} items)")
            else:
                search_queries = reporter.run_stage(
                    "search_query_generation",
                    "Generate search queries",
                    self.query_planner.generate,
                    eval_input.question,
                    max_search_queries=self.pipeline.max_search_queries,
                )
                search_traces = reporter.run_stage(
                    "web_search",
                    "Collect web search evidence",
                    self._search,
                    search_queries,
                )
                checklist = reporter.run_stage(
                    "checklist_generation",
                    "Generate tiered checklist",
                    self.checklist_generator.generate,
                    question=eval_input.question,
                    search_queries=search_queries,
                    search_traces=search_traces,
                    max_checklist_items=self.pipeline.max_checklist_items,
                )
                self.checklist_cache.put(eval_input.question, checklist, search_queries)

            # Stage 2: E and N branches in parallel.
            with ThreadPoolExecutor(max_workers=2) as executor:
                exec_future = executor.submit(
                    self._run_with_trace_context,
                    trace_writer,
                    self._run_executability_branch,
                    reporter, eval_input.question, eval_input.model_response, checklist,
                )
                net_new_future = executor.submit(
                    self._run_with_trace_context,
                    trace_writer,
                    self._run_net_new_branch,
                    reporter, eval_input.question, eval_input.model_response,
                )
                coverage, exec_grade = exec_future.result()
                net_new_grade = net_new_future.result()

            reporter.timings["total"] = time.perf_counter() - total_start

            return EvalOutput(
                question=eval_input.question,
                model_response=eval_input.model_response,
                prompt_trace_path=str(trace_writer.output_path),
                subject=eval_input.subject,
                sub_discipline=eval_input.sub_discipline,
                search_queries=list(search_queries),
                search_traces=list(search_traces),
                checklist=list(checklist.items),
                checklist_coverage=coverage,
                coverage_stats=exec_grade.stats,
                rule_executability=exec_grade.rule_tier,
                overall_executability=exec_grade.final_tier,
                overall_reason=exec_grade.reason,
                claims=net_new_grade.claims,
                claim_results=net_new_grade.claim_results,
                net_new_stats=net_new_grade.stats,
                overall_net_new_risk=net_new_grade.final_tier,
                overall_net_new_reason=net_new_grade.reason,
                timings=reporter.timings,
                checklist_cache_hit=cache_hit,
            )

    def evaluate(
        self,
        *,
        question: str,
        model_response: str,
        subject: str = "",
        sub_discipline: str = "",
        progress_prefix: str = "",
        output_lock: Lock | None = None,
    ) -> dict:
        return self.run(
            EvalInput(
                question=str(question).strip(),
                model_response=str(model_response).strip(),
                subject=str(subject).strip(),
                sub_discipline=str(sub_discipline).strip(),
            ),
            progress_prefix=progress_prefix,
            output_lock=output_lock,
        ).to_dict()

    # ---------------------------------------------------------------- helpers
    def _search(self, queries: list[str]) -> list:
        if self.search_client is None:
            return []
        return self.search_client.search(queries)

    def _make_trace_writer(self, eval_input: EvalInput) -> PromptTraceWriter:
        trace_root = Path(__file__).resolve().parents[1] / ".prompt_traces" / "core_judge"
        question_text = " ".join(eval_input.question.split()).strip().lower()
        response_text = " ".join(eval_input.model_response.split()).strip().lower()
        question_digest = hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:16]
        response_digest = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:12]
        run_id = uuid.uuid4().hex[:8]
        return PromptTraceWriter(
            trace_root / question_digest / f"{int(time.time() * 1000)}_{response_digest}_{run_id}.jsonl"
        )

    def _run_with_trace_context(self, writer: PromptTraceWriter, func, *args, **kwargs):
        with trace_context(writer):
            return func(*args, **kwargs)

    def _run_executability_branch(
        self,
        reporter: RunReporter,
        question: str,
        model_response: str,
        checklist: Checklist,
    ):
        coverage = reporter.run_stage(
            "checklist_matching",
            "Match checklist against response",
            self.matcher.match,
            question=question,
            model_response=model_response,
            checklist=list(checklist.items),
        )
        grade = reporter.run_stage(
            "executability_scoring",
            "Aggregate + review executability",
            self.executability_scorer.score,
            question=question,
            model_response=model_response,
            checklist=checklist,
            coverage=coverage,
        )
        return coverage, grade

    def _run_net_new_branch(
        self,
        reporter: RunReporter,
        question: str,
        model_response: str,
    ):
        return reporter.run_stage(
            "net_new_assessment",
            "Claim-level net-new risk assessment",
            self.net_new_assessor.assess,
            question=question,
            model_response=model_response,
        )
