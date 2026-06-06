from pydantic import BaseModel
from typing import Literal, Optional


class InputTopic(BaseModel):
    topic: str
    domain: str
    target_venue: Optional[str] = None
    date_range: tuple[int, int] = (2021, 2025)
    max_papers: int = 25


class Paper(BaseModel):
    title: str
    authors: list[str]
    year: int
    venue: str
    url: str
    abstract_summary: str
    key_contributions: list[str]
    stated_limitations: list[str]
    semantic_scholar_id: Optional[str] = None


class PaperList(BaseModel):
    papers: list[Paper]
    search_queries_used: list[str]
    timestamp: str


class Gap(BaseModel):
    gap_id: str
    description: str
    supporting_paper_titles: list[str]
    impact_score: float
    feasibility_score: float


class GapAnalysis(BaseModel):
    gaps: list[Gap]
    top_gap_id: str
    rationale: str


class IdeaCandidate(BaseModel):
    idea_id: str
    hypothesis: str
    novelty_explanation: str
    minimum_viable_experiment: str
    compute_cost: Literal["low", "medium", "high"]
    time_estimate: str
    novelty_score: float
    feasibility_score: float
    impact_score: float


class IdeaCandidates(BaseModel):
    ideas: list[IdeaCandidate]
    recommended_idea_id: str


class TitleOptions(BaseModel):
    descriptive: str
    punchy: str
    question_form: str
    recommended: Literal["descriptive", "punchy", "question_form"]
    rationale: str


class Baseline(BaseModel):
    name: str
    citation: str
    reason_for_inclusion: str


class Dataset(BaseModel):
    name: str
    url: str
    size: str
    license: str


class Metric(BaseModel):
    name: str
    formula: str
    higher_is_better: bool


class ExperimentPlan(BaseModel):
    hypothesis_null: str
    hypothesis_alternative: str
    independent_variables: list[str]
    dependent_variables: list[str]
    baselines: list[Baseline]
    datasets: list[Dataset]
    metrics: list[Metric]
    statistical_tests: list[str]
    ablation_design: str
    compute_requirements: str
    estimated_runtime_hours: float


class CodeTask(BaseModel):
    script_path: str
    requirements_path: str
    run_command: str
    expected_outputs: list[str]


class CodeResult(BaseModel):
    stdout: str
    stderr: str
    output_files: list[str]
    execution_time_seconds: float
    success: bool
    retry_count: int


class MetricResult(BaseModel):
    metric_name: str
    mean: float
    std: float
    n_runs: int


class ResultSummary(BaseModel):
    metrics: list[MetricResult]
    hypothesis_verdict: Literal["supported", "partially_supported", "rejected"]
    key_findings: list[str]
    anomalies: list[str]
    suggested_ablation: Optional[str] = None


class PaperDraft(BaseModel):
    abstract: str
    introduction: str
    related_work: str
    methodology: str
    experiments: str
    results: str
    discussion: str
    conclusion: str
    limitations: str
    references: list[str]


class ReviewIssue(BaseModel):
    severity: Literal["critical", "major", "minor"]
    section: str
    description: str


class ReviewerFeedback(BaseModel):
    issues: list[ReviewIssue]
    overall_verdict: Literal["accept", "revise", "reject"]
    required_changes: list[str]
    citation_audit_passed: bool
    number_audit_passed: bool
    hallucination_flags: list[str]


class FinalDecision(BaseModel):
    human_approved: bool
    approver_notes: str
    export_formats: list[Literal["pdf", "latex", "markdown"]]
    timestamp: str
