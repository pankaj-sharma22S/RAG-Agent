"""Pydantic schemas and structured output models for GitHub repository analysis."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """An evidence-based technical finding regarding potential issues or architecture nuances."""

    severity: Literal["Critical", "High", "Medium", "Low", "Informational"] = Field(
        description="Severity level of the finding."
    )
    category: Literal[
        "Logic Break",
        "Runtime Error",
        "Documentation Mismatch",
        "Dependency/Import Issue",
        "Performance",
        "Security",
        "State Management",
        "Regression",
        "Architecture/Best Practice",
    ] = Field(description="Category of the finding.")
    title: str = Field(description="Concise summary title of the finding.")
    description: str = Field(description="Detailed explanation of what was identified and how it affects the application.")
    evidence: str = Field(description="Concrete code, commit, or doc evidence backing up the finding.")
    file: str | None = Field(default=None, description="Relevant file path if applicable.")
    line_range: str | None = Field(default=None, description="Line number or range (e.g. '12-35') if applicable.")
    commit: str | None = Field(default=None, description="Relevant commit SHA if applicable.")
    branch: str | None = Field(default=None, description="Relevant branch if applicable.")
    confidence: Literal["Confirmed", "Likely", "Possible", "Needs Verification"] = Field(
        description="Confidence level based on available evidence."
    )
    verification_status: Literal[
        "Static Analysis Finding",
        "Runtime/Test-Confirmed",
        "Documentation Verified",
        "Needs Verification",
    ] = Field(description="Verification type (Static analysis vs runtime confirmed).")
    recommendation: str = Field(description="Actionable remediation or next step recommendation.")


class CodeStructureAnalysis(BaseModel):
    """Intermediate model for code structure, entry points, and dependencies."""

    languages: list[str] = Field(default_factory=list, description="Identified programming languages.")
    frameworks: list[str] = Field(default_factory=list, description="Identified frameworks or major libraries.")
    entry_points: list[str] = Field(default_factory=list, description="Identified application entry point files.")
    key_modules: list[str] = Field(default_factory=list, description="Primary module/directory paths and their roles.")
    dependencies: list[str] = Field(default_factory=list, description="Key dependencies and build configurations.")
    test_frameworks: list[str] = Field(default_factory=list, description="Test directories or frameworks found.")


class DocumentationComparison(BaseModel):
    """Comparison between documentation claims and codebase realities."""

    claimed_features: list[str] = Field(default_factory=list, description="Features claimed in README or docs.")
    missing_or_outdated: list[str] = Field(
        default_factory=list, description="Documented APIs, env vars, or commands that do not exist or are outdated."
    )
    undocumented_features: list[str] = Field(
        default_factory=list, description="Key capabilities present in code but absent from documentation."
    )
    readme_accuracy_score: float = Field(
        ge=0.0, le=1.0, default=1.0, description="Estimated accuracy score of documentation (0.0 to 1.0)."
    )


class StructuredAnalysisReport(BaseModel):
    """Complete structured technical report produced by the GitHub Analyzer Agent."""

    repo_name: str = Field(description="Owner and repository name (e.g. 'owner/repo').")
    overview: str = Field(description="Concise executive overview of the repository's purpose and functionality.")
    architecture: str = Field(description="Comprehensive architectural overview and component interaction description.")
    tech_stack: list[str] = Field(default_factory=list, description="Languages, frameworks, and key libraries.")
    entry_points: list[str] = Field(default_factory=list, description="Main application entry points and launch flows.")
    important_files: list[str] = Field(default_factory=list, description="List of the most critical files in the codebase.")
    execution_flow: str = Field(description="Step-by-step description of the runtime execution flow.")
    data_flow: str = Field(description="Description of data ingestion, transformation, storage, and egress.")
    dependencies_summary: str = Field(description="Summary of package dependencies and build configuration.")
    branch_analysis: str = Field(description="Analysis of branches, diverged features, or unmerged fixes.")
    commit_analysis: str = Field(description="Analysis of recent commits, notable changes, and suspected regressions.")
    readme_vs_code: str = Field(description="Analysis of documentation accuracy vs implementation realities.")
    findings: list[Finding] = Field(default_factory=list, description="List of evidence-backed technical findings.")
    recommendations: list[str] = Field(default_factory=list, description="Key strategic and tactical recommendations.")
