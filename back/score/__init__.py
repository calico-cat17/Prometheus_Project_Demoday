"""추리 게임 채점 모듈."""

from .qwen_evaluator import (
    CASE_CONFIG,
    EvaluationResult,
    OpenAIAPI,
    evaluate_deduction,
)

__all__ = [
    "CASE_CONFIG",
    "EvaluationResult",
    "OpenAIAPI",
    "evaluate_deduction",
]
