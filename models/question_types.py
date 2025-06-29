from enum import Enum, auto

class QuestionType(Enum):
    FACTUAL_RECALL = auto()
    INFERENCE = auto()
    MULTI_HOP_REASONING = auto()
    APPLICATION = auto()
    COMPARATIVE_ANALYSIS = auto()
    CAUSE_EFFECT = auto()
    SUMMARIZATION = auto()
    HYPOTHETICAL = auto()
    CRITICAL_ANALYSIS = auto()
    TECHNICAL_EXPLANATION = auto()
    PROCESS_WORKFLOW = auto()
    TRUE_FALSE_FILL_BLANK = auto()
    CONTEXTUAL_AMBIGUITY = auto()
    FACT_BASED = auto()
    SECTION_SUMMARY = auto()

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

QUESTION_TYPE_DIFFICULTY = {
    QuestionType.FACTUAL_RECALL: DifficultyLevel.EASY,
    QuestionType.INFERENCE: DifficultyLevel.MEDIUM,
    QuestionType.MULTI_HOP_REASONING: DifficultyLevel.HARD,
    QuestionType.APPLICATION: DifficultyLevel.MEDIUM,
    QuestionType.COMPARATIVE_ANALYSIS: DifficultyLevel.MEDIUM,
    QuestionType.CAUSE_EFFECT: DifficultyLevel.MEDIUM,
    QuestionType.SUMMARIZATION: DifficultyLevel.EASY,
    QuestionType.HYPOTHETICAL: DifficultyLevel.HARD,
    QuestionType.CRITICAL_ANALYSIS: DifficultyLevel.HARD,
    QuestionType.TECHNICAL_EXPLANATION: DifficultyLevel.MEDIUM,
    QuestionType.PROCESS_WORKFLOW: DifficultyLevel.EASY,
    QuestionType.TRUE_FALSE_FILL_BLANK: DifficultyLevel.EASY,
    QuestionType.CONTEXTUAL_AMBIGUITY: DifficultyLevel.HARD,
    QuestionType.FACT_BASED: DifficultyLevel.EASY,
    QuestionType.SECTION_SUMMARY: DifficultyLevel.MEDIUM,
}