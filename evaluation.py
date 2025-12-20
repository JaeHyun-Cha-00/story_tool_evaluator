import json
import re
from dataclasses import dataclass
from clients import WolverineClient

EVALUATION_SYSTEM_PROMPT = (
    "You are a literary critic. Always respond with JSON containing the keys "
    '"score" (an integer from 1 to 10) and "explanation" (a short justification).'
)

STORY_EVALUATION_CATEGORIES = [
    # Language Quality
    "Grammar, spelling, and punctuation quality",
    "Sentence pattern variety",
    "Avoidance of clichés and overused phrases",
    # Clarity & Logic
    "Clarity and understandability",
    "Logical connection between events and ideas",
    "Internal consistency within the story's context",
    # Narrative Construction
    "Scene construction and purpose",
    "Avoidance of predictable narrative tropes",
    "Ability to hold reader interest",
    # Character Realism
    "Character consistency",
    "Character motivation and actions making sense",
    "Natural dialogue",
    "Character depth and dimensionality",
    "Realistic character interactions",
]

def build_user_prompt(story: str, category: str) -> str:
    """Build the user prompt sent to the model."""
    return f"Evaluate the following story focusing strictly on the category: {category}.\n\nStory:\n{story}"


######## Data Classes ########
@dataclass
class EvaluationResult:
    """Evaluation Response for a single category."""

    category: str
    score: float
    explanation: str

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "score": self.score,
            "explanation": self.explanation,
        }
    
##### Story Evaluator (Main) ########
class StoryEvaluator:
    """Evaluate stories for multiple literary categories."""

    def __init__(self, client: WolverineClient):
        """Initialize the evaluator with a Wolverine client."""
        self._client = client

    def evaluate(self, story: str) -> dict[str, EvaluationResult]:
        """Evaluate a story across all categories and return results."""
        results = {}
        for category in STORY_EVALUATION_CATEGORIES:
            user_prompt = build_user_prompt(story, category)
            response = self._client.chat(
                system_prompt=EVALUATION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
            score, explanation = parse_response(response)
            if score is None:
                score = 0.0  # Default score if parsing fails
            results[category] = EvaluationResult(
                category=category,
                score=score,
                explanation=explanation,
            )
        return results

###### Response Parsing ########
def parse_response(response: str) -> tuple[float | None, str]:
    """Parse the model's response into a numeric score and explanation."""
    response = response.strip()
    if not response:
        return None, ""

    # Try structured JSON response
    try:
        payload = json.loads(response)
        score = float(payload.get("score")) if "score" in payload else None
        explanation = str(payload.get("explanation", "")).strip()
        return score, explanation
    except json.JSONDecodeError:
        pass

    # Fallback: extract numeric score from text
    match = re.search(r"(?<!\d)(10|[1-9])(?!\d)", response)
    score = float(match.group(1)) if match else None
    explanation = response if not match else response.replace(match.group(0), "", 1).strip()
    return score, explanation