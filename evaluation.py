import json
import re
from dataclasses import dataclass
from clients import WolverineClient

EVALUATION_SYSTEM_PROMPT = (
    "You are a literary critic. Always respond with JSON containing the keys "
    '"score" (a number from 1.0 to 10.0, can include one decimal place like 7.5) and "explanation" (a short justification).'
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

    def evaluate_all_categories(self, story: str) -> dict[str, EvaluationResult]:
        """Evaluate a story across all categories and return results, then evaluate creativity based on the results."""
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
        
        # Evaluate creativity based on all category results
        category_summary = "\n".join([
            f"- {cat}: {res.score}/10 - {res.explanation}"
            for cat, res in results.items()
        ])
        creativity_prompt = (
            f"Based on the following comprehensive evaluation results across all categories, "
            f"what creativity score (1.0-10.0, can include one decimal place) would you give this story? "
            f"Consider how the story demonstrates originality, innovation, unique perspectives, and imaginative elements.\n\n"
            f"Evaluation Results:\n{category_summary}\n\n"
            f"Original Story:\n{story}"
        )
        creativity_response = self._client.chat(
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            user_prompt=creativity_prompt,
        )
        creativity_score, creativity_explanation = parse_response(creativity_response)
        if creativity_score is None:
            creativity_score = 0.0
        results["Creativity"] = EvaluationResult(
            category="Creativity",
            score=creativity_score,
            explanation=creativity_explanation,
        )
        
        return results

    def evaluate_creativity(self, story: str) -> EvaluationResult:
        """Evaluate a story's creativity directly without breaking it into categories."""
        user_prompt = f"Evaluate the creativity of the following story. Consider originality, innovation, unique perspectives, and imaginative elements.\n\nStory:\n{story}"
        response = self._client.chat(
            system_prompt=EVALUATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        score, explanation = parse_response(response)
        if score is None:
            score = 0.0  # Default score if parsing fails
        return EvaluationResult(
            category="Creativity",
            score=score,
            explanation=explanation,
        )

    def analyze_creativity_difference(
        self, 
        story: str,
        standalone_creativity: EvaluationResult,
        all_categories_results: dict[str, EvaluationResult]
    ) -> dict:
        """Analyze which categories influenced the difference in creativity scores."""
        standalone_score = standalone_creativity.score
        contextual_score = all_categories_results.get("Creativity", standalone_creativity).score
        
        # If scores are the same (within 0.1 tolerance), return early
        if abs(standalone_score - contextual_score) < 0.1:
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": 0.0,
                "analysis": "The creativity scores are essentially the same. No significant difference to analyze.",
                "influential_categories": []
            }
        
        # Build category summary (excluding Creativity itself)
        category_summary = "\n".join([
            f"- {cat}: {res.score}/10 - {res.explanation}"
            for cat, res in all_categories_results.items()
            if cat != "Creativity"
        ])
        
        # List all available categories for reference
        available_categories = "\n".join([f"- {cat}" for cat in STORY_EVALUATION_CATEGORIES])
        
        analysis_prompt = (
            f"Two different creativity scores were given for the same story:\n"
            f"- Standalone creativity score (evaluated without category context): {standalone_score}/10\n"
            f"- Contextual creativity score (evaluated after seeing all category results): {contextual_score}/10\n"
            f"- Difference: {abs(standalone_score - contextual_score):.1f} points\n\n"
            f"All category evaluation results:\n{category_summary}\n\n"
            f"Original Story:\n{story}\n\n"
            f"Available categories :\n{available_categories}\n\n"
            f"Please analyze which specific categories influenced the change in creativity score. "
            f"You MUST only select from the 14 categories listed above. Do not create new category names. "
            f"Respond with JSON containing: "
            f'"influential_categories" (list of category names from the available categories that most influenced the difference), '
            f'"analysis" (explanation of how these categories affected the creativity score change), '
            f'and "impact" (positive/negative/neutral for each influential category).'
        )
        
        analysis_response = self._client.chat(
            system_prompt="You are a literary analysis expert. Always respond with valid JSON.",
            user_prompt=analysis_prompt,
        )
        
        # Parse the analysis response
        try:
            analysis_data = json.loads(analysis_response)
            influential_categories = analysis_data.get("influential_categories", [])
            
            # Filter to only include valid categories from STORY_EVALUATION_CATEGORIES
            valid_categories = [
                cat for cat in influential_categories 
                if cat in STORY_EVALUATION_CATEGORIES
            ]
            
            # Filter impact dict to only include valid categories
            impact = analysis_data.get("impact", {})
            valid_impact = {
                cat: impact[cat] 
                for cat in valid_categories 
                if cat in impact
            }
            
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": round(abs(standalone_score - contextual_score), 1),
                "analysis": analysis_data.get("analysis", ""),
                "influential_categories": valid_categories,
                "impact": valid_impact
            }
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "standalone_creativity_score": standalone_score,
                "contextual_creativity_score": contextual_score,
                "difference": round(abs(standalone_score - contextual_score), 1),
                "analysis": analysis_response,
                "influential_categories": [],
                "impact": {}
            }

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

    # Fallback: extract numeric score from text (supports integers and one decimal place)
    match = re.search(r"(?<!\d)(10\.\d|10|[1-9]\.\d|[1-9])(?!\d)", response)
    score = float(match.group(1)) if match else None
    explanation = response if not match else response.replace(match.group(0), "", 1).strip()
    return score, explanation