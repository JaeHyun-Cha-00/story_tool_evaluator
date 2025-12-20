from fastmcp import FastMCP
from evaluation import StoryEvaluator, STORY_EVALUATION_CATEGORIES
from clients import WolverineClient

mcp = FastMCP(name="Story Evaluator")
client = WolverineClient()
evaluator = StoryEvaluator(client)

@mcp.tool()
def list_categories() -> list[str]:
    """Return all supported evaluation categories."""
    print("[INFO] Tool called: list_categories")
    return STORY_EVALUATION_CATEGORIES

@mcp.tool()
def evaluate_all_categories(story: str) -> dict[str, dict]:
    """Evaluate a story across all evaluation categories."""
    print("[INFO] Tool called: evaluate_all_categories")
    results = evaluator.evaluate_all_categories(story)
    return {cat: res.to_dict() for cat, res in results.items()}

@mcp.tool()
def evaluate_creativity(story: str) -> dict:
    """Evaluate a story's creativity directly without breaking it into categories."""
    print("[INFO] Tool called: evaluate_creativity")
    result = evaluator.evaluate_creativity(story)
    return result.to_dict()

@mcp.tool()
def compare_creativity_scores(story: str) -> dict:
    """Run both evaluate_creativity and evaluate_all_categories, then analyze which categories influenced the creativity score difference if they differ."""
    print("[INFO] Tool called: compare_creativity_scores")
    
    # Run both evaluations
    standalone_result = evaluator.evaluate_creativity(story)
    all_categories_results = evaluator.evaluate_all_categories(story)
    
    # Analyze the difference
    analysis = evaluator.analyze_creativity_difference(
        story, 
        standalone_result, 
        all_categories_results
    )
    
    # Include both creativity results in the response
    return {
        "standalone_creativity": standalone_result.to_dict(),
        "contextual_creativity": all_categories_results.get("Creativity", standalone_result).to_dict(),
        "all_categories": {cat: res.to_dict() for cat, res in all_categories_results.items() if cat != "Creativity"},
        "difference_analysis": analysis
    }


if __name__ == "__main__":
    mcp.run()