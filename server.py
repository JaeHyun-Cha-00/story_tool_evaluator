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
def evaluate(story: str) -> dict[str, dict]:
    print("[INFO] Tool called: evaluate")
    results = evaluator.evaluate(story)
    return {cat: res.to_dict() for cat, res in results.items()}


if __name__ == "__main__":
    mcp.run()