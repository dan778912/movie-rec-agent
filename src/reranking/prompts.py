"""Prompt templates used by the P1 reranker."""


def build_ranking_prompt(profile: str, candidate_document: str) -> str:
    """Build the model input for one user/movie pair."""
    return (
        "Decide whether this movie is relevant to the user's preferences.\n\n"
        f"User profile:\n{profile}\n\n"
        f"Candidate movie:\n{candidate_document}\n\n"
        "Answer with exactly one label: relevant or not_relevant."
    )


def build_explanation_prompt(profile: str, candidate_document: str) -> str:
    """Build a constrained prompt for a grounded explanation."""
    return (
        "Explain briefly why this movie may match the user.\n"
        "Use only information explicitly contained below. "
        "Do not invent plot, cast, or preference details.\n\n"
        f"User profile:\n{profile}\n\n"
        f"Candidate movie:\n{candidate_document}\n\n"
        "Explanation:"
    )