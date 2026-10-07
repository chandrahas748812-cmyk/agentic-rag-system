"""Human-in-the-loop escalation policy."""


def maybe_escalate(state, threshold: float = 0.75) -> bool:
    """Decide whether the draft answer needs human review.

    Escalate when:
    - model confidence is below threshold, or
    - no evidence survived verification, or
    - the answer explicitly says it lacks information.
    """
    if state.confidence < threshold:
        state.needs_human_review = True
        return True
    if not state.kept_chunks:
        state.needs_human_review = True
        return True
    if "don't have enough information" in state.draft_answer.lower():
        state.needs_human_review = True
        return True
    state.needs_human_review = False
    return False
