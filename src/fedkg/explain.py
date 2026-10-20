"""LLM explanation layer with and without KG grounding. Phase 4 - to be implemented."""


def build_prompt(prediction, facts, grounded=True):
    base = f"The model predicts remaining useful life = {prediction:.0f} cycles.\n"
    if grounded:
        base += "Use ONLY these verified facts:\n" + "\n".join(f"- {f}" for f in facts) + "\n"
        base += "Explain the likely cause and recommend an action. Do not add facts not listed.\n"
    else:
        base += "Explain the likely cause and recommend a maintenance action.\n"
    return base
