"""User-owned context and prompt construction TODOs."""

from api_gateway.schemas import RetrievalMatch


def build_context(matches: list[RetrievalMatch]) -> str:
    """TODO: Build context from retrieved chunks manually."""
    sections = []
    for index, match in enumerate(matches, start=1):
        section = f"[Source {index} | document_id={match.document_id}]\n{match.text}"
        sections.append(section)
    return "\n\n".join(sections)

def build_prompt(query: str, context: str) -> str:
    """Build a grounded generation prompt."""
    prompt = f"""Answer the question using only the provided context.
If the context does not contain enough information to answer the question,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
