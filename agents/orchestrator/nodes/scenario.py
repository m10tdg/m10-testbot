import openai
from clients import embed, qdrant
from rag import retrieve_context

SYSTEM_PROMPT = """You are a senior QA automation engineer. Given a target URL and a
test intent, write the BODY of a Python function using Playwright's SYNC API.

Rules:
- Do NOT write the function signature, imports, or browser setup - ONLY the body.
- The body receives a variable `page` (an already-open Playwright Page) and `console_errors`.
- Use resilient locators: page.get_by_role(...), page.get_by_text(...).
- NEVER call non-existent attributes like .selector on Playwright Locator objects.
- End with at least one assert statement tied to the test intent.
- Return ONLY Python code. No markdown fences, no prose, no explanation.
"""


def scenario_agent_node(state: dict) -> dict:
    print(f"[scenario-agent] generating script for run {state['run_id']}")

    query_vector = embed(state["prompt"])
    context_chunks = retrieve_context(
        qdrant,
        tenant_id=state["tenant_id"],
        project_id=state["project_id"],
        query_vector=query_vector,
        types=["requirement", "user_story", "acceptance_criteria"],
        top_k=8,
    )

    context_block = "\n".join(f"- {c}" for c in context_chunks) or "(no project documents indexed yet)"

    user_prompt = f"""URL: {state['url']}
Test intent: {state['prompt']}

Relevant project context:
{context_block}
"""
    print(f"[scenario-agent] sending prompt to LLM:\n{user_prompt}")
    completion = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    script_body = completion.choices[0].message.content.strip()
    # Strip accidental markdown fences - LLMs do this even when told not to.
    script_body = script_body.removeprefix("```python").removeprefix("```").removesuffix("```").strip()

    print(f"[scenario-agent] generated script:\n{script_body}")

    print(f"[scenario-agent] generated {len(script_body.splitlines())} line(s) of test code")

    return {
        **state,
        "requirements_context": context_chunks,
        "playwright_script": script_body,
        "status": "scenario_generated",
    }
