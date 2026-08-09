# agents/scenario_agent/node.py
from openai import OpenAI
from libs.shared_py.rag import retrieve_context

client = OpenAI()

SCENARIO_SYSTEM_PROMPT = """You are a senior QA automation engineer. Given a natural-language
test intent and relevant project requirements, write a single Playwright test file (TypeScript)
that:
- Navigates to the given URL
- Exercises the flow described in the prompt
- Uses resilient selectors (role/text based, not brittle CSS paths)
- Includes basic assertions tied to the acceptance criteria provided
Return ONLY the code, no prose."""

def scenario_agent_node(state: dict, qdrant_client) -> dict:
    query_embedding = embed(state["prompt"])  # reuse embed() from embedding-service lib
    context_chunks = retrieve_context(
        qdrant_client,
        tenant_id=state["tenant_id"],
        project_id=state["project_id"],
        query_vector=query_embedding,
        types=["requirement", "user_story", "acceptance_criteria"],
        top_k=8,
    )

    user_prompt = f"""URL: {state['url']}
Test intent: {state['prompt']}

Relevant project context:
{chr(10).join(f"- {c}" for c in context_chunks)}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SCENARIO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,   # low — this is code generation, not creative writing
    )

    script = completion.choices[0].message.content

    return {
        **state,
        "requirements_context": context_chunks,
        "playwright_script": script,
        "scenario_count": script.count("test("),  # rough count of generated test() blocks
        "status": "scenario_generated",
    }