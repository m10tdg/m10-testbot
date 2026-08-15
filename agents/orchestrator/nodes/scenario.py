"""
Enhanced Scenario Agent

Improvements:
1. Uses actual page structure from crawler
2. Retrieves RAG context for the project
3. Better system prompt with resilience rules
4. Handles self-healing (previous errors)
5. Generates more robust Playwright code
"""

import openai
from clients import embed, qdrant, OPENAI_API_KEY
from rag import retrieve_context

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are a senior QA automation engineer writing Playwright tests.

CRITICAL RULES:
1. You write ONLY the function body (no imports, no setup, no signature).
2. The body has access to:
   - `page`: An already-open Playwright Page object
   - `console_errors`: A list you append to if you want to track errors

3. IMPORTANT: Use SYNCHRONOUS Playwright API (NOT async/await):
   - ✓ page.goto(url)
   - ✓ page.click(selector)
   - ✓ page.fill(selector, text)
   - ✗ await page.goto() - WRONG!
   - ✗ async def - WRONG!

4. RESILIENT LOCATORS (in order of preference):
   a) Specific attributes: page.get_by_role("button", name="Login")
   b) Test IDs: page.locator('[data-testid="submit-btn"]')
   c) Name or label: page.get_by_label("Email")
   d) Placeholder: page.get_by_placeholder("Enter email")
   e) Text content: page.get_by_text("Click here")
   f) Last resort: CSS selectors (but only if specific)

5. ALWAYS handle visibility and waits:
   - Use `.wait_for()` before interacting with elements
   - Use explicit timeouts: page.wait_for_selector(..., timeout=5000)
   - Check visibility: locator.is_visible()

6. FORM FILLING:
   - Always wait for the input to be visible first
   - Use .fill() for text inputs, not .type() (it's faster)
   - Tab out after filling to trigger validation

7. BUTTON CLICKS:
   - Always wait for the button to be enabled
   - Use page.click() or locator.click()
   - Register dialog handlers BEFORE clicking if expecting popups

8. ASSERTIONS:
   - Always assert something concrete at the end
   - Check URL changes, text visibility, element counts
   - Never assert on absence (use try/except instead)

9. ERROR RECOVERY (if {error_log} is provided):
   - Analyze exactly what failed (Timeout? Not visible? Wrong selector?)
   - Rewrite the code to fix that specific issue
   - Try alternative selectors if the original timed out
   - Add extra waits or retries if needed

10. RETURN FORMAT:
   - ONLY Python code
   - NO markdown fences (no ```python ``` wrappers)
   - NO explanations or prose
   - Code must be directly executable
   - NO async/await - this must run synchronously!

EXAMPLE OF CORRECT CODE:
# Wait for and click login button
page.wait_for_selector('button:has-text("Login")')
page.click('button:has-text("Login")')
page.wait_for_url("**/dashboard")
assert "dashboard" in page.url
"""


def scenario_agent_node(state: dict) -> dict:
    """
    Generate a Playwright test script for the given URL and prompt.
    
    Flow:
    1. Retrieve project context from Qdrant (RAG)
    2. Get the actual page structure from crawler
    3. Use LLM to generate script
    4. Return the script in RunState
    """
    run_id = state["run_id"]
    print(f"\n[scenario-agent] generating script for run {run_id}")
    
    # Step 1: Retrieve project context from RAG
    print(f"[scenario-agent] retrieving project context from Qdrant...")
    try:
        query_vector = embed(state["prompt"])
        context_chunks = retrieve_context(
            qdrant,
            tenant_id=state["tenant_id"],
            project_id=state["project_id"],
            query_vector=query_vector,
            types=["requirement", "user_story", "acceptance_criteria"],
            top_k=8,
        )
    except Exception as e:
        print(f"[scenario-agent] warning: could not retrieve RAG context: {e}")
        context_chunks = []
    
    context_block = "\n".join(f"- {c}" for c in context_chunks) if context_chunks else "(no project documents indexed)"
    
    # Step 2: Get actual page structure
    page_structure = state.get("page_structure", "(could not fetch page structure)")
    
    # Step 3: Build the user prompt
    user_prompt = f"""
=== TARGET APPLICATION ===
URL: {state['url']}
Page Title: {state.get('page_title', 'Unknown')}
Viewport: {state.get('viewport_size', {}).get('width', 1920)}x{state.get('viewport_size', {}).get('height', 1080)}

=== TEST INTENT ===
{state['prompt']}

=== PROJECT CONTEXT (from requirements) ===
{context_block}

=== CURRENT PAGE STRUCTURE (what's actually on the page) ===
{page_structure}

{"=== PREVIOUS ERROR (SELF-HEALING) ===" + chr(10) + state.get('_previous_error', '') if state.get('_previous_error') else ""}

Now generate the Playwright test script body.
"""
    
    print(f"[scenario-agent] calling LLM...")
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        
        script_body = completion.choices[0].message.content.strip()
        
        # Remove accidental markdown fences (LLMs often do this despite instructions)
        script_body = script_body.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        
        lines = script_body.split('\n')
        print(f"[scenario-agent] generated {len(lines)} lines of test code")
        print(f"[scenario-agent] first 5 lines:")
        for line in lines[:5]:
            print(f"  {line}")
        
        return {
            **state,
            "requirements_context": context_chunks,
            "playwright_script": script_body,
            "status": "scenario_generated",
        }
    
    except Exception as e:
        print(f"[scenario-agent] error calling LLM: {e}")
        # Return a fallback script
        fallback_script = f"""
# Fallback script - LLM call failed
page.goto("{state['url']}")
page.wait_for_load_state("load")
# Basic assertions
assert page.url == "{state['url']}" or "{state['url']}" in page.url
print("Basic navigation test passed")
"""
        return {
            **state,
            "requirements_context": [],
            "playwright_script": fallback_script,
            "status": "scenario_generated",
            "error": str(e),
        }