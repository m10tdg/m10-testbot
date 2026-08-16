"""
Enhanced Scenario Agent v3

Improvements:
1. Multi-strategy selector generation (ID, name, label, text, XPath)
2. Better DOM element highlighting for retries
3. Explicit fallback selector lists in generated code
4. Previous error context properly integrated
5. Visual debugging aids (selector comments with alternatives)
"""

import openai
import re
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

4. ELEMENT FINDING STRATEGY - TRY MULTIPLE APPROACHES:
   When finding an element, ALWAYS have a fallback strategy.
   Use this pattern for critical elements:
   
   ```python
   # Strategy: Try multiple selectors in order
   element = None
   selectors_to_try = [
       lambda: page.locator("#email-input"),           # Try ID first
       lambda: page.get_by_label("Email"),             # Try label
       lambda: page.get_by_placeholder("email"),       # Try placeholder
       lambda: page.locator('input[name="email"]'),    # Try name attribute
       lambda: page.get_by_role("textbox", name="Email"),  # Try role
   ]
   
   for selector_fn in selectors_to_try:
       try:
           elem = selector_fn()
           elem.wait_for(timeout=2000)
           if elem.is_visible():
               element = elem
               break
       except:
           continue
   
   if not element:
       raise Exception("Could not find email input with any strategy")
   ```

5. PREFERENCE ORDER (always respect this):
   a) By ID: page.locator("#email") - MOST RELIABLE
   b) By data-testid: page.locator('[data-testid="email-input"]')
   c) By exact name: page.locator('input[name="email"]')
   d) By label text: page.get_by_label("Email")
   e) By placeholder: page.get_by_placeholder("Enter email")
   f) By role + name: page.get_by_role("textbox", name="Email")
   g) By button text: page.get_by_text("Submit")
   h) LAST RESORT: page.locator("//button[contains(text(), 'Submit')]")

6. WHEN ELEMENT NOT FOUND - EXPLICIT DEBUGGING:
   Always add helpful error messages:
   
   ```python
   try:
       email = page.get_by_label("Email")
       email.wait_for(timeout=5000)
   except Exception as e:
       # Debug: show what we found instead
       page_text = page.content()
       if "Email" in page_text:
           print("DEBUG: 'Email' text found on page but selector failed")
       else:
           print("DEBUG: 'Email' label not found on page at all")
       raise Exception(f"Email input not found: {e}")
   ```

7. FORM FILLING - ROBUST PATTERN:
   ```python
   # Get element with fallbacks
   email_input = page.locator("#email")
   if not email_input.is_visible():
       email_input = page.get_by_label("Email")
   
   email_input.wait_for(timeout=5000)
   email_input.fill("test@example.com")
   ```

8. ASSERTIONS - BE SPECIFIC:
   - Check URL changes: assert "dashboard" in page.url
   - Check element visibility: assert page.get_by_text("Welcome").is_visible()
   - Check text content: heading = page.get_by_role("heading", level=1)
                         assert heading.text_content() == "Dashboard"

9. CRITICAL: IF THIS IS A RETRY {is_retry_attempt}:
   - Look at the actual DOM provided (copy/pasted below)
   - Use EXACT IDs and text from the DOM, not guesses
   - If previous selector was "get_by_label", try ID or name next
   - Don't repeat the same selector approach
   - Match EXACT element text values from DOM
   
   Example:
   If DOM shows: <input id="email-field" type="email" />
   But previous attempt used: page.get_by_label("Email")
   
   This time use: page.locator("#email-field")
   NOT: page.get_by_label("Email") again

10. RETURN FORMAT:
    - ONLY Python code
    - NO markdown fences
    - NO explanations
    - Code must run synchronously
    - No async/await
    - Comments OK for debugging

REMEMBER: DOM structure is provided. USE IT. Don't guess at selectors.
"""


def scenario_agent_node(state: dict) -> dict:
    """
    Generate a Playwright test script for the given URL and prompt.
    
    Enhanced to:
    1. Use actual page structure from crawler
    2. Apply RAG context from requirements
    3. Include previous error context for self-healing
    4. Generate multi-strategy selectors for robustness
    """
    run_id = state["run_id"]
    attempt = state.get("attempt", 1)
    is_retry = state.get("is_retry", False)
    
    print(f"\n[scenario-agent] generating script for run {run_id} (attempt {attempt})")
    
    # Retrieve project context from RAG
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
    
    # Get actual page structure from crawler
    page_structure = state.get("page_structure", "(could not fetch page structure)")
    
    # Build retry/self-healing context
    retry_context_block = ""
    if is_retry and state.get("retry_context"):
        retry_ctx = state["retry_context"]
        
        retry_context_block = f"""
=== SELF-HEALING MODE: PREVIOUS ATTEMPT FAILED ===

Previous Attempt: {retry_ctx.get('attempt_number', '?')}

ERROR MESSAGE:
{retry_ctx.get('previous_error', 'Unknown error')}

ERROR DETAILS:
{retry_ctx.get('previous_error_context', 'No additional context')}

CONSOLE ERRORS CAPTURED:
{chr(10).join(['- ' + err for err in retry_ctx.get('console_errors', [])]) if retry_ctx.get('console_errors') else 'None'}

WHAT THIS MEANS:
1. The selector strategy from the previous attempt didn't work
2. The element we were looking for might exist but needs a different approach
3. Look at the ACTUAL DOM below and use EXACT IDs/names/text from it
4. Try a different selector strategy (ID → name → label → text → role)
5. Include fallback selectors in the script

ACTUAL PAGE STRUCTURE TO USE (extract exact selectors from this):
"""
    
    # Build retry instructions separately to avoid nested f-string issues
    retry_instructions = ""
    if is_retry:
        retry_instructions = """
=== IMPORTANT FOR THIS RETRY ===
- DO NOT use the same selector that failed last time
- Match EXACT text values from the DOM above
- Use multi-strategy approach with fallbacks
- If an element shows id="xyz", use page.locator("#xyz")
- If an element shows name="email", use page.locator('input[name="email"]')
- Always include try/except for timeouts with helpful error messages
"""
    
    # Build the user prompt
    user_prompt = f"""
=== TARGET APPLICATION ===
URL: {state['url']}
Page Title: {state.get('page_title', 'Unknown')}
Viewport: {state.get('viewport_size', {}).get('width', 1920)}x{state.get('viewport_size', {}).get('height', 1080)}
Is Retry: {is_retry}
Attempt: {attempt}/3

=== TEST INTENT ===
{state['prompt']}

=== PROJECT CONTEXT (from requirements) ===
{context_block}

{retry_context_block}

=== ACTUAL PAGE STRUCTURE (THIS IS THE GROUND TRUTH) ===
{page_structure}

{retry_instructions}

Now generate the Playwright test script body.
Make sure to:
1. Use EXACT selectors from the DOM structure shown above
2. Include fallback strategies for critical elements
3. Add clear error messages for debugging
4. Keep selectors specific and unambiguous
"""
    
    print(f"[scenario-agent] calling LLM...")
    print(f"[scenario-agent] mode: {'RETRY/SELF-HEALING' if is_retry else 'FIRST ATTEMPT'}")
    
    try:
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2 if not is_retry else 0.3,  # Slightly higher temp for retry to encourage different approach
        )
        
        script_body = completion.choices[0].message.content.strip()
        
        # Remove accidental markdown fences
        script_body = script_body.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        
        lines = script_body.split('\n')
        print(f"[scenario-agent] generated {len(lines)} lines of test code")
        if len(lines) > 0:
            print(f"[scenario-agent] first 5 lines:")
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line[:100]}")
        
        return {
            **state,
            "requirements_context": context_chunks,
            "playwright_script": script_body,
            "status": "scenario_generated",
            "generation_mode": "retry_self_healing" if is_retry else "first_attempt",
        }
    
    except Exception as e:
        print(f"[scenario-agent] error calling LLM: {e}")
        # Return a fallback script that at least navigates and checks
        fallback_script = f"""
# Fallback script - LLM call failed
try:
    page.goto("{state['url']}")
    page.wait_for_load_state("load")
    assert page.url == "{state['url']}" or "{state['url']}" in page.url
    print("Basic navigation test passed")
except Exception as e:
    print(f"Fallback script failed: {{e}}")
    raise
"""
        return {
            **state,
            "requirements_context": [],
            "playwright_script": fallback_script,
            "status": "scenario_generated",
            "error": str(e),
            "generation_mode": "fallback",
        }