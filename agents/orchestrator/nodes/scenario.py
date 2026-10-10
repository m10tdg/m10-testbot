"""
Enhanced Scenario Agent v4

Key change from v3: the system prompt was heavily biased toward
"find a form input and fill it" (login-style flows). Real test intents
span many action types (mode toggles, preset/theme selection, CRUD with
role selection, sidebar navigation, filters, etc). v4 generalizes by:

1. Forcing an explicit "plan before code" step so the LLM classifies the
   action type and names its evidence/target BEFORE writing selectors.
2. Broadening the element-finding strategy beyond <input>/<button> to any
   clickable element (divs, list items, swatches, menu items, toggles).
3. Adding action-type playbooks with the RIGHT kind of assertion for each
   (mode toggle -> class/toolbar change, theme -> CSS var/class change,
   CRUD -> new row appears with expected value, nav -> URL/active state).
4. Explicitly handling "target isn't in the DOM map yet" (it may only
   appear after opening a menu/tab/dropdown) instead of assuming presence.
5. Generalizing the self-healing retry section so it isn't anchored to
   an "email input" example.
6. Nudging the agent to derive assertions from retrieved acceptance
   criteria when available, and sensible defaults otherwise.
"""

import openai
from clients import embed, qdrant, OPENAI_API_KEY
from rag import retrieve_context

openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT = """You are a senior QA automation engineer writing Playwright tests
against arbitrary, complex, and dynamic enterprise web applications. Each test intent 
may describe a completely different interaction. Handle all action types equally well:
authentication, navigation, mode toggles, themes/visual presets, CRUD operations with 
complex controls, iFrames/Shadow DOM, multi-tab popups, drag-and-drop, and filtering.

=== OUTPUT CONTRACT ===
1. You write ONLY the function body (no imports, no setup, no signature).
2. The body has access to:
   - `page`: an already-open Playwright Page object
   - `console_errors`: a list you may append to for tracking console errors
3. Use SYNCHRONOUS Playwright API only:
   - Correct: page.goto(url), page.click(selector), page.fill(selector, text)
   - Wrong:   await page.goto()  — NEVER use async/await.
4. Return ONLY raw Python code. No markdown code fences, no explanations outside comments.

=== STEP 0 — PLAN BEFORE YOU CODE (write as leading comments) ===
Before writing Playwright calls, write a short comment block (3-6 lines):
  1. Classify the primary action type (see PLAYBOOKS below).
  2. List the ordered UI interaction flow.
  3. Identify element boundary challenges (e.g., inside an iFrame, Shadow DOM, or hidden until hover/click).
  4. Name the concrete DOM/Visual evidence for targeting each element.
  5. State the concrete, observable success signal to assert on (derived directly from page structure or context).

=== DOM BOUNDARIES: IFRAMES, SHADOW DOM & POPUPS ===
- IFRAMES: If an element resides inside an `<iframe>`, locate the frame first:
    frame = page.frame_locator('iframe#target-frame')
    frame.locator('button#submit').click()
- SHADOW DOM: Playwright piercing selectors penetrate Shadow DOM by default. Use standard
  locators (`page.locator('custom-element >> button')`) unless closed Shadow roots require shadow-piercing paths.
- NEW TABS / POPUPS: If an action opens a new browser window/tab (e.g., OAuth/SSO login):
    with page.expect_popup() as popup_info:
        page.click('#sso-button')
    popup = popup_info.value
    popup.wait_for_load_state("domcontentloaded")

=== ELEMENT FINDING STRATEGY (GENERALIZED ENUMERATION) ===
Interactive controls can be `<div>`, `<span>`, `<svg>`, or custom web components. Try locating elements in this strict priority order:
  a) id                       — page.locator("#exact-id")
  b) data-testid / data-qa    — page.locator('[data-testid="..."]')
  c) name attribute           — page.locator('input[name="..."]')
  d) aria-label / role        — page.get_by_role(role, name="...")
  e) exact visible text       — page.get_by_text("...", exact=True)
  f) placeholder / label      — page.get_by_label(...) / get_by_placeholder(...)
  g) href fragment            — page.locator('a[href*="..."]')
  h) CSS class + text         — page.locator('.btn-primary:has-text("...")')
  i) XPath contains(text())   — final fallback only

CRITICAL — REVEAL BEFORE LOCATE:
If a target is absent from the current DOM map, it may require a prior action (expanding a menu, hovering over a parent card, clicking a tab, opening a modal). Perform the revealing action, wait for UI stabilization, and THEN query the target.

=== ACTION-TYPE PLAYBOOKS ===
- AUTHENTICATION: Fill credentials -> submit -> assert URL changed away from login route OR container elements visible post-login.
- NAVIGATION (Sidebar / Tab / Menu): Click item -> wait for route or DOM change -> assert target section container/heading visible or active state indicator set.
- MODE TOGGLES / CANVASES: Click trigger control -> assert class/attribute shifts (e.g., `is-editing`, `aria-checked="true"`), toolbars appearing, or input fields becoming enabled.
- PRESETS / THEME / STYLING: Open configuration panel -> select option -> assert root/body class changes, style attribute update, or selection checkmark/border appearing on target element.
- CRUD OPERATIONS: Execute create/update/delete flow -> submit -> assert target item appears in list/table (or row count changes, or modal closes). Clean up created test data when feasible.
- SEARCH & FILTERING: Enter query/apply filters -> wait for dynamic update -> assert row count decreases/changes or specific search query string appears in URL params or active filter chip.

=== DYNAMIC ASSERTIONS — GROUNDED & RESILIENT (NO HARDCODED GUESSES) ===
CRITICAL: NEVER hardcode arbitrary strings like "Dashboard", "Overview", or "Success" unless they appear explicitly in the `PROJECT CONTEXT` or `ACTUAL PAGE STRUCTURE`. Flaky assertions cause false failures. 

Use these resilient assertion strategies based on available ground truth:

1. URL REDIRECTION & DELTA ASSERTIONS:
   - Capture initial URL: `initial_url = page.url`
   - Assert page navigated away: `assert page.url != initial_url`
   - Assert route pattern (if clear from context): `assert "/login" not in page.url`

2. GROUNDED TEXT ASSERTIONS (From DOM Map / RAG Context):
   - Assert ONLY on text strings present in the `ACTUAL PAGE STRUCTURE` or retrieved `PROJECT CONTEXT`.
   - Example: If the target page structure shows a main heading `<h1 class="title">Projects</h1>`, use:
     `assert page.locator("h1.title").is_visible()` or `assert "Projects" in page.locator("h1").text_content()`

3. STATE & STRUCTURAL DELTA ASSERTIONS (When Text is Unknown):
   - Element Disappearance (Modals, Delete operations, Toast dismissals):
     `page.wait_for_selector("#modal-dialog", state="detached", timeout=10000)`
     `assert not page.locator("#modal-dialog").is_visible()`
   - Attribute & Class Changes (Toggles, Selections, Active Links):
     `assert page.locator(target_selector).get_attribute("aria-selected") == "true"`
     `assert "active" in (page.locator(target_selector).get_attribute("class") or "")`
   - List / Table Count Updates (CRUD & Filtering):
     `assert page.locator("table row").count() > initial_count`

4. UNIVERSAL CONTAINER VISIBILITY (Post-Auth / Navigation Fallback):
   - If authenticating or navigating without explicit text requirements, assert the presence of structural main application layouts:
     `assert page.locator("main, [role='main'], #app, #root").is_visible()`

=== WAITING, STABILITY & ASYNC BEHAVIOR ===
- Prefer explicit state predicate waiting over arbitrary sleep timers:
    page.wait_for_selector(selector, state="visible", timeout=15000)
    page.wait_for_load_state("networkidle")
- Wrap high-risk UI transitions in try/except blocks with clear diagnostic error logging appended to `console_errors`.

=== SELF-HEALING RETRY RULES ===
If this execution is a RETRY:
  1. Inspect the provided runtime DOM map to verify the page's current state.
  2. Switch locator strategies from the failing run (e.g., if `get_by_label` failed, switch to `data-testid` or `id`).
  3. Verify if missing targets required an unexecuted preceding "reveal" step.
  4. Re-evaluate the assertion: if the previous failure was `AssertionError`, check if the expected text was an incorrect guess and switch to a state/URL delta assertion.
"""


def scenario_agent_node(state: dict) -> dict:
    """
    Generate a Playwright test script for the given URL and prompt.

    Generalized to handle any UI action type (not just form-filling):
    1. Uses actual page structure from crawler
    2. Applies RAG context from requirements/acceptance criteria
    3. Includes previous error context for self-healing retries
    4. Asks the LLM to classify the action type and plan before coding
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

PREVIOUS SCRIPT BODY (note which selector strategy it used):
{retry_ctx.get('previous_script', '(not captured)')}

ERROR MESSAGE:
{retry_ctx.get('previous_error', 'Unknown error')}

ERROR DETAILS:
{retry_ctx.get('previous_error_context', 'No additional context')}

CONSOLE ERRORS CAPTURED:
{chr(10).join(['- ' + err for err in retry_ctx.get('console_errors', [])]) if retry_ctx.get('console_errors') else 'None'}

ACTUAL PAGE STRUCTURE TO USE (extract exact selectors from this, and switch
strategy from whatever the previous script used):
"""

    retry_instructions = ""
    if is_retry:
        retry_instructions = """
=== IMPORTANT FOR THIS RETRY ===
- Do NOT reuse the same selector strategy that failed last time.
- Match EXACT text/ids/names from the DOM map above.
- If the target never appeared, check whether a "reveal" action (opening a
  menu/tab/dropdown) is missing before you try to locate it.
- Keep your plan-first comment block, but call out explicitly what changed
  from the previous attempt and why.
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

=== PROJECT CONTEXT (requirements / acceptance criteria, if any) ===
{context_block}

{retry_context_block}

=== ACTUAL PAGE STRUCTURE (THIS IS THE GROUND TRUTH) ===
{page_structure}

{retry_instructions}

Now generate the Playwright test script body.
Remember:
1. Start with the plan-first comment block (classify action type, list steps,
   name evidence per step, state the success signal).
2. Use EXACT selectors from the DOM structure shown above; if a target isn't
   shown, reveal it first via the appropriate prior action.
3. Include fallback strategies for elements central to the test.
4. Add clear, specific error messages for debugging.
5. Assert on the signal that matches the action type, not a generic check.
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
            temperature=0.2 if not is_retry else 0.3,
        )

        script_body = completion.choices[0].message.content.strip()
        script_body = script_body.removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        print("--------------------Script Body------------------------")
        print(script_body)
        print("--------------------------------------------")
        lines = script_body.split('\n')
        print(f"[scenario-agent] generated {len(lines)} lines of test code")

        return {
            **state,
            "requirements_context": context_chunks,
            "playwright_script": script_body,
            "status": "scenario_generated",
            "generation_mode": "retry_self_healing" if is_retry else "first_attempt",
        }

    except Exception as e:
        print(f"[scenario-agent] error calling LLM: {e}")
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