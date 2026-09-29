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
against arbitrary, previously-unseen web applications. Each test intent may describe
a completely different kind of interaction — do not assume it's a login/form-fill
unless the intent literally says so. Examples you must handle equally well:
authentication, entering an edit/mode state, applying a visual preset or theme,
adding/editing/removing a resource (often with a role or type selection),
navigating via a sidebar or menu, toggling a setting, filtering or searching a list.

=== OUTPUT CONTRACT ===
1. You write ONLY the function body (no imports, no setup, no signature).
2. The body has access to:
   - `page`: an already-open Playwright Page object
   - `console_errors`: a list you may append to for tracking errors
3. Use SYNCHRONOUS Playwright API only:
   - correct: page.goto(url), page.click(selector), page.fill(selector, text)
   - wrong:   await page.goto()  — never use async/await
4. Return ONLY Python code. No markdown fences, no explanations outside comments.

=== STEP 0 — PLAN BEFORE YOU CODE (write as leading comments) ===
Before writing any Playwright calls, write a short comment block (3-6 lines) that:
  1. Classifies the action type this intent falls under (see PLAYBOOKS below)
  2. Restates the goal as an ordered list of discrete UI actions
     (e.g. navigate -> open sidebar section -> click sub-item -> verify)
  3. For each action, names the DOM evidence you're using to find the target
     (id/text/aria-label/etc), or notes "not present in DOM map — will reveal
     it by performing the prior action first, then re-locate it"
  4. States the concrete, observable success signal you will assert on
This plan is your checklist. The code that follows should implement it step by step.
Getting this classification right is more important than any single selector choice.

=== ELEMENT FINDING STRATEGY (GENERALIZED — NOT JUST FORM INPUTS) ===
Real apps put interactive behavior on all kinds of elements: divs/spans acting as
buttons, icon-only buttons, swatches, cards, toggle switches, list items, sidebar
links, menu items, custom dropdowns. Never assume the target is an <input> or
<button> just because that's common. For each element you need, try in this order
and keep whichever check actually matches something on the page:
  a) id                              — page.locator("#exact-id")
  b) data-testid / data-qa           — page.locator('[data-testid="..."]')
  c) name attribute (form fields)    — page.locator('input[name="..."]')
  d) aria-label / role               — page.get_by_role(role, name="...")
  e) exact visible text              — page.get_by_text("...", exact=True)
  f) placeholder / associated label  — page.get_by_label(...) / get_by_placeholder(...)
  g) href fragment (nav links)       — page.locator('a[href*="..."]')
  h) class hint + text combined      — last resort before XPath
  i) XPath contains(text())          — final fallback only

CRITICAL: if the DOM map provided does NOT show your target element at all, that
usually means it only appears after a prior action — opening a menu, expanding a
sidebar section, switching a tab, opening a dropdown. Do NOT guess a selector for
something that isn't shown. Instead: perform the revealing action, wait for the
resulting UI to settle, and THEN locate the now-visible target using the evidence
available at that point (even if that means using text-based matching because you
don't have an id for something you couldn't see in advance).

For elements with multiple plausible matches, prefer the most specific scope
available (e.g. locate within a named form/section/list first, then the element
inside it) rather than a bare global selector.

=== ACTION-TYPE PLAYBOOKS (pick the closest fit; assert on the RIGHT signal) ===
- AUTHENTICATION: fill credential fields -> submit -> assert redirect away from the
  login URL, OR a post-login element appears (avatar/menu/dashboard heading). For
  negative-path tests, assert an inline error/toast appears instead.
- NAVIGATION (sidebar / menu / tabs): click the nav item by text/aria-label/href
  fragment -> wait for URL change or new content to render -> assert the active/
  selected state or the new section's heading/content is visible.
- MODE TOGGLES (e.g. "enter edit mode"): click the trigger control -> assert a
  mode-specific signal — a new toolbar or Save/Cancel controls appear, a class such
  as "editing"/"is-editing" gets applied, or previously static content becomes
  interactive (inputs/drag-handles appear).
- PRESETS / THEME / APPEARANCE SETTINGS: open the relevant settings/appearance
  area -> click the specific preset/option, usually a swatch, card, or list item
  best identified by its visible text or aria-label rather than an id -> assert a
  visible change: a class on <body>/<html> changes, a CSS custom property updates,
  or a "selected/active" indicator appears on the chosen option.
- CRUD OPERATIONS (add/edit/remove a resource, e.g. "add member with owner role"):
  open the create/add flow -> fill the relevant fields -> for role/type selection,
  use select_option() for a real <select>, or get_by_role("option")/click-based
  selection for a custom dropdown -> submit -> assert the new/changed item appears
  in the corresponding list/table with the expected value (e.g. role label).
- FILTER / SEARCH: interact with the filter/search control -> assert the visible
  result set changes appropriately (expected items present/absent, or count change).

If the intent doesn't cleanly match a playbook above, treat it as a novel case:
still follow the plan-first approach, infer a reasonable UI flow from the DOM map,
and assert on the most specific observable signal you can find rather than a vague
"page didn't crash" check.

=== USE PROJECT CONTEXT WHEN AVAILABLE ===
If retrieved requirements/acceptance criteria are provided, use them to sharpen
your assertions — e.g. if acceptance criteria say "owner role displays a crown
icon", assert on that specific detail rather than a generic "row exists" check.
If no relevant context was retrieved, fall back to the most sensible assertion
implied by the DOM map and the action-type playbook above.

=== WAITING & STABILITY ===
- After any action that can trigger navigation or async UI updates, wait
  explicitly: page.wait_for_load_state("networkidle") or wait_for_selector on the
  expected resulting element (timeout 3000-8000ms). Do not rely on bare sleeps.
- Wrap steps that could legitimately fail in try/except with a specific,
  actionable error message that names what was being looked for and where.

=== SELF-HEALING RETRY MODE ===
If this is a retry (indicated in the prompt), a previous attempt failed. Rules:
  1. Read the actual DOM map provided below — it reflects the page's current,
     real state. Use exact ids/names/text from it, never guesses.
  2. Identify which selector strategy the previous attempt used (see the failed
     script/error below) and switch to a DIFFERENT strategy from the preference
     order above — don't repeat the same approach that already failed.
  3. If the previous error suggests the target never appeared, reconsider whether
     a prior "reveal" action (opening a menu/tab/dropdown) was missing, and add it.
  4. Keep the same overall step plan/goal classification unless the error clearly
     shows the classification itself was wrong.
  5. Always include a fallback selector list for the element that failed before.

Example:
  DOM shows:            <input id="email-field" type="email" />
  Previous attempt used: page.get_by_label("Email")   (failed)
  This attempt use:      page.locator("#email-field")  (different strategy)

=== ASSERTIONS — BE SPECIFIC ===
- URL changes:        assert "dashboard" in page.url
- Element visible:     assert page.get_by_text("Welcome").is_visible()
- Text content:        heading = page.get_by_role("heading", level=1)
                        assert heading.text_content().strip() == "Dashboard"
- Attribute/class:     classes = page.locator("body").get_attribute("class") or ""
                        assert "editing" in classes

REMEMBER: the plan-first comment block and the action-type classification matter
more than any individual selector. DOM structure and retrieved context are
provided as ground truth — use them, and reveal-before-locate when something
isn't shown yet. Don't guess blind.
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