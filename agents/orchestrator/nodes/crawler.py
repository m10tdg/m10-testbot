"""
Improved Crawler Node v3 - Extracts better DOM information

Improvements:
1. Extracts ALL interactive elements (not just top 50)
2. Includes actual text values and placeholder text
3. Better attribute extraction
4. More descriptive labels for the LLM
5. Captures form structure (which elements are in which forms)
6. Handles retry re-crawling internally (checks attempt number)
"""

import time
import json
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def extract_element_info(el) -> dict:
    """
    Extract comprehensive information about an element.
    """
    info = {
        'tag': el.name,
        'text': el.get_text(strip=True)[:200],  # Limit but don't truncate too much
    }
    
    # Extract all potentially useful attributes
    attrs = {}
    for key in ['id', 'name', 'class', 'type', 'placeholder', 'aria-label', 'role', 
                'href', 'action', 'data-testid', 'data-qa', 'value']:
        val = el.get(key)
        if val:
            attrs[key] = val if isinstance(val, str) else ' '.join(val)
    
    if attrs:
        info['attributes'] = attrs
    
    return info


def build_dom_map(html_content: str) -> str:
    """
    Build a detailed DOM map showing form structure and interactive elements.
    This gives the LLM much better understanding of the page.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove noise
    for tag in soup.find_all(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()
    
    dom_map = []
    dom_map.append("PAGE STRUCTURE MAP")
    dom_map.append("=" * 60)
    
    # Get page title
    title = soup.find('title')
    if title:
        dom_map.append(f"Title: {title.get_text()}")
    
    # Find and document forms
    forms = soup.find_all('form')
    if forms:
        dom_map.append(f"\nFORMS ({len(forms)} found):")
        for i, form in enumerate(forms, 1):
            form_id = form.get('id', f'form-{i}')
            form_action = form.get('action', 'N/A')
            dom_map.append(f"  Form {i}: id={form_id}, action={form_action}")
            
            # List inputs in this form
            inputs = form.find_all(['input', 'textarea', 'select'])
            for inp in inputs[:10]:  # Max 10 inputs per form
                inp_type = inp.get('type', 'text')
                inp_name = inp.get('name', 'unnamed')
                inp_id = inp.get('id', '')
                inp_placeholder = inp.get('placeholder', '')
                inp_label = inp.get('aria-label', '')
                
                desc = f"{inp_type} '{inp_name}'"
                if inp_id:
                    desc += f" (id={inp_id})"
                if inp_placeholder:
                    desc += f" placeholder='{inp_placeholder}'"
                if inp_label:
                    desc += f" label='{inp_label}'"
                
                dom_map.append(f"    - {desc}")
            
            if len(form.find_all(['input', 'textarea', 'select'])) > 10:
                dom_map.append(f"    ... and {len(form.find_all(['input', 'textarea', 'select'])) - 10} more")
            
            # Find submit button
            submit = form.find('button', {'type': 'submit'})
            if submit:
                submit_text = submit.get_text(strip=True)
                submit_id = submit.get('id', '')
                dom_map.append(f"    → Submit button: '{submit_text}'" + (f" (id={submit_id})" if submit_id else ""))
    
    # Find standalone buttons
    buttons = soup.find_all('button')
    if buttons:
        dom_map.append(f"\nBUTTONS ({len(buttons)} found):")
        for btn in buttons[:15]:  # Show first 15
            btn_text = btn.get_text(strip=True)
            btn_id = btn.get('id', '')
            btn_type = btn.get('type', 'button')
            btn_class = btn.get('class', [])
            
            desc = f"'{btn_text}'"
            if btn_id:
                desc += f" (id={btn_id})"
            if btn_type != 'button':
                desc += f" [type={btn_type}]"
            
            dom_map.append(f"  - {desc}")
        
        if len(buttons) > 15:
            dom_map.append(f"  ... and {len(buttons) - 15} more buttons")
    
    # Find input fields (not in forms)
    inputs = soup.find_all(['input', 'textarea', 'select'])
    if inputs:
        dom_map.append(f"\nINPUT FIELDS ({len(inputs)} found, including form inputs):")
        for inp in inputs[:20]:  # Show first 20
            inp_type = inp.get('type', 'text')
            inp_id = inp.get('id', '')
            inp_name = inp.get('name', '')
            inp_placeholder = inp.get('placeholder', '')
            inp_label = inp.get('aria-label', '')
            
            identifiers = []
            if inp_id:
                identifiers.append(f"id={inp_id}")
            if inp_name:
                identifiers.append(f"name={inp_name}")
            if inp_placeholder:
                identifiers.append(f"placeholder='{inp_placeholder}'")
            if inp_label:
                identifiers.append(f"label='{inp_label}'")
            
            desc = f"{inp_type}"
            if identifiers:
                desc += f": {', '.join(identifiers)}"
            
            dom_map.append(f"  - {desc}")
        
        if len(inputs) > 20:
            dom_map.append(f"  ... and {len(inputs) - 20} more")
    
    # Find links
    links = soup.find_all('a', {'href': True})
    if links:
        dom_map.append(f"\nLINKS ({len(links)} found):")
        for link in links[:10]:
            link_text = link.get_text(strip=True)[:50]
            link_href = link.get('href', '')
            link_id = link.get('id', '')
            
            desc = f"'{link_text}'"
            if link_id:
                desc += f" (id={link_id})"
            
            dom_map.append(f"  - {desc} → {link_href}")
        
        if len(links) > 10:
            dom_map.append(f"  ... and {len(links) - 10} more")
    
    # Find headings (good for page structure)
    headings = soup.find_all(['h1', 'h2', 'h3'])
    if headings:
        dom_map.append(f"\nHEADINGS (page structure):")
        for heading in headings[:10]:
            heading_text = heading.get_text(strip=True)[:60]
            dom_map.append(f"  <{heading.name}> {heading_text}")
    
    dom_map.append("\n" + "=" * 60)
    
    return "\n".join(dom_map)


def crawler_node(state: dict) -> dict:
    """
    Crawl the live URL and extract comprehensive DOM structure.
    
    Handles both first-attempt and retry scenarios:
    - Attempt 1: Standard crawl
    - Attempt 2+: Re-crawl to get fresh page state
    
    Returns detailed information about:
    - Forms and their inputs
    - Buttons and their labels
    - Input fields with IDs/names/placeholders
    - Links
    - Page headings
    """
    url = state["url"]
    attempt = state.get("attempt", 1)
    
    # Check if this is a retry
    is_retry = attempt > 1
    
    if is_retry:
        print(f"\n[crawler-node] (RETRY) re-visiting {url} for fresh page state...")
    else:
        print(f"\n[crawler-node] visiting {url}...")
    
    try:
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set a reasonable timeout
            page.set_default_timeout(10000)  # 10 seconds
            
            # Navigate
            page.goto(url, wait_until="load")
            
            # Wait for network to settle
            try:
                page.wait_for_load_state("networkidle", timeout=5000)
            except:
                # If networkidle times out, proceed anyway
                pass
            
            # Give the page a moment to render
            time.sleep(1)
            
            # Capture what we need
            html_content = page.content()
            viewport_size = page.evaluate("() => ({ width: window.innerWidth, height: window.innerHeight })")
            page_title = page.title()
            
            browser.close()
    
    except Exception as e:
        print(f"[crawler-node] error visiting {url}: {e}")
        return {
            **state,
            "page_structure": f"ERROR: Could not visit {url}: {str(e)}",
            "page_title": "Error",
            "viewport_size": {"width": 1920, "height": 1080},
        }
    
    # Build the DOM map
    dom_map = build_dom_map(html_content)
    
    mode_label = "(RETRY)" if is_retry else "(FIRST ATTEMPT)"
    print(f"[crawler-node] {mode_label} extracted comprehensive page structure")
    print(f"[crawler-node] page title: {page_title}")
    print(f"[crawler-node] viewport: {viewport_size['width']}x{viewport_size['height']}")
    print(f"[crawler-node] DOM map size: {len(dom_map)} chars")
    
    return {
        **state,
        "page_structure": dom_map,
        "page_title": page_title,
        "viewport_size": viewport_size,
    }