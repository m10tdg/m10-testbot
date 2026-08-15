"""
Dynamic Crawler Node

This node:
1. Actually visits the URL
2. Extracts interactive elements (buttons, inputs, forms)
3. Captures the viewport size and page structure
4. Returns clean HTML for the LLM to understand what's actually on the page

This solves the "Locator.click: Timeout" problem because the LLM now knows
what elements actually exist, what their attributes are, and how to target them.
"""

import time
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


def clean_html_for_llm(html_content: str, url: str) -> str:
    """
    Extract interactive elements and page structure.
    Returns a clean, focused HTML snippet that helps LLM understand the page.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove noise
    for tag in soup.find_all(['script', 'style', 'meta', 'link']):
        tag.decompose()
    
    # Extract interactive elements with their attributes
    interactive = []
    for el in soup.find_all(['button', 'input', 'select', 'textarea', 'form', 'a']):
        attrs = {}
        for key in ['id', 'name', 'class', 'type', 'placeholder', 'aria-label', 'role', 'href', 'action']:
            if el.get(key):
                attrs[key] = el.get(key)
        
        text = el.get_text(strip=True)[:100]  # Limit text length
        
        interactive.append({
            'tag': el.name,
            'attributes': attrs,
            'text': text,
        })
    
    # Build clean HTML for LLM
    clean_html = f"<page url='{url}'>\n"
    clean_html += f"<title>{soup.find('title').get_text() if soup.find('title') else 'No title'}</title>\n"
    clean_html += f"<interactive-elements count='{len(interactive)}'>\n"
    
    for el in interactive[:30]:  # Limit to 30 elements for token efficiency
        attrs_str = ' '.join([f"{k}='{v}'" for k, v in el['attributes'].items()])
        if attrs_str:
            clean_html += f"  <{el['tag']} {attrs_str}>{el['text']}</{el['tag']}>\n"
        else:
            clean_html += f"  <{el['tag']}>{el['text']}</{el['tag']}>\n"
    
    clean_html += "</interactive-elements>\n"
    clean_html += "</page>"
    
    return clean_html


def crawler_node(state: dict) -> dict:
    """
    Crawl the live URL and extract its structure.
    
    This node runs BEFORE scenario generation so the Scenario Agent
    knows what elements are actually available on the page.
    """
    url = state["url"]
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
                # If networkidle times out, that's OK - we'll proceed anyway
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
        # Return empty structure if crawl fails - scenario agent will work with RAG context only
        return {
            **state,
            "page_structure": f"<page url='{url}'><error>{str(e)}</error></page>",
            "page_title": "Unknown",
            "viewport_size": {"width": 1920, "height": 1080},
        }
    
    # Clean the HTML
    clean_structure = clean_html_for_llm(html_content, url)
    
    print(f"[crawler-node] extracted {len(clean_structure)} chars of structure")
    print(f"[crawler-node] page title: {page_title}")
    print(f"[crawler-node] viewport: {viewport_size['width']}x{viewport_size['height']}")
    
    return {
        **state,
        "page_structure": clean_structure,
        "page_title": page_title,
        "viewport_size": viewport_size,
    }