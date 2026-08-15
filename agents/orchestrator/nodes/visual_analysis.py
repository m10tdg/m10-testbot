"""
Enhanced Visual Analysis Agent

Instead of just pixel comparison, this agent:
1. Uses SSIM for structural similarity
2. Detects regions of change
3. Uses LLM to semantically evaluate if change is real or expected
4. Provides actionable recommendations
"""

import tempfile
import cv2
import numpy as np
import openai
from skimage.metrics import structural_similarity as ssim
from botocore.exceptions import ClientError
from clients import s3, ARTIFACTS_BUCKET, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

VISUAL_ANALYSIS_PROMPT = """You are a visual regression testing expert.

Given:
- Current screenshot details
- Baseline screenshot details
- Pixel difference percentage
- Change description

Analyze whether this is:
A) Real bug (visual defect affecting UX)
B) Expected update (intentional design change documented in requirements)
C) Environmental difference (font rendering, OS differences)

Provide:
1. Classification (real_bug / expected_update / environmental)
2. Severity if it's a real bug (critical / warning / info)
3. Recommendation for the developer
4. Affected UI component(s)

Be concise and actionable."""


def visual_analysis_agent_node(state: dict) -> dict:
    """
    Compare current screenshot against baseline.
    Use both pixel analysis and semantic AI evaluation.
    """
    print(f"\n[visual-analysis-agent] analyzing visual changes for run {state['run_id']}")
    
    tenant_id = state["tenant_id"]
    project_id = state["project_id"]
    run_id = state["run_id"]
    visual_diffs = []
    
    # Get execution results (should have screenshots)
    execution_results = state.get("execution_results", [])
    if not execution_results:
        print(f"[visual-analysis-agent] no execution results, skipping visual analysis")
        return {**state, "visual_diffs": []}
    
    for exec_result in execution_results:
        if not exec_result.get("screenshot_s3_path"):
            continue
        
        page_url = exec_result["pageUrl"]
        print(f"\n[visual-analysis-agent] comparing {page_url}...")
        
        # Download current screenshot
        current_local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        try:
            s3.download_file(
                ARTIFACTS_BUCKET,
                exec_result["screenshot_s3_path"],
                current_local
            )
        except Exception as e:
            print(f"[visual-analysis-agent] could not download current screenshot: {e}")
            continue
        
        # Get or create baseline
        # Baseline naming: use simplified URL as identifier
        baseline_name = page_url.split("://")[-1].split("/")[0]  # e.g., "taskflow.com"
        baseline_key = f"{tenant_id}/{project_id}/baselines/{baseline_name}.png"
        baseline_local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        
        baseline_exists = True
        try:
            s3.download_file(ARTIFACTS_BUCKET, baseline_key, baseline_local)
            print(f"[visual-analysis-agent] baseline exists, comparing...")
        except ClientError as e:
            if "NoSuchKey" in str(e):
                print(f"[visual-analysis-agent] no baseline found, creating one...")
                try:
                    s3.upload_file(current_local, ARTIFACTS_BUCKET, baseline_key)
                    print(f"[visual-analysis-agent] baseline created at {baseline_key}")
                except Exception as upload_err:
                    print(f"[visual-analysis-agent] could not save baseline: {upload_err}")
                baseline_exists = False
            else:
                print(f"[visual-analysis-agent] error accessing baseline: {e}")
                continue
        
        if not baseline_exists:
            # First run - no comparison to make
            continue
        
        # Compare images
        try:
            current_img = cv2.imread(current_local)
            baseline_img = cv2.imread(baseline_local)
            
            if current_img is None or baseline_img is None:
                print(f"[visual-analysis-agent] could not read image files")
                continue
            
            # Resize to match if dimensions differ
            if current_img.shape != baseline_img.shape:
                print(f"[visual-analysis-agent] resizing: {current_img.shape} -> {baseline_img.shape}")
                current_img = cv2.resize(current_img, (baseline_img.shape[1], baseline_img.shape[0]))
            
            # Calculate SSIM
            gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            gray_baseline = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
            
            score, diff = ssim(gray_current, gray_baseline, full=True)
            difference_percent = round((1 - score) * 100, 2)
            
            print(f"[visual-analysis-agent] SSIM score: {score:.4f} ({difference_percent}% different)")
            
            # Threshold: if <1% different, definitely expected
            if difference_percent < 1.0:
                verdict = "no_change"
                severity = "info"
                needs_review = False
            # 1-5% might be rendering differences
            elif difference_percent < 5.0:
                verdict = "minor_change"
                needs_review = True
                severity = "info"
            # >5% is definitely significant
            else:
                verdict = "significant_change"
                needs_review = True
                severity = "warning"
            
            # If significant change, use LLM to understand what changed
            ai_analysis = None
            if needs_review and difference_percent > 2.0:
                try:
                    # Find regions of change
                    diff_normalized = (diff * 255).astype("uint8")
                    threshold = cv2.threshold(diff_normalized, 50, 255, cv2.THRESH_BINARY)[1]
                    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    changed_regions = []
                    for contour in contours[:5]:  # Top 5 changed regions
                        x, y, w, h = cv2.boundingRect(contour)
                        if w > 10 and h > 10:  # Ignore tiny noise
                            changed_regions.append(f"Region at ({x},{y}) size {w}x{h}")
                    
                    # Ask LLM what this might mean
                    prompt = f"""
URL: {page_url}
Difference: {difference_percent}%

Changed regions detected:
{chr(10).join(changed_regions)}

What likely changed?
(Be specific about which UI components are affected.)
"""
                    
                    completion = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": VISUAL_ANALYSIS_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=200,
                    )
                    ai_analysis = completion.choices[0].message.content
                    print(f"[visual-analysis-agent] AI analysis: {ai_analysis[:100]}...")
                
                except Exception as e:
                    print(f"[visual-analysis-agent] could not get AI analysis: {e}")
            
            visual_diffs.append({
                "page": page_url,
                "difference_percent": difference_percent,
                "verdict": verdict,
                "severity": severity,
                "needs_review": needs_review,
                "ai_analysis": ai_analysis,
                "baseline_key": baseline_key,
            })
        
        except Exception as e:
            print(f"[visual-analysis-agent] error comparing images: {e}")
    
    print(f"\n[visual-analysis-agent] found {len([d for d in visual_diffs if d['needs_review']])} visual differences requiring review")
    
    return {**state, "visual_diffs": visual_diffs}