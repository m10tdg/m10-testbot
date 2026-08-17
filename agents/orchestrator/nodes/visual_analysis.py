"""
Fixed Visual Analysis Agent

Issues fixed:
1. Proper S3 error handling for missing baselines
2. Better exception catching for HeadObject/GetObject errors
3. Graceful baseline creation on first run
4. Fallback if images are missing
"""

import tempfile
import os
import cv2
import numpy as np
import openai
from skimage.metrics import structural_similarity as ssim
from botocore.exceptions import ClientError
from clients import s3, ARTIFACTS_BUCKET, OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

VISUAL_ANALYSIS_PROMPT = """You are a visual regression testing expert.

Given information about visual changes between baseline and current screenshots,
provide analysis of whether this represents:
A) Real bug (visual defect affecting UX)
B) Expected update (intentional design change)
C) Environmental difference (font rendering, minor layout shift)

Provide:
1. Classification
2. Severity if it's a real bug
3. Recommendation

Be concise and actionable."""


def visual_analysis_agent_node(state: dict) -> dict:
    """
    Compare current screenshot against baseline using SSIM + optional AI analysis.
    """
    print(f"\n[visual-analysis-agent] analyzing visual changes for run {state['run_id']}")
    
    tenant_id = state["tenant_id"]
    project_id = state["project_id"]
    run_id = state["run_id"]
    visual_diffs = []
    
    # Get execution results
    execution_results = state.get("execution_results", [])
    if not execution_results:
        print(f"[visual-analysis-agent] no execution results, skipping visual analysis")
        return {**state, "visual_diffs": []}
    
    for exec_result in execution_results:
        if not exec_result.get("screenshot_s3_path"):
            continue
        
        page_url = exec_result["pageUrl"]
        print(f"[visual-analysis-agent] comparing {page_url}...")
        
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
        
        # Determine baseline key
        # Use simplified URL as identifier: https://example.com/path → example.com
        baseline_name = page_url.split("://")[-1].split("/")[0]
        baseline_key = f"{tenant_id}/{project_id}/baselines/{baseline_name}.png"
        baseline_local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        
        baseline_exists = False
        try:
            # Try to download baseline
            s3.download_file(ARTIFACTS_BUCKET, baseline_key, baseline_local)
            baseline_exists = True
            print(f"[visual-analysis-agent] baseline found, comparing...")
        except ClientError as e:
            # Check if it's a "not found" error (first run scenario)
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['404', 'NoSuchKey', 'Not Found']:
                print(f"[visual-analysis-agent] no baseline found, creating one...")
                try:
                    s3.upload_file(current_local, ARTIFACTS_BUCKET, baseline_key)
                    print(f"[visual-analysis-agent] baseline created at {baseline_key}")
                except Exception as upload_err:
                    print(f"[visual-analysis-agent] could not save baseline: {upload_err}")
                baseline_exists = False
            else:
                print(f"[visual-analysis-agent] error accessing baseline: {e}")
                baseline_exists = False
        except Exception as e:
            print(f"[visual-analysis-agent] unexpected error: {e}")
            baseline_exists = False
        
        # If no baseline, skip comparison (first run)
        if not baseline_exists:
            print(f"[visual-analysis-agent] first run for this URL, no comparison needed")
            continue
        
        # Compare images
        try:
            current_img = cv2.imread(current_local)
            baseline_img = cv2.imread(baseline_local)
            
            if current_img is None:
                print(f"[visual-analysis-agent] could not read current image")
                continue
            
            if baseline_img is None:
                print(f"[visual-analysis-agent] could not read baseline image")
                continue
            
            # Resize if dimensions differ
            if current_img.shape != baseline_img.shape:
                print(f"[visual-analysis-agent] resizing images to match baseline")
                current_img = cv2.resize(current_img, (baseline_img.shape[1], baseline_img.shape[0]))
            
            # Calculate SSIM
            gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
            gray_baseline = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
            
            score, diff = ssim(gray_current, gray_baseline, full=True)
            difference_percent = round((1 - score) * 100, 2)
            
            print(f"[visual-analysis-agent] SSIM score: {score:.4f} ({difference_percent}% different)")
            
            # Determine verdict
            if difference_percent < 1.0:
                verdict = "no_change"
                needs_review = False
                severity = "info"
            elif difference_percent < 5.0:
                verdict = "minor_change"
                needs_review = True
                severity = "info"
            else:
                verdict = "significant_change"
                needs_review = True
                severity = "warning"
            
            # Optional AI analysis for significant changes
            ai_analysis = None
            if needs_review and difference_percent > 2.0:
                try:
                    prompt = f"""
URL: {page_url}
Visual Difference: {difference_percent}%

Is this a real bug or expected design change?
(Respond with classification: real_bug / expected_update / environmental)
"""
                    
                    completion = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": VISUAL_ANALYSIS_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=100,
                    )
                    ai_analysis = completion.choices[0].message.content
                    print(f"[visual-analysis-agent] AI analysis: {ai_analysis[:80]}...")
                
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
        
        # Cleanup temp files
        try:
            os.remove(current_local)
            if baseline_exists and os.path.exists(baseline_local):
                os.remove(baseline_local)
        except:
            pass
    
    print(f"[visual-analysis-agent] found {len([d for d in visual_diffs if d['needs_review']])} visual differences requiring review")
    
    return {**state, "visual_diffs": visual_diffs}