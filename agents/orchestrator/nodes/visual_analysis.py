import tempfile
import cv2
from skimage.metrics import structural_similarity as ssim
from botocore.exceptions import ClientError
from clients import s3, ARTIFACTS_BUCKET


def visual_analysis_agent_node(state: dict) -> dict:
    print(f"[visual-analysis-agent] comparing screenshots for run {state['run_id']}")

    tenant_id, project_id = state["tenant_id"], state["project_id"]
    visual_diffs = []

    for page_result in state["execution_results"]:
        current_local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        s3.download_file(ARTIFACTS_BUCKET, page_result["screenshot_s3_path"], current_local)

        baseline_key = f"{tenant_id}/{project_id}/baselines/homepage.png"
        baseline_local = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name

        try:
            s3.download_file(ARTIFACTS_BUCKET, baseline_key, baseline_local)
        except ClientError:
            # No baseline exists yet - this run's screenshot BECOMES the baseline.
            # This is expected and normal on the very first run of a page.
            print("[visual-analysis-agent] no baseline found, saving this screenshot as the new baseline")
            s3.upload_file(current_local, ARTIFACTS_BUCKET, baseline_key)
            continue

        current_img = cv2.imread(current_local)
        baseline_img = cv2.imread(baseline_local)
        if current_img.shape != baseline_img.shape:
            current_img = cv2.resize(current_img, (baseline_img.shape[1], baseline_img.shape[0]))

        gray_current = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        gray_baseline = cv2.cvtColor(baseline_img, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_current, gray_baseline, full=True)
        difference_percent = round((1 - score) * 100, 2)

        # Pixel diff gates the (expensive, hallucination-prone) LLM call - only escalate
        # to semantic judgement if the difference is big enough to plausibly be real.
        verdict = "needs_review" if difference_percent > 1.0 else "expected_change"

        print(f"[visual-analysis-agent] difference={difference_percent}% verdict={verdict}")
        visual_diffs.append({
            "page": page_result["pageUrl"],
            "difference_percent": difference_percent,
            "verdict": verdict,
        })

    return {**state, "visual_diffs": visual_diffs}
