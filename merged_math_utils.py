import re
import math
import logging
from math_utils import is_correct as is_correct_hapo

logger = logging.getLogger(__name__)

# ============================================================
# SHARED HELPERS (From evaluate_s1k.py)
# ============================================================

def extract_boxed_answer(text: str):
    """Extract content from \boxed{...} handling nested braces."""
    if not text: return None
    idx = text.rfind("\\boxed{")
    if idx < 0:
        if "\\boxed " in text:
            return text.split("\\boxed ")[-1].split("$")[0].strip()
        return None
    i = idx + 7
    brace_count = 1
    while i < len(text) and brace_count > 0:
        if text[i] == '{': brace_count += 1
        elif text[i] == '}': brace_count -= 1
        i += 1
    if brace_count == 0: return text[idx + 7:i - 1].strip()
    return None

def extract_model_answer(response: str) -> str:
    """Extract the final answer from model response using S1K heuristics."""
    if not response: return ""
    boxed = extract_boxed_answer(response)
    if boxed: return boxed
    match = re.search(r'(?i)the final answer is[:\s]*([^\n.]+)', response)
    if match: return match.group(1).strip()
    match = re.search(r'(?i)\banswer[:\s]+([^\n]+)', response)
    if match: return match.group(1).strip()
    lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
    return lines[-1] if lines else ""

# ============================================================
# S1K SPECIFIC CHECKERS
# ============================================================

def is_correct_s1k_crossword(response: str, ground_truth: str) -> bool:
    # Extract GT from "### Answer: WORD" format if present
    match = re.search(r'###\s*Answer:\s*(\w+)', ground_truth, re.IGNORECASE)
    gold_answer = match.group(1).upper() if match else ground_truth.strip().upper()
    
    model_answer = extract_model_answer(response).upper()
    
    model_clean = re.sub(r'[^\w\s]', '', model_answer).strip()
    gold_clean = re.sub(r'[^\w\s]', '', gold_answer).strip()
    
    return model_clean == gold_clean or (gold_clean in model_clean and len(model_clean) < len(gold_clean) + 10)

def is_correct_s1k_science(response: str, ground_truth: str) -> bool:
    def normalize_science(text):
        text = text.strip().upper()
        text = re.sub(r'^(THE\s+ANSWER\s+IS|ANSWER|OPTION)[:\s]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\(([A-D])\)$', r'\1', text)
        return text.strip()

    model_ans = extract_model_answer(response)
    # S1K Science GT often has metadata/explanations, but we assume the training script
    # passes the clean answer or we rely on heuristic extraction
    
    # Try to extract "Answer: A" pattern from GT if it's a long solution string
    match = re.search(r'(?i)correct answer is[:\s]*([^\n.,]+)', ground_truth)
    gold_ans = match.group(1).strip() if match else ground_truth
    
    model_norm = normalize_science(model_ans)
    gold_norm = normalize_science(gold_ans)
    
    if model_norm == gold_norm: return True
    
    # Heuristic for Single Letter Answers (A, B, C, D)
    if len(gold_norm) <= 3 and gold_norm.isalpha():
         if len(gold_norm) == 1 and gold_norm in model_norm[:5]: return True
         
    return False

def is_correct_s1k_math(response: str, ground_truth: str) -> bool:
    model_answer = extract_model_answer(response)
    gold_answer = ground_truth.strip()
    
    # Try float comparison
    try:
        # Remove common non-numeric chars for float conversion
        m_clean = re.sub(r'[^\d.\-]', '', model_answer)
        g_clean = re.sub(r'[^\d.\-]', '', gold_answer)
        if m_clean and g_clean:
            if math.isclose(float(m_clean), float(g_clean), rel_tol=1e-5):
                return True
    except ValueError:
        pass
        
    # Fallback to string normalization
    def norm(s): return s.lower().replace(' ', '').replace('\\', '')
    return norm(model_answer) == norm(gold_answer)

# ============================================================
# RECLOR CHECKER
# ============================================================

def is_correct_reclor(response: str, ground_truth: str) -> bool:
    gt = ground_truth.strip().upper()
    boxed = extract_boxed_answer(response)
    if boxed:
        pred = re.sub(r'^(OPTION|ANSWER)\s*', '', boxed.upper()).strip("()[]")
        if pred == gt: return True
        
    lines = response.strip().split('\n')
    if lines:
        last_line = lines[-1].upper()
        match = re.search(r'(?:ANSWER|OPTION)?\s*[:=]?\s*([A-E])\b', last_line)
        if match and match.group(1) == gt: return True
    return False

# ============================================================
# MAIN ROUTER
# ============================================================

def is_correct(response: str, ground_truth: str, use_math_verify: bool = False) -> bool:
    """
    Universal Router.
    Expected GT format: "SOURCE[:TYPE]:::REAL_ANSWER"
    """
    if ":::" not in ground_truth:
        return is_correct_hapo(response, ground_truth, True)
        
    tag, real_gt = ground_truth.split(":::", 1)
    
    # Parse tag (e.g. "S1K:science" or "RECLOR")
    if ":" in tag:
        source, subtype = tag.split(":", 1)
        subtype = subtype.lower()
    else:
        source = tag
        subtype = "math" # default
        
    if source == "RECLOR":
        return is_correct_reclor(response, real_gt)
        
    elif source == "S1K":
        if subtype == "crossword":
            return is_correct_s1k_crossword(response, real_gt)
        elif subtype == "science":
            return is_correct_s1k_science(response, real_gt)
        else:
            return is_correct_s1k_math(response, real_gt)
            
    # Default
    return is_correct_hapo(response, real_gt, True)