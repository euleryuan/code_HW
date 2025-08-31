import re
import json
import math

aes_weights = {
    "Bad": 0.13330,
    "Poor": 0.00911,
    "Fair": 0.00200,
    "Good": 0.03361,
    "Excellent": 0.08797
}
ads_weights = {
    "Bad": 0.04102,
    "Poor": 0.01277,
    "Fair": 0.00192,
    "Good": 0.04947,
    "Excellent": 0.89482
}

reward_weight = {
    "single": 0.058,
    "multi": 0.382,
    "quality_score": 0.28,
    "ads": 0.28,
    "aes": 0.28,
}

def vision_reasoner_format_reward(predict_str: str) -> float:
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    return 1.0 if match else 0.0

def vision_reasoner_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    accuracy_reward = 0.0
    try:
        gt = json.loads(ground_truth)
        answer_type = gt.get("answer_type", "").lower()
        answer = gt.get("answer")
        # 抽取 <answer>…</answer> JSON
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return 0.0
        data = json.loads(json_match.group(1))
        if not isinstance(data, list) or len(data) == 0:
            return 0.0
        obj = data[0]

        # ---------- single ----------
        if answer_type == "single":
            y = 1 if str(answer).lower() == "yes" else 0
            pred = str(obj.get("answer", "")).lower()
            p = obj.get("confidence", None)
            if p is None:
                p = 1.0 if pred == "yes" else 0.0
            accuracy_reward = 1.0 - 2.0 * abs(float(p) - y)
            accuracy_reward *= reward_weight["single"]

        # ---------- multi ----------
        elif answer_type == "multi":
            gold = set([s.lower() for s in answer])
            pred = set([s.lower() for s in obj.get("answer", [])])
            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)
            if tp + fp + fn == 0:
                accuracy_reward = 1.0
            else:
                beta = 0.7
                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-6)
                jaccard = tp / (tp + fp + fn + 1e-6)
                accuracy_reward = 0.5 * f_beta + 0.5 * jaccard
            accuracy_reward *= reward_weight["multi"]

        # ---------- quality_score ----------
        elif answer_type == "quality_score":
            try:
                s = float(answer)
                shat = float(obj.get("answer", 0))
                sigma = 6.0
                accuracy_reward = math.exp(-((s - shat) ** 2) / (2 * sigma ** 2))
            except Exception:
                accuracy_reward = 0.0
            accuracy_reward *= reward_weight["quality_score"]

        # ---------- ads / aes 分类任务 ----------
        elif answer_type in ["ads", "aes"]:
            pred = str(obj.get("answer", "")).capitalize()
            truth = str(answer).capitalize()
            if answer_type == "aes":
                class_weights = aes_weights
            else:
                class_weights = ads_weights
            if pred == truth:
                accuracy_reward = class_weights.get(truth, 0.0)
            else:
                accuracy_reward = 0.0
            accuracy_reward *= reward_weight[answer_type]

    except Exception:
        pass

    return accuracy_reward

def vision_reasoner_non_repeat_reward(predict_str: str) -> float:
    non_repeat_reward = 1.0
    try:
        sentences = predict_str.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        seen = set()
        repeats = 0
        for sentence in sentences:
            if sentence in seen:
                repeats += 1
            if repeats >= 2:
                non_repeat_reward = 0.0
                break
            seen.add(sentence)
    except Exception:
        pass
    return non_repeat_reward

def vision_reasoner_compute_score(predict_str: str, ground_truth: str) -> float:
    format_reward = vision_reasoner_format_reward(predict_str)
    accuracy_reward = vision_reasoner_accuracy_reward(predict_str, ground_truth)
    non_repeat_reward = vision_reasoner_non_repeat_reward(predict_str)
    reward = format_reward + accuracy_reward + non_repeat_reward
    return reward

if __name__ == "__main__":
    # ---------- aes 分类 ----------
    predict_str_aes = """<think>Colors are well balanced.</think><answer>[{"answer":"Excellent"}]</answer>"""
    ground_truth_aes = json.dumps({
        "answer_type": "aes",
        "answer": "Excellent"
    })

    predict_str_aes_bad = """<think>Blurry and unclear.</think><answer>[{"answer":"Bad"}]</answer>"""
    ground_truth_aes_bad = json.dumps({
        "answer_type": "aes",
        "answer": "Bad"
    })

    # ---------- ads 分类 ----------
    predict_str_ads_fair = """<think>The ad delivers its message somewhat.</think><answer>[{"answer":"Fair"}]</answer>"""
    ground_truth_ads_fair = json.dumps({
        "answer_type": "ads",
        "answer": "Fair"
    })

    predict_str_ads_exc = """<think>The ad is very persuasive and clear.</think><answer>[{"answer":"Excellent"}]</answer>"""
    ground_truth_ads_exc = json.dumps({
        "answer_type": "ads",
        "answer": "Excellent"
    })

    # ---------- single ----------
    predict_str_single = """<think>Definitely a healthy tooth.</think><answer>[{"answer":"Yes", "confidence":0.95}]</answer>"""
    ground_truth_single = json.dumps({
        "answer_type": "single",
        "answer": "Yes"
    })

    # ---------- multi ----------
    predict_str_multi = """<think>Logo and slogan both present.</think><answer>[{"answer":["Logo", "Slogan"]}]</answer>"""
    ground_truth_multi = json.dumps({
        "answer_type": "multi",
        "answer": ["Logo", "Slogan"]
    })

    # ---------- quality_score ----------
    predict_str_qscore = """<think>Image is noisy.</think><answer>[{"answer":65.0}]</answer>"""
    ground_truth_qscore = json.dumps({
        "answer_type": "quality_score",
        "answer": 78
    })

    cases = [
        ("aes_Excellent", predict_str_aes, ground_truth_aes),
        ("aes_Bad", predict_str_aes_bad, ground_truth_aes_bad),
        ("ads_Fair", predict_str_ads_fair, ground_truth_ads_fair),
        ("ads_Excellent", predict_str_ads_exc, ground_truth_ads_exc),
        ("single", predict_str_single, ground_truth_single),
        ("multi", predict_str_multi, ground_truth_multi),
        ("quality_score", predict_str_qscore, ground_truth_qscore),
    ]

    for name, pred, gt in cases:
        fr = vision_reasoner_format_reward(pred)
        ar = vision_reasoner_accuracy_reward(pred, gt)
        nr = vision_reasoner_non_repeat_reward(pred)
        total = fr + ar + nr
        print(f"{name}: format={fr:.2f}, accuracy={ar:.5f}, non-repeat={nr:.2f}  ==> total={total:.5f}")
