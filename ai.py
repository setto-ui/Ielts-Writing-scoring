# ai.py
import language_tool_python
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
from datetime import datetime
import os
from collections import Counter
import math

# ------------ Init ------------
tool = language_tool_python.LanguageTool('en-US')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Small AWL (sample). You can expand this list if desired.
AWL = set([
    "analysis","approach","area","assess","assume","authority","available","benefit","concept",
    "consistent","constitute","context","contract","create","data","define","derive","distribute",
    "economy","environment","establish","estimate","evidence","export","factor","financial","function",
    "identify","income","indicate","individual","interpret","involve","issue","labour","legislate",
    "legal","major","method","occur","percent","period","policy","principle","process","require",
    "research","respond","role","section","sector","significant","similar","source","structure",
    "theory","variable","volume","promote","advantage","disadvantage","crucial"
])

# ------------ Helpers ------------
def tokenize_words(text):
    return re.findall(r"\b\w+\b", (text or "").lower())

def split_sentences(text):
    s = re.split(r'[.!?]+', (text or ""))
    return [seg.strip() for seg in s if seg.strip()]

def get_embeddings(text):
    if not text:
        text = " "
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return emb

def semantic_similarity(a, b):
    a = a or ""
    b = b or ""
    try:
        emb_a = get_embeddings(a if a.strip() else " ")
        emb_b = get_embeddings(b if b.strip() else " ")
        sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-9))
        sim = (sim + 1.0) / 2.0
        return sim
    except Exception:
        wa = set(tokenize_words(a))
        wb = set(tokenize_words(b))
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

def jaccard(a, b):
    sa = set(tokenize_words(a))
    sb = set(tokenize_words(b))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def first_sentences_of_paragraphs(essay):
    paras = [p.strip() for p in (essay or "").split('\n\n') if p.strip()]
    topic_sents = []
    for p in paras:
        sents = split_sentences(p)
        if sents:
            topic_sents.append(sents[0])
    return topic_sents

def detect_tense_consistency(sentences):
    future_markers = ["will ", "shall ", "'ll "]
    present_markers = [" is ", " are ", " am ", " do ", " does ", " have ", " has "]
    past_markers = [" was ", " were ", " did ", " had "]
    f = p = pa = 0
    for s in sentences:
        ls = " " + s.lower() + " "
        if any(m in ls for m in future_markers):
            f += 1
        if any(m in ls for m in present_markers):
            p += 1
        if any(m in ls for m in past_markers) or re.search(r'\b\w+ed\b', s):
            pa += 1
    total = max(1, f + p + pa)
    dominant = max(("future", f), ("present", p), ("past", pa), key=lambda x: x[1])
    dominant_ratio = dominant[1] / total
    inconsistent = dominant_ratio < 0.6
    return not inconsistent, dominant[0], dominant_ratio

# ------------ Scoring logic ------------
def check_off_topic(essay, prompt):
    sim = semantic_similarity(prompt or "", essay or "")
    is_off = sim < 0.4
    return is_off, sim

def evaluate_ielts_writing(essay, prompt):
    essay = essay or ""
    prompt = prompt or ""

    words = tokenize_words(essay)
    word_count = len(words)
    sentences = split_sentences(essay)
    num_sentences = len(sentences)
    paragraphs = [p for p in essay.split('\n\n') if p.strip()]
    num_paragraphs = len(paragraphs) if paragraphs else 1
    topic_sents = first_sentences_of_paragraphs(essay)
    num_main_ideas = len(topic_sents)

    matches = tool.check(essay)
    grammar_errors = len(matches)

    off_topic, similarity = check_off_topic(essay, prompt)

    # Task Achievement
    ta = 4.0
    if off_topic:
        ta = 3.0
    else:
        if similarity >= 0.85:
            ta += 2.0
        elif similarity >= 0.70:
            ta += 1.0
        elif similarity >= 0.55:
            ta += 0.5
    if num_main_ideas < 2:
        ta -= 1.5
    elif num_main_ideas >= 3:
        ta += 0.5
    if word_count < 50:
        ta = min(ta, 3.0)
    elif word_count < 150:
        ta -= 1.0
    elif 200 <= word_count < 250:
        ta += 0.2
    elif word_count >= 250:
        ta += 1.0
    if re.search(r'\b(example|for instance|for example|such as)\b', essay.lower()):
        ta += 0.5
    ta = max(0.0, min(9.0, round(ta, 1)))

    # Coherence & Cohesion
    cc = 4.0
    if num_paragraphs >= 4:
        cc += 1.5
    elif num_paragraphs >= 3:
        cc += 0.8
    cohesive_list = ['however','moreover','furthermore','in addition','on the other hand',
                     'for example','such as','therefore','thus','consequently','nevertheless',
                     'in conclusion','to conclude','to sum up','firstly','secondly','finally']
    used_cohesive = set(w for w in cohesive_list if w in essay.lower())
    cc += 0.25 * len(used_cohesive)
    if len(paragraphs) >= 2:
        jaccards = []
        for i in range(len(paragraphs)-1):
            jaccards.append(jaccard(paragraphs[i], paragraphs[i+1]))
        avg_j = float(np.mean(jaccards)) if jaccards else 0.0
        if avg_j > 0.6:
            cc -= 1.0
        elif avg_j > 0.4:
            cc -= 0.4
    if num_sentences > 3:
        try:
            sent_embs = [get_embeddings(s) for s in sentences]
            sims = []
            for i in range(len(sent_embs)-1):
                sims.append(float(np.dot(sent_embs[i], sent_embs[i+1])/(np.linalg.norm(sent_embs[i])*np.linalg.norm(sent_embs[i+1])+1e-9)))
            mean_s = np.mean(sims) if sims else 0.0
            mean_s = (mean_s + 1.0)/2.0
            if mean_s > 0.75:
                cc += 0.6
            elif mean_s > 0.6:
                cc += 0.2
        except Exception:
            pass
    if any(len(p.split()) > 120 for p in paragraphs):
        cc -= 0.5
    cc = max(0.0, min(9.0, round(cc,1)))

    # Lexical Resource
    lr = 4.5
    total_words = max(1, word_count)
    unique = len(set(words))
    ttr = unique / total_words
    if ttr < 0.35:
        lr -= 1.5
    elif ttr < 0.45:
        lr -= 0.5
    else:
        lr += 0.5
    awl_matches = sum(1 for w in set(words) if w in AWL)
    if awl_matches >= 3:
        lr += 0.5
    if awl_matches >= 6:
        lr += 0.5
    freq = Counter(words)
    max_freq = max(freq.values()) if freq else 0
    if max_freq > total_words * 0.2:
        lr -= 0.5
    colloc_errors = 0
    for m in matches:
        msg = str(m).lower()
        if "collocation" in msg or "word choice" in msg or "choice" in msg:
            colloc_errors += 1
    if colloc_errors > 0:
        lr -= 0.5
    lr = max(0.0, min(9.0, round(lr,1)))

    # Grammatical Range & Accuracy
    gra = 4.5
    err_rate = grammar_errors / (total_words / 100.0)
    if err_rate > 10:
        gra -= 1.5
    elif err_rate > 6:
        gra -= 0.7
    elif err_rate > 3:
        gra -= 0.2
    else:
        gra += 0.7
    complex_sentences = sum(1 for s in sentences if len(s.split()) > 15)
    complex_ratio = complex_sentences / max(1, num_sentences)
    if complex_ratio >= 0.3:
        gra += 0.5
    elif complex_ratio < 0.1:
        gra -= 0.3
    tense_consistent, dominant_tense, dom_ratio = detect_tense_consistency(sentences)
    if not tense_consistent:
        gra -= 0.5
    gra = max(0.0, min(9.0, round(gra,1)))

    # Final band và làm tròn toàn bộ sub-scores về .0 hoặc .5
    raw_band = (ta + cc + lr + gra) / 4.0

    def round_band(x):
        return round(x * 2) / 2.0

    ta = round_band(ta)
    cc = round_band(cc)
    lr = round_band(lr)
    gra = round_band(gra)
    band_score = round_band(raw_band)

    grammar_matches_serializable = []
    for m in matches:
        grammar_matches_serializable.append({
            "message": m.message,
            "replacements": m.replacements,
            "offset": int(m.offset),
            "errorLength": int(m.errorLength),
            "context": m.context if hasattr(m, "context") else None
        })

    return {
        'Task Achievement': float(ta),
        'Coherence and Cohesion': float(cc),
        'Lexical Resource': float(lr),
        'Grammatical Range and Accuracy': float(gra),
        'Band Score': float(band_score),
        'Grammar Errors': grammar_matches_serializable,
        'Off Topic': bool(off_topic),
        'Similarity Score': float(round(similarity, 3)),
        'TTR': float(round(ttr, 3)),
        'AWL_matches': int(awl_matches),
        'error_rate_per_100_words': float(round(err_rate,2)),
        'dominant_tense': dominant_tense if 'dominant_tense' in locals() else dominant_tense,
        'tense_dom_ratio': float(round(dom_ratio,3)) if 'dom_ratio' in locals() else None
    }

# ------------ Correction function ------------
def correct_grammar(essay):
    essay = essay or ""
    matches = tool.check(essay)
    corrected_essay = essay
    for match in reversed(matches):
        if match.replacements:
            start = match.offset
            end = match.offset + match.errorLength
            try:
                corrected_essay = corrected_essay[:start] + match.replacements[0] + corrected_essay[end:]
            except Exception:
                pass
    return corrected_essay

def generate_model_answer(prompt, essay=""):
    prompt = (prompt or "").strip()
    essay = (essay or "").strip()

    paragraphs = [p for p in essay.split('\n\n') if p.strip()] if essay else []
    main_ideas = []
    for para in paragraphs:
        sentences = [s.strip() for s in para.split('.') if s.strip()]
        if sentences:
            try:
                sentence_embeddings = [get_embeddings(s) for s in sentences]
                para_embedding = get_embeddings(para)
                similarities = [cosine_similarity([para_embedding], [s_emb])[0][0] for s_emb in sentence_embeddings]
                main_idea = sentences[similarities.index(max(similarities))]
            except Exception:
                main_idea = sentences[0]
            main_ideas.append(main_idea)

    is_discuss_both_views = "discuss both views" in prompt.lower()
    is_opinion = "your opinion" in prompt.lower() or "to what extent" in prompt.lower()

    model_answer = []
    topic = prompt.split('.')[0].strip().lower() if prompt else "this topic"
    intro = f"The issue of {topic} has generated considerable debate in recent years. "
    if is_discuss_both_views:
        intro += "While some individuals advocate for one perspective, others support a contrasting viewpoint. "
    if is_opinion:
        intro += "In this essay, I will examine both sides of the argument and present my own perspective."
    else:
        intro += "This essay will explore both perspectives in detail."
    model_answer.append(intro.strip())

    if len(main_ideas) > 0:
        body1 = f"To begin with, one argument is that {main_ideas[0].lower()}. "
        body1 += "This highlights one side of the issue and shows how it can affect relevant areas."
        model_answer.append(body1.strip())
    else:
        model_answer.append("To begin with, proponents argue that this approach brings several benefits, such as promoting responsibility and practical skills.")

    if len(main_ideas) > 1:
        body2 = f"Conversely, another viewpoint is that {main_ideas[1].lower()}. "
        body2 += "This perspective underlines potential drawbacks or challenges which merit consideration."
        model_answer.append(body2.strip())
    else:
        model_answer.append("On the other hand, critics contend that it may take time away from education and leisure, which are also important for children's development.")

    conclusion = "In conclusion, both viewpoints offer valuable insights; a balanced approach is recommended."
    model_answer.append(conclusion.strip())

    return '\n\n'.join(model_answer)

# ------------ API wrapper (returns JSON-friendly keys) ------------
def evaluate_for_api(essay, prompt):
    res = evaluate_ielts_writing(essay, prompt)

    def round_band(x):
        return round(x * 2) / 2 if x is not None else None

    return {
        "grammar_score": round_band(res.get("Grammatical Range and Accuracy")),
        "task_response": round_band(res.get("Task Achievement")),
        "coherence": round_band(res.get("Coherence and Cohesion")),
        "lexical_resource": round_band(res.get("Lexical Resource")),
        "overall_band": round_band(res.get("Band Score")),
        "corrected_essay": correct_grammar(essay),
        "model_answer": generate_model_answer(prompt, essay),
        "grammar_errors": res.get("Grammar Errors", []),
        "similarity": res.get("Similarity Score"),
        "off_topic": res.get("Off Topic"),
        "ttr": res.get("TTR"),
        "awl_matches": res.get("AWL_matches"),
        "error_rate_per_100_words": res.get("error_rate_per_100_words"),
        "dominant_tense": res.get("dominant_tense"),
        "tense_dom_ratio": res.get("tense_dom_ratio")
    }

# ------------ Main interactive flow ------------
if __name__ == "__main__":
    print("Nhập đề bài IELTS Writing Task 2 (nhấn Enter khi xong):")
    prompt = input()

    print("\nNhập bài viết IELTS của bạn (dùng Enter hai lần để tách đoạn, nhập 'DONE' để kết thúc):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == 'DONE':
            break
        lines.append(line)
    sample_essay = '\n'.join(lines).strip()

    api_res = evaluate_for_api(sample_essay, prompt)
    internal = evaluate_ielts_writing(sample_essay, prompt)

    print("\n=== API-friendly result ===")
    for k, v in api_res.items():
        print(f"{k}: {v}")

    print("\n=== Detailed internal metrics ===")
    for k, v in internal.items():
        print(f"{k}: {v}")

    output_file = 'results.csv'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    grammar_errors_str = "; ".join([f"{m['message']} -> {m.get('replacements')}" for m in internal.get('Grammar Errors', [])])

    data = {
        'Timestamp': timestamp,
        'Prompt': prompt,
        'Original Essay': sample_essay,
        'Corrected Essay': api_res['corrected_essay'],
        'Task Achievement': internal.get('Task Achievement'),
        'Coherence and Cohesion': internal.get('Coherence and Cohesion'),
        'Lexical Resource': internal.get('Lexical Resource'),
        'Grammatical Range and Accuracy': internal.get('Grammatical Range and Accuracy'),
        'Band Score': internal.get('Band Score'),
        'Off Topic': 'Lạc đề' if internal.get('Off Topic') else 'Không lạc đề',
        'Similarity Score': internal.get('Similarity Score'),
        'Grammar Errors': grammar_errors_str,
        'Model Answer': api_res['model_answer']
    }

    df = pd.DataFrame([data])
    if os.path.exists(output_file):
        try:
            df.to_csv(output_file, mode='a', header=False, index=False)
        except PermissionError:
            print(f"Warning: cannot write to {output_file} (permission denied). Skipping save.")
    else:
        df.to_csv(output_file, mode='w', header=True, index=False)

    print(f"\nKết quả đã được xử lý. Kết quả API-friendly và metrics chi tiết đã in ra. Kết quả (nếu có) đã lưu vào: {output_file}")
