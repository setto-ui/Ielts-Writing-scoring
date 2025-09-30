from flask import Flask, request
from ai import evaluate_for_api
from collections import OrderedDict
import json

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    essay = data.get("essay", "")
    prompt = data.get("prompt", "")

    res = evaluate_for_api(essay, prompt)

    # Sắp xếp thủ công theo thứ tự bạn muốn
    ordered = OrderedDict([
        ("task_response", res["task_response"]),
        ("coherence", res["coherence"]),
        ("lexical_resource", res["lexical_resource"]),
        ("grammar_score", res["grammar_score"]),
        ("overall_band", res["overall_band"]),
        ("corrected_essay", res["corrected_essay"]),
        ("model_answer", res["model_answer"]),
        ("grammar_errors", res["grammar_errors"]),
        ("similarity", res["similarity"]),
        ("off_topic", res["off_topic"]),
        ("ttr", res["ttr"]),
        ("awl_matches", res["awl_matches"]),
        ("error_rate_per_100_words", res["error_rate_per_100_words"]),
        ("dominant_tense", res["dominant_tense"]),
        ("tense_dom_ratio", res["tense_dom_ratio"])
    ])

    return app.response_class(
        response=json.dumps(ordered, ensure_ascii=False, indent=2, sort_keys=False),
        mimetype="application/json"
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
