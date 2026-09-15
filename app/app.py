import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from src.data.load_data import load_data, get_summary
from src.MLmodels.predict_service import predict_placement_profile


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)

    return render_template(
        "dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False, classes="dataset-table")
    )


@app.route("/eda")
def eda():
    df = load_data()
    summary = get_summary(df)
    return render_template("eda.html", summary=summary)


@app.route("/models")
def models_view():
    return render_template("models.html")


@app.route("/clustering")
def clustering_view():
    return render_template("clustering.html")


@app.route("/comparison")
def comparison_view():
    return render_template("comparison.html")


@app.route("/api/predict_top_model", methods=["POST"])
def api_predict_top_model():
    try:
        data = request.get_json(force=True) if request.is_json else request.form.to_dict()
        result = predict_placement_profile(data)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)


