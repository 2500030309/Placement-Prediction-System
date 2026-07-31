import sys
import os
# Add project root to Python path so 'src' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template
from src.data.load_data import load_data , get_summary


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
        first_rows = df.head().to_html(index=False)
    )

if __name__ == "__main__":
    app.run(debug=True)

