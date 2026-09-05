from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/concept")
def concept():
    return render_template("concept.html")

@app.route("/types")
def types():
    return render_template("types.html")

@app.route("/linearRegression/concepts")
def linear_regression_concepts():
    return render_template("linear-regression-concepts.html")

@app.route("/use-cases")
def use_cases():
    return render_template("use-cases.html")


if __name__ == "__main__":
    app.run(debug=True)