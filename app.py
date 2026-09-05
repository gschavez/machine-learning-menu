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

@app.route("/usecase1")
def usecase1():
    return render_template("usecase1.html")

@app.route("/usecase2")
def usecase2():
    return render_template("usecase2.html")

@app.route("/usecase3")
def usecase3():
    return render_template("usecase3.html")

@app.route("/usecase4")
def usecase4():
    return render_template("usecase4.html")


if __name__ == "__main__":
    app.run(debug=True)