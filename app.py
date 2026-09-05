from flask import Flask, render_template, request
import taxi_model

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

@app.route("/useCases/usecase2")
def usecase2():
    return render_template("usecase2.html")

@app.route("/linearRegression/application", methods = ["GET, POST"])
def linear_regression_application():
    prediction = None
    error = None
    input_distance = None

    if request.method == "POST":
        value = request.form.get("distance", "").strip()
        input_distance = value

        if value == "":
            error = "Please enter a distance value."
        else:
            try:
                distance = float(value)
                prediction = taxi_model_calculate_cost(distance)
            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "linear-regression-application.html",
        num_records = taxi_model.get_num_records(),
        plot_url = taxi_model.generate_plot(),
        prediction = prediction,
        error = error,
        input_distance = input_distance
    )


if __name__ == "__main__":
    app.run(debug=True)