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

@app.route("/use-cases")
def use_cases():
    return render_template("use-cases-index.html")


@app.route("/useCases/usecase1")
def usecase1():
    return render_template("usecase1.html")

@app.route("/useCases/usecase2")
def usecase2():
    return render_template("usecase2.html")

@app.route("/useCases/usecase3")
def usecase3():
    return render_template("usecase3.html")

<<<<<<< HEAD
@app.route("/linearRegression/application", methods = ["GET", "POST"])
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
                prediction = taxi_model.calculate_cost(distance)
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
=======
@app.route("/useCases/usecase4")
def usecase4():
    return render_template("usecase4.html")
>>>>>>> 531a526 (use case 4)


if __name__ == "__main__":
    app.run(debug=True)