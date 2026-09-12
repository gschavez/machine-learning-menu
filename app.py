from flask import Flask, render_template, request
import taxi_model
import tennis_model

app = Flask(__name__)


# =========================
# Activity 1 - General ML
# =========================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/concept")
def concept():
    return render_template("concept.html")


@app.route("/types")
def types():
    return render_template("types.html")


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


@app.route("/useCases/usecase4")
def usecase4():
    return render_template("usecase4.html")


# =========================
# Activity 1 - Linear Regression
# =========================

@app.route("/linearRegression/concepts")
def linear_regression_concepts():
    return render_template("linear-regression-concepts.html")


@app.route("/linearRegression/application", methods=["GET", "POST"])
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
        num_records=taxi_model.get_num_records(),
        plot_url=taxi_model.generate_plot(),
        prediction=prediction,
        error=error,
        input_distance=input_distance
    )


# =========================
# Activity 2 - Logistic Regression
# =========================

@app.route("/logisticRegression/concepts")
def logistic_regression_concepts():
    return render_template("logistic-regression-concepts.html")


@app.route("/logisticRegression/application", methods=["GET", "POST"])
def logistic_regression_application():
    prediction = None
    error = None
    input_ranking_difference = None

    if request.method == "POST":
        value = request.form.get("ranking_difference", "").strip()
        input_ranking_difference = value

        if value == "":
            error = "Please enter a ranking difference."
        else:
            try:
                ranking_difference = float(value)
                prediction = tennis_model.predict_logistic(ranking_difference)
            except ValueError:
                error = "Please enter a valid numeric value."

    return render_template(
        "logistic-regression-application.html",
        num_records=tennis_model.get_num_records(),
        training_records=tennis_model.get_training_records(),
        testing_records=tennis_model.get_testing_records(),
        prediction=prediction,
        error=error,
        input_ranking_difference=input_ranking_difference
    )


@app.route("/logisticRegression/evaluation")
def logistic_regression_evaluation():
    return render_template(
        "logistic-regression-evaluation.html",
        metrics=tennis_model.get_logistic_metrics(),
        confusion_matrix=tennis_model.get_logistic_confusion_matrix()
    )


# =========================
# Activity 2 - Decision Tree
# =========================

@app.route("/decisionTree/concepts")
def decision_tree_concepts():
    return render_template("decision-tree-concepts.html")


@app.route("/decisionTree/application", methods=["GET", "POST"])
def decision_tree_application():
    prediction = None
    error = None

    input_player_a_ranking = None
    input_player_b_ranking = None
    input_player_a_recent_win_rate = None
    input_player_b_recent_win_rate = None

    if request.method == "POST":
        input_player_a_ranking = request.form.get(
            "player_a_ranking", ""
        ).strip()

        input_player_b_ranking = request.form.get(
            "player_b_ranking", ""
        ).strip()

        input_player_a_recent_win_rate = request.form.get(
            "player_a_recent_win_rate", ""
        ).strip()

        input_player_b_recent_win_rate = request.form.get(
            "player_b_recent_win_rate", ""
        ).strip()

        if (
            input_player_a_ranking == ""
            or input_player_b_ranking == ""
            or input_player_a_recent_win_rate == ""
            or input_player_b_recent_win_rate == ""
        ):
            error = "Please complete all fields."
        else:
            try:
                player_a_ranking = float(input_player_a_ranking)
                player_b_ranking = float(input_player_b_ranking)
                player_a_recent_win_rate = float(
                    input_player_a_recent_win_rate
                )
                player_b_recent_win_rate = float(
                    input_player_b_recent_win_rate
                )

                if not 0 <= player_a_recent_win_rate <= 1:
                    error = "Player A recent win rate must be between 0 and 1."
                elif not 0 <= player_b_recent_win_rate <= 1:
                    error = "Player B recent win rate must be between 0 and 1."
                else:
                    prediction = tennis_model.predict_tree(
                        player_a_ranking,
                        player_b_ranking,
                        player_a_recent_win_rate,
                        player_b_recent_win_rate
                    )

            except ValueError:
                error = "Please enter valid numeric values."

    return render_template(
        "decision-tree-application.html",
        prediction=prediction,
        error=error,
        player_a_ranking=input_player_a_ranking,
        player_b_ranking=input_player_b_ranking,
        player_a_recent_win_rate=input_player_a_recent_win_rate,
        player_b_recent_win_rate=input_player_b_recent_win_rate
    )


@app.route("/decisionTree/evaluation")
def decision_tree_evaluation():
    return render_template(
        "decision-tree-evaluation.html",
        metrics=tennis_model.get_tree_metrics(),
        confusion_matrix=tennis_model.get_tree_confusion_matrix()
    )


if __name__ == "__main__":
    app.run(debug=True)
