from operator import methodcaller
from flask import Flask, render_template, request, send_file
import numpy as np
from model_prediction import trained_model
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
from model_for_upload import model_csv
from logger import App_Logger

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")


@app.route("/contactus", methods=["GET"])
def Contact():
    return render_template("contactus.html")


@app.route("/projects", methods=["GET"])
def projects():
    return render_template("projects.html")


@app.route("/team", methods=["GET"])
def team():
    return render_template("team.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    file_object = open("Logs/trained_model.txt", "a+")
    log_writer = App_Logger()

    try:
        age = float(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = float(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        if smoker == "yes":
            smoker = 1
        else:
            smoker = 0

        if sex == "Male":
            gender = 1
        else:
            gender = 0

        if region == "Northeast":
            region1 = 1
            region2 = 0
            region3 = 0

        elif region == "Northwest":
            region1 = 0
            region2 = 1
            region3 = 0

        elif region == "Southeast":
            region1 = 0
            region2 = 0
            region3 = 1

        else:
            region1 = 0
            region2 = 0
            region3 = 0

        train_model = trained_model()
        pred = train_model.model_prediction(
            age,
            bmi,
            children,
            gender,
            smoker,
            region1,
            region2,
            region3,
        )
        return render_template("result.html", data=pred)

    except Exception as ex:

        log_writer.log(
            file_object,
            f"An Error have occured while predicting the data from trained model!!! ERROR: {ex}",
        )

        error = f"Error: {ex}"
        pred = "The data you provided was not relevant or in correct format.  Please retry!!!"
        return render_template("result.html", data=pred, error=error)


@app.route("/upload", methods=["GET", "POST"])
def upload():
    try:
        csv = request.files["csv"]
        filename = secure_filename(csv.filename)
        csv.save(f"raw_data/{filename}")

        model = model_csv(filename)
        merged_df, df = model.preprocessing(filename)
        pred = model.predict_for_csv(merged_df, df)
        # new_file name created to remove .csv from file name
        new_filename = filename.split(".")
        new_filename = new_filename[0]
        file = f"exported_csv/{new_filename}_predicted_data.csv"

        return send_file(file, as_attachment=True)

    except Exception as ex:
        error = f"Error: {ex}"
        pred = "The data you provided was not relevant or was in incorrect format.  Please check the sample data (.csv) file from home page and retry!!!"
        return render_template("result.html", data=pred, error=error)


if __name__ == "__main__":
    app.run(debug=True)
