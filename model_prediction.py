import pickle
import sklearn
import warnings
import numpy as np
from logger import App_Logger

warnings.filterwarnings("ignore")


class trained_model:
    def __init__(self):
        self.file_object = open("Logs/trained_model.txt", "a+")
        self.log_writer = App_Logger()

    def model_prediction(
        self, age, bmi, children, gender, smokers, northeast, northwest, southeast
    ):
        try:
            with open("trained_model/gradient_boosting_model", "rb") as f:
                clf = pickle.load(f)

            pred = np.round(
                clf.predict(
                    [
                        [
                            age,
                            bmi,
                            children,
                            gender,
                            smokers,
                            northeast,
                            northwest,
                            southeast,
                        ]
                    ]
                ),
                2,
            )
            self.log_writer.log(
                self.file_object, "The Trained model has sucessfully made a Prediction."
            )
            return f"The Predicted Insurance Premium is ${float(pred)}"

        except Exception as ex:
            self.log_writer.log(
                self.file_object,
                f"An Error have occured while predicting the data from trained model!!! ERROR: {ex}",
            )
            raise ex
