from fileinput import filename
from heapq import merge
import pickle
import pandas as pd
import numpy as np
from sklearn.utils import column_or_1d
from logger import App_Logger
from sklearn.impute import SimpleImputer


class model_csv:
    def __init__(self, filename):
        self.file_object = open("Logs/model_csv.txt", "a+")
        self.log_writer = App_Logger()
        self.new_filename = filename.split(".")
        self.new_filename = self.new_filename[0]

    def preprocessing(self, filename):
        self.Imputer = SimpleImputer()

        try:
            df = pd.read_csv(f"raw_data/{filename}")
            # dropping all the NaN values
            df = df.dropna()
            # drop the labels specified in the columns
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

            # using pandas get dummies
            dummies_region = pd.get_dummies(df.region)
            dummies_region = dummies_region.drop("southwest", axis="columns")

            dummies_gender = pd.get_dummies(df.sex)
            dummies_gender = dummies_gender.drop("female", axis="columns")
            dummies_gender["gender"] = dummies_gender.male
            dummies_gender = dummies_gender.drop("male", axis="columns")

            dummies_smoker = pd.get_dummies(df.smoker)
            dummies_smoker = dummies_smoker.drop("no", axis="columns")
            dummies_smoker["smokers"] = dummies_smoker.yes
            dummies_smoker = dummies_smoker.drop("yes", axis="columns")

            merged_df = pd.concat(
                [df, dummies_gender, dummies_smoker, dummies_region], axis="columns"
            )
            merged_df = merged_df.drop(["sex", "smoker", "region"], axis="columns")

            # Imputing missing values using Simple Imputer
            for col in merged_df:
                merged_df[col] = self.Imputer.fit_transform(merged_df[[col]])

            self.log_writer.log(
                self.file_object, "Data Preprocessing have been sucessfully Done."
            )

            return merged_df, df

        except Exception as ex:
            self.log_writer.log(
                self.file_object,
                f"An Error have occured while preprocessing the data!!! ERROR: {ex}",
            )
            raise ex

    def predict_for_csv(self, merged_df, df):
        try:
            with open("trained_model/gradient_boosting_model", "rb") as f:
                clf = pickle.load(f)
            prediction_for_csv = clf.predict(merged_df)
            prediction_for_csv = pd.DataFrame(prediction_for_csv, columns=["predicted"])
            csv_to_export = pd.concat([df, prediction_for_csv], axis="columns")
            export = csv_to_export.to_csv(
                f"exported_csv/{self.new_filename}_predicted_data.csv", index=False
            )

            self.log_writer.log(
                self.file_object,
                "The Processed Data have been sucessfully exported to .csv file",
            )
            return export

        except Exception as ex:
            self.log_writer.log(
                self.file_object,
                f"An Error have occured while exporting the file to .csv!!! ERROR: {ex}",
            )
            raise ex
