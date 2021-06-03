"""

"""
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path
from zipfile import ZipFile


import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from adhoc.processing import Inspector


class Preprocessor:
    imputer: Optional[ColumnTransformer] = None
    columns_after_imputation: List[str] = None
    encoder: Optional[ColumnTransformer] = None
    col_ordered: List[str] = None
    col_nominal: List[str] = None
    target: str = "SalePrice"

    @classmethod
    def load_data(cls, env: str) -> pd.DataFrame:
        file_name = "house-prices-advanced-regression-techniques.zip"

        if env == "Prod":
            data_path = Path("/tmp") / file_name
        else:
            data_path = Path(__file__).parents[1] / "data" / file_name

        if not data_path.exists():
            raise FileNotFoundError(data_path.absolute())

        with ZipFile(data_path) as data_zip:
            with data_zip.open("train.csv", "r") as fo:
                df = pd.read_csv(fo, index_col=0)

        return df

    @classmethod
    def correct_dtype(cls, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["MSSubClass"] = df["MSSubClass"].apply(lambda x: f"C{x:03d}")
        df["MoSold"] = df["MoSold"].apply(lambda x: f"M{x:02d}")

        ## YearBuilt (Original construction date), YearRemodAdd (Remodel date) and YrSold (Year Sold)
        ## Instead we use differences.
        df["Age"] = df["YrSold"] - df["YearBuilt"]
        df["AgeAfterRemod"] = df["YrSold"] - df["YearRemodAdd"]

        ## Apply the same idea to GarageYrBlt
        df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]

        df.drop(["YearBuilt", "YearRemodAdd", "YrSold", "GarageYrBlt"], axis=1, inplace=True)

        ## only four houses have a pool
        df["hasPool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
        df.drop("hasPool", axis=1, inplace=True)

        ## remove badly filled columns
        badly_filled_columns = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
        df.drop(badly_filled_columns, axis=1, inplace=True)
        return df

    @classmethod
    def init_imputer(cls, original_columns: pd.Index):
        ## rules for filling missing values
        cols_na_is_na = [
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "BsmtQual",
            "GarageCond",
            "GarageFinish",
            "GarageQual",
            "GarageType",
            "MasVnrType",
        ]
        cols_na_is_zero = ["LotFrontage", "GarageArea", "MasVnrArea"]
        cols_na_is_unknown = ["Electrical"]
        cols_median_fills_na = ["GarageAge"]

        cols_with_na = cols_na_is_na + cols_na_is_zero + cols_na_is_unknown + cols_median_fills_na
        cols_without_na = [col for col in original_columns if col not in cols_with_na]
        cls.columns_after_imputation = cols_with_na + cols_without_na

        cls.imputer = ColumnTransformer(
            [
                ("NA", SimpleImputer(strategy="constant", fill_value="NA"), cols_na_is_na),
                ("Zero", SimpleImputer(strategy="constant", fill_value=0), cols_na_is_zero),
                (
                    "Unknown",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                    cols_na_is_unknown,
                ),
                ("Median", SimpleImputer(strategy="median"), cols_median_fills_na),
            ],
            remainder="passthrough",
        )

    @classmethod
    def names2indices(cls, names: List[str]) -> List[int]:
        """
        Because after imputation there is no column names available,
        we need a function which converts a list of (original) column
        names into the column number.
        """

        if cls.columns_after_imputation is None:
            raise ValueError("You need to instantiate an imputer at first.")

        return [i for i, name in enumerate(cls.columns_after_imputation) if name in names]

    @classmethod
    def fit_encoder(cls, data: pd.DataFrame):
        col_quality = [
            "BsmtCond",
            "BsmtQual",
            "ExterCond",
            "ExterQual",
            "GarageCond",
            "GarageQual",
            "HeatingQC",
            "KitchenQual",
        ]
        inspector = Inspector(data)
        cls.col_ordered = col_quality + ["LandSlope", "BsmtExposure", "GarageFinish"]
        cls.col_nominal = [cat for cat in inspector.get_cats() if cat not in cls.col_ordered]

        cls.encoder = ColumnTransformer(
            [
                (
                    "Quality",
                    OrdinalEncoder(
                        categories=[["NA", "Po", "Fa", "TA", "Gd", "Ex"]] * len(col_quality)
                    ),
                    cls.names2indices(col_quality),
                ),
                (
                    "LandSlope",
                    OrdinalEncoder(categories=[["Gtl", "Mod", "Sev"]]),
                    cls.names2indices(["LandSlope"]),
                ),
                (
                    "BsmtExposure",
                    OrdinalEncoder(categories=[["NA", "No", "Mn", "Av", "Gd"]]),
                    cls.names2indices(["BsmtExposure"]),
                ),
                (
                    "GarageFinish",
                    OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]]),
                    cls.names2indices(["GarageFinish"]),
                ),
                (
                    "Nominal",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    cls.names2indices(cls.col_nominal),
                ),
            ],
            remainder="passthrough",
        )
        cls.encoder.fit(cls.imputer.fit_transform(data))

    @classmethod
    def generate_names_from_ohe(cls, names: List[str], one_hot_encoder: OneHotEncoder) -> List[str]:
        column_names = []
        for name, cats in zip(names, one_hot_encoder.categories_):
            column_names.extend([f"{name}_{cat}" for cat in cats])
        return column_names

    @classmethod
    def process(cls, data: pd.DataFrame) -> pd.DataFrame:
        if cls.col_ordered is None:
            raise ValueError("You forgot to fit an encoder.")

        columns_after_encoding = cls.col_ordered.copy()
        columns_after_encoding.extend(
            cls.generate_names_from_ohe(cls.col_nominal, cls.encoder.transformers_[-2][1])
        )
        columns_after_encoding.extend(
            [
                col
                for col in cls.columns_after_imputation
                if col not in cls.col_ordered + cls.col_nominal
            ]
        )

        df_after_encoding = pd.DataFrame(
            cls.encoder.transform(cls.imputer.transform(data)).astype(float),
            columns=columns_after_encoding,
            index=data.index,
        )
        return df_after_encoding

    @classmethod
    def separate_x_y(cls, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        X = data.drop(cls.target, axis=1)
        y = np.log1p(data[cls.target]).rename(cls.target)
        return X, y


def main():
    env = os.environ.get("Env", "Dev")

    df = Preprocessor.load_data(env)
    df = Preprocessor.correct_dtype(df)
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=1)
    Preprocessor.init_imputer(df_train.columns)
    Preprocessor.fit_encoder(df_train)

    df_train = Preprocessor.process(df_train)
    df_test = Preprocessor.process(df_test)


if __name__ == "__main__":
    main()
