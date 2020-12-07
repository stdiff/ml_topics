# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Feature Selection
#
# The aim of this document is to provide an brief introduction to feature selection techniques and an example code and analysis. 

from typing import *
from pathlib import Path

import numpy as np
import scipy.stats as stats
import pandas as pd

# +
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display, HTML
plt.style.use("fivethirtyeight")

from pylab import rcParams
rcParams['figure.figsize'] = 14, 6
# -

import warnings
warnings.filterwarnings("ignore") ## if lots of warnings are annoying you

# +
import altair as alt

def adhoc_theme():
    theme_dict = {
        'config': {"view"      : {"height":400, "width":800 },
                   "title"     : {"fontSize":24, "fontWeight":"normal", "titleAlign":"center"},
                   "axisLeft"  : {"labelFontSize":14, "titleFontSize":16},
                   "axisRight" : {"labelFontSize":14, "titleFontSize":16},
                   "header"    : {"labelFontSize":14, "titleFontSize":16, "titleAlign":"left"},
                   "axisBottom": {"labelFontSize":14, "titleFontSize":16},
                   "legend"    : {"labelFontSize":12, "titleFontSize":14},
                   "range"     : {"category": {"scheme": "category10"}}
    }}
    return theme_dict

alt.themes.register("adhoc_theme", adhoc_theme)
alt.themes.enable("adhoc_theme");

alt.data_transformers.enable('default', max_rows=30000); ## needed for large data set
# -
# ## Data set
#
# The data set we use is [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). The target variable is (logarithm of) the sale price. We remove two data points just after importing the data set because of the discussion [[1299, 524] considered harmful](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/105648).

import os
PROJ_ROOT = Path(os.getcwd()).parent
DATA_DIR = PROJ_ROOT / "data"
DATA_ZIP = DATA_DIR / "house-prices-advanced-regression-techniques.zip"

# +
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

with ZipFile(DATA_ZIP) as data_zip:
    with data_zip.open("train.csv","r") as fo:
        df_train = pd.read_csv(fo, index_col=0)
        
target = "SalePrice"
        
df_train.drop([524,1299], axis=0, inplace=True)
df_train, df_dev = train_test_split(df_train, test_size=0.3, random_state=1)
print("Training set size:", len(df_train))
print("dev set size     : ", len(df_dev))
# -


# ## Data preprocessing
#
# Some part of our data processing is based on the notebook [stacked housing](https://www.kaggle.com/danielj6/stacked-housing).

# ### Data fields
#
# Here's a brief version of what you'll find in the data description file.
#
# - SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# - MSSubClass: The building class
# - MSZoning: The general zoning classification
# - LotFrontage: Linear feet of street connected to property
# - LotArea: Lot size in square feet
# - Street: Type of road access
# - Alley: Type of alley access
# - LotShape: General shape of property
# - LandContour: Flatness of the property
# - Utilities: Type of utilities available
# - LotConfig: Lot configuration
# - LandSlope: Slope of property
# - Neighborhood: Physical locations within Ames city limits
# - Condition1: Proximity to main road or railroad
# - Condition2: Proximity to main road or railroad (if a second is present)
# - BldgType: Type of dwelling
# - HouseStyle: Style of dwelling
# - OverallQual: Overall material and finish quality
# - OverallCond: Overall condition rating
# - YearBuilt: Original construction date
# - YearRemodAdd: Remodel date
# - RoofStyle: Type of roof
# - RoofMatl: Roof material
# - Exterior1st: Exterior covering on house
# - Exterior2nd: Exterior covering on house (if more than one material)
# - MasVnrType: Masonry veneer type
# - MasVnrArea: Masonry veneer area in square feet
# - ExterQual: Exterior material quality
# - ExterCond: Present condition of the material on the exterior
# - Foundation: Type of foundation
# - BsmtQual: Height of the basement
# - BsmtCond: General condition of the basement
# - BsmtExposure: Walkout or garden level basement walls
# - BsmtFinType1: Quality of basement finished area
# - BsmtFinSF1: Type 1 finished square feet
# - BsmtFinType2: Quality of second finished area (if present)
# - BsmtFinSF2: Type 2 finished square feet
# - BsmtUnfSF: Unfinished square feet of basement area
# - TotalBsmtSF: Total square feet of basement area
# - Heating: Type of heating
# - HeatingQC: Heating quality and condition
# - CentralAir: Central air conditioning
# - Electrical: Electrical system
# - 1stFlrSF: First Floor square feet
# - 2ndFlrSF: Second floor square feet
# - LowQualFinSF: Low quality finished square feet (all floors)
# - GrLivArea: Above grade (ground) living area square feet
# - BsmtFullBath: Basement full bathrooms
# - BsmtHalfBath: Basement half bathrooms
# - FullBath: Full bathrooms above grade
# - HalfBath: Half baths above grade
# - Bedroom: Number of bedrooms above basement level
# - Kitchen: Number of kitchens
# - KitchenQual: Kitchen quality
# - TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# - Functional: Home functionality rating
# - Fireplaces: Number of fireplaces
# - FireplaceQu: Fireplace quality
# - GarageType: Garage location
# - GarageYrBlt: Year garage was built
# - GarageFinish: Interior finish of the garage
# - GarageCars: Size of garage in car capacity
# - GarageArea: Size of garage in square feet
# - GarageQual: Garage quality
# - GarageCond: Garage condition
# - PavedDrive: Paved driveway
# - WoodDeckSF: Wood deck area in square feet
# - OpenPorchSF: Open porch area in square feet
# - EnclosedPorch: Enclosed porch area in square feet
# - 3SsnPorch: Three season porch area in square feet
# - ScreenPorch: Screen porch area in square feet
# - PoolArea: Pool area in square feet
# - PoolQC: Pool quality
# - Fence: Fence quality
# - MiscFeature: Miscellaneous feature not covered in other categories
# - MiscVal: $Value of miscellaneous feature
# - MoSold: Month Sold
# - YrSold: Year Sold
# - SaleType: Type of sale
# - SaleCondition: Condition of sale
#

# +
from adhoc.processing import Inspector

inspector_train = Inspector(df_train)

with pd.option_context("display.max_rows",None):
    display(inspector_train)


# -

# ### Data type correction
#
# Some categorical values are described as numbers in the data set so that they look like continuous variables. For example `MSSubClass` (the building class). 

def correct_dtype(data:pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["MSSubClass"] = df["MSSubClass"].apply(lambda x: f"C{x:03d}")
    df["MoSold"] = df["MoSold"].apply(lambda x: f"M{x:02d}")
    
    ## YearBuilt (Original construction date), YearRemodAdd (Remodel date) and YrSold (Year Sold)
    ## Instead we use differences.
    df["Age"] = df["YrSold"] - df["YearBuilt"]
    df["AgeAfterRemod"] = df["YrSold"] - df["YearRemodAdd"]
    
    ## Apply the same idea to GarageYrBlt
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    
    df.drop(["YearBuilt","YearRemodAdd","YrSold", "GarageYrBlt"], axis=1, inplace=True)    
    
    ## only four houses have a pool
    df["hasPool"] = df["PoolArea"].apply(lambda x: 1 if x > 0 else 0)
    df.drop("hasPool", axis=1, inplace=True)

    return df


# +
df_train = correct_dtype(df_train)
df_dev = correct_dtype(df_dev)

inspector_train = Inspector(df_train)
# -

# ### Missing values 

inspector_train.result.query("count_na > 0").sort_values(by="count_na", ascending=False)

# +
## badly filled columns 
badly_filled_columns = inspector_train.result.query("rate_na > 0.4").index.tolist()

## remove badly filled columns
df_train.drop(badly_filled_columns, axis=1, inplace=True)
df_dev.drop(badly_filled_columns, axis=1, inplace=True)

## update Inspector instance
inspector_train = Inspector(df_train)

## columns which are removed
badly_filled_columns
# -

with pd.option_context("display.max_rows",None):
    display(inspector_train)

# ## Feature selection with statistical tests
#
# We can check whether a feature is statistically significant to the target variable. If it is not the case, then we might drop the feature from the data set. 
#
# Since our target variable is continuous, we perform the following two types of (non-parametric) statistical tests.
#
# - For categorical feature (such as `SaleType`): **one-way ANOVA on ranks**
#   - Null-hypothesis: The medians of the target variable in the categories are the same.
# - For continuous feature (such as `GarageArea`): **Speaman correlation**
#   - Null-hypothesis: The feature variable is not correlated to the target variable.
#
# Advantages:
#
# - We can compute the $p$-values just after simple preprocessing.
# - The huge number of observations (rows) hardly ever matters. 
# - The huge number of features (columns) hardly ever matters, neither.
#
# Disadvantages:
#
# - The $p$-values are not suitable for a ranking. Only for a screening.
#   - The $p$-value measures how small/large the difference is under the null-hypothesis. (The $p$-value is small => the difference is large.)
# - We ignore the possibility that the feature is important under a certain combination with another feature.

# +
## Actually we should apply this method after imputation.
## Because this is just demonstration and we do not use the result,
## we just drop rows with a missing values and the field "Utilities",
## which is constant because of dropping missing values.

df_na_dropped = df_train.dropna(how="any").drop("Utilities", axis=1)
inspector_na_dropped = Inspector(df_na_dropped)

with pd.option_context("display.max_rows", None):
    display(inspector_na_dropped.significance_test_features(target).sort_values(by="pval"))


# -

# ## ML Pipeline

# +
def separate_x_y(data:pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = data.drop(target, axis=1)
    y = np.log1p(data[target]).rename("LogSalePrice")
    return X, y

X_train, y_train = separate_x_y(df_train)
X_dev, y_dev = separate_x_y(df_dev)
# -

# ### Filling missing values
#
# Some columns describe extra information about another column. For example `FireplaceQu`. This column describes a fireplace and therefore the column is filled if (and only if) `Fireplaces` is not zero. `Bsmt`, the abbreviation of "basement", is used as part of feature names. These features are missing if the house has no basement. 
#
# - NA means 'NA': `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`, `BsmtQual`, 
#   `GarageCond`, `GarageFinish`, `GarageQual`, `GarageType`,  `MasVnrType`
# - NA means 0: `GarageArea`, `MasVnrArea`
# - NA means 'unknown': `Electrical`, `GarageYrBuilt` (or `GarageAge`)

inspector_train.result.query("count_na > 0").sort_index()

cols_na_is_na = ["BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "BsmtQual", 
                   "GarageCond", "GarageFinish", "GarageQual", "GarageType", "MasVnrType"]
cols_na_is_zero = ["LotFrontage", "GarageArea", "MasVnrArea"]
cols_na_is_unknown = ["Electrical"]
cols_median_fills_na = ["GarageAge"]

# +
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

imputer = ColumnTransformer([
    ("NA", SimpleImputer(strategy="constant", fill_value="NA"), cols_na_is_na),
    ("Zero", SimpleImputer(strategy="constant", fill_value=0), cols_na_is_zero),
    ("Unknown", SimpleImputer(strategy="constant", fill_value="unknown"), cols_na_is_unknown),
    ("Median", SimpleImputer(strategy="median"), cols_median_fills_na)
], remainder="passthrough")

# +
## check if the imputer works expectedly

cols_with_na = cols_na_is_na + cols_na_is_zero + cols_na_is_unknown + cols_median_fills_na
cols_without_na = [col for col in X_train.columns if col not in cols_with_na]
columns_after_imputation = cols_with_na + cols_without_na
pd.DataFrame(imputer.fit_transform(X_train), columns=columns_after_imputation)


# -

# ### Encoding of categorical variables
#
# #### Ordinal variable
#
# Some categorical variables can be ordered. For example `ExterQual`, `BsmtQual`, `KitchenQual` and `GarageQual` have values `Po` (Poor), `Fa` (Fair), `TA` (Average/Typical), `Gd` (Good), `Ex` (Excellent). (The detailed description of labels can be found `data_description.txt`.) Such a variable should be regarded as an ordered variable. 
#
# #### Nominal variable
#
# We just apply an ordinary one hot encoder to the nominal variables.

def names2indices(names:str) -> List[int]:
    """
    Because after imputation there is no column names available,
    we need a function which converts a list of (original) column
    names into the 
    
    global variable
    columns_after_imputation
    """
    return [i for i,name in enumerate(columns_after_imputation) if name in names]


# +
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

col_quality = ["BsmtCond", "BsmtQual", "ExterCond", "ExterQual", 
               "GarageCond", "GarageQual", "HeatingQC", "KitchenQual"]
col_ordered = col_quality + ["LandSlope", "BsmtExposure", "GarageFinish"]
col_nominal = [cat for cat in inspector_train.get_cats() if cat not in col_ordered]

encoder = ColumnTransformer([
    ("Quality", 
     OrdinalEncoder(categories=[["NA","Po","Fa","TA","Gd","Ex"]]*len(col_quality)), 
     names2indices(col_quality)),
    ("LandSlope", 
     OrdinalEncoder(categories=[["Gtl","Mod","Sev"]]), 
     names2indices(["LandSlope"])),
    ("BsmtExposure", 
     OrdinalEncoder(categories=[["NA", "No", "Mn", "Av", "Gd"]]), 
     names2indices(["BsmtExposure"])),
    ("GarageFinish", 
     OrdinalEncoder(categories=[["NA", "Unf", "RFn", "Fin"]]), 
     names2indices(["GarageFinish"])),
    ("Nominal",
     OneHotEncoder(handle_unknown="ignore", sparse=False), 
     names2indices(col_nominal))],
    remainder='passthrough')

encoder.fit(imputer.fit_transform(X_train));


# -

def generate_names_from_ohe(names:List[str], one_hot_encoder:OneHotEncoder) -> List[str]:
    column_names = []
    for name, cats in zip(names, one_hot_encoder.categories_):
        column_names.extend([f"{name}_{cat}" for cat in cats])
    return column_names


# +
## columns after encoding 

columns_after_encoding = col_ordered
columns_after_encoding.extend(generate_names_from_ohe(col_nominal, encoder.transformers_[-2][1]))
columns_after_encoding.extend([col for col in columns_after_imputation if col not in col_ordered+col_nominal])
len(columns_after_encoding)

# +
df_after_encoding = pd.DataFrame(encoder.transform(imputer.transform(X_train)).astype(np.float),
                                 columns=columns_after_encoding,
                                 index=X_train.index)
inspector_after_encoding = Inspector(df_after_encoding)

with pd.option_context("display.max_rows", None):
    display(inspector_after_encoding)

# +
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ("imputer", imputer),
    ("encoder", encoder),
    ("rf", RandomForestRegressor(max_features="auto", random_state=3))
])

param_grid = {"rf__n_estimators": [400, 600, 800], 
              "rf__max_depth": [10, 12, 16]}

rf = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="neg_root_mean_squared_error", return_train_score=True)

# +
# %%time
from adhoc.modeling import cv_results_summary

rf.fit(X_train, y_train)
cv_results_summary(rf)

# +
#from xgboost import XGBRegressor
#
#pipeline = Pipeline([
#    ("imputer", imputer),
#    ("encoder", encoder),
#    ("xgb", XGBRegressor(learning_rate=0.01, random_state=3))
#])
#
#param_grid = {"xgb__n_estimators": [200, 400, 600], 
#              "xgb__max_depth": [4, 6, 8]}
#
#xgb = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=4, scoring="neg_root_mean_squared_error", return_train_score=True)

# +
# #%%time
#from adhoc.modeling import cv_results_summary
#
#xgb.fit(X_train, y_train)
#cv_results_summary(xgb)
# -

# ## Feature Importance
#
# A feature importance is a technique to measure the impact of a feature on the model performance. 
#
# Advantages:
#
# - The feature importance can be used to rank features. 
# - We do not underestimate the possibility that a feature has an impact under a combination with another feature.
#
# Disadvantages:
#
# - You need to train a model.
#   - Data processing / feature engineering is required. 
# - The feature importance depends not only on the data set, but also on the model.
#   - You can not compare feature importance of different models.

# ### Feature Importance of tree-based models
#
# A tree-based model (decision tree, random forest, etc.) is equipped with its own feature importance.
#
# Given a decision tree, the splitting variable of each node reduces the impurity. We takes the sum of the reduction ([weighted impurity decreasing equation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)) by each (splitting) feature as the feature importance. The feature importance of a random forest model is just a (normalized) average of feature importance of the decision trees of the model.

model = rf.best_estimator_.steps[2][1]
model

pd.Series(model.feature_importances_, index=pd.Series(columns_after_encoding, name="feature"))\
  .sort_values(ascending=False)\
  .rename("feature importance").reset_index().head(10)

# ### Permutation (feature) importance
#
# (cf. scikit-learn documentation for [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html))
#
# In order to measure the impact of a feature, we may drop the feature and train a model of the same architecture/hyperparameters. But this is time/resource consuming. Instead we can permute the values of the features randomly so that the feature pretends to be absent.
#
# Advantages:
#
# - We can apply the algorithm to any trained model.
# - It is easy to implement it. (`sklearn` has an API for it.)
# - (If the model is a scikit-learn pipeline, then we can compute the feature importance of the variable before (one-hot) encoding.)
#
# Disadvantages:
#
# - Not deterministic. We might need to look at confidence intervals in some cases. (But we can increase the number of experiments easily.)
# - We need to have a dev set. i.e. the data set which has the same distribution of the test set, but we are allowed to look at the dataset. 

# %%time
from sklearn.inspection import permutation_importance
p_importance = permutation_importance(rf, X_dev, y_dev, n_repeats=30, n_jobs=4, random_state=9)

df_fi = pd.DataFrame({
    "feature": X_dev.columns.tolist(),
    "importance": p_importance["importances_mean"]
})
df_fi.sort_values(by="importance", ascending=False, inplace=True)

# +
from sklearn.metrics import mean_squared_error

rmse_dev = mean_squared_error(y_dev, rf.predict(X_dev), squared=False)
print(f"RMSE on the dev set: {rmse_dev:0.4f}")
# -

# The following bar chart shows the permutation feature importance. If the bar is orange, then the permutation importance of the corresponding feature is smaller than 1% of the RMSE on the dev set.

alt.Chart(df_fi)\
   .mark_bar()\
   .encode(x=alt.X("importance:Q"), 
           y=alt.Y("feature:N", sort="-x"),
           tooltip=["feature:N", alt.Tooltip("importance:Q", format="0.5f")],
           color=alt.condition(alt.datum.importance > 0.01*rmse_dev, 
                               alt.value("#1f77b4"), alt.value("#ff7f0e")))\
   .properties(height=1024, title="Permutation feature importance")


# +
def scatter_plot(data:pd.DataFrame, field:str, width:int=240) -> alt.Chart:
    chart = alt.Chart(df_dev[[field,target]])\
               .mark_circle()\
               .encode(x=alt.X(field, scale=alt.Scale(zero=False), title=None), 
                       y=alt.Y(target, scale=alt.Scale(type="log"), title=f"Log{target}"))\
               .properties(width=width, title=field)
    return chart

def box_plot(data:pd.DataFrame, field:str, sort:Optional[List[str]]=None, width:int=200, labelAngle:int=0) -> alt.Chart:
    chart = alt.Chart(df_dev[[field,target]])\
                .mark_boxplot()\
                .encode(x=alt.X(f"{field}:N", sort=sort, axis=alt.Axis(labelAngle=labelAngle), title=None),
                        y=alt.Y(target, scale=alt.Scale(type="log")),
                        color=alt.Color("count()", legend=None))\
                .properties(width=width, title=field)
    return chart


# -

# The following diagrams show the relation between the best 3 features and the target variables.

# +
chart_top1 = scatter_plot(df_dev, "GrLivArea")
chart_top2 = box_plot(df_dev, "ExterQual", sort=["Po","Fa","TA","Gd","Ex"])
chart_top3 = scatter_plot(df_dev, "TotalBsmtSF")

chart_top1 | chart_top2 | chart_top3
# -

# The following diagrams show the relation between the worst 3 features and the target variables.

# +
worst1 = box_plot(df_dev, "SaleCondition", labelAngle=90)
worst2 = box_plot(df_dev, "GarageType", labelAngle=90)
worst3 = box_plot(df_dev, "Exterior2nd", width=320, labelAngle=90)

worst1 | worst2 | worst3

# +
from adhoc.utilities import bins_by_tree

def BinnedPrediction(data:pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=data.index)
    df["GrLivArea"] = bins_by_tree(data, "GrLivArea", target, target_is_continuous=True, n_bins=4)
    df["ExterQual"] = data["ExterQual"]
    df["TotalBsmtSF"] = bins_by_tree(data, "TotalBsmtSF", target, target_is_continuous=True, n_bins=4)
    df["Prediction"] = np.expm1(rf.predict(data.drop(target, axis=1)))
    dg = df.groupby(["GrLivArea","ExterQual","TotalBsmtSF"])["Prediction"].agg(size="size", average="mean").reset_index()
    dg["average"] = np.round(dg["average"])
    return dg

df_binned_pred = BinnedPrediction(df_dev)
OrderGrLivArea = [str(x) for x in df_binned_pred["GrLivArea"].cat.categories]
OrderTotalBsmtSF = list(reversed([str(x) for x in df_binned_pred["TotalBsmtSF"].cat.categories]))


# -

# The following heatmaps show the average of the predictions of regions.

# +
def heatmap(data:pd.DataFrame, annot:bool=True) -> alt.Chart:
    charts = None
    for exter_qual in ["Fa","TA","Gd","Ex"]:
        base = alt.Chart(data.query("ExterQual == @exter_qual").astype(str))
        heat = base.mark_rect(opacity=0.7)\
                   .encode(x=alt.X("GrLivArea:N", axis=alt.Axis(labelAngle=0), sort=OrderGrLivArea),
                           y=alt.Y("TotalBsmtSF:N", sort=OrderTotalBsmtSF),
                           color=alt.Color("average:Q",
                                           scale=alt.Scale(type="log"), 
                                           title="Prediction"),
                           tooltip=["GrLivArea:N","totalBsmtSF:N","average:Q"])

        text = base.mark_text(fontSize=16)\
                   .encode(x=alt.X("GrLivArea:N", sort=OrderGrLivArea), 
                           y=alt.Y("TotalBsmtSF:N", sort=OrderTotalBsmtSF), 
                           text=alt.Text("average:Q"))
        
        chart = (heat + text).properties(width=600, height=120, title=f"ExterQual = {exter_qual}")
        
        if charts is None:
            charts = chart
        else:
            charts = charts & chart

    return charts

heatmap(df_binned_pred)
# -

# ## Environment

# %load_ext watermark
# %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn,altair,torch
