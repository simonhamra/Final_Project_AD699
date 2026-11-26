from pathlib import Path

import pandas as pd

import numpy as np

data_path = Path(__file__).with_name("geneva_listings.csv")
df = pd.read_csv(data_path)

print("Rows and columns:", df.shape)

df["price"] = (df["price"]
               .str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .astype(float))

print(df["price"].head())
print("Missing prices:", df["price"].isna().sum())
print(df["price"].describe())

# The highest price (CHF 25,000) is far above the 75th percentile (CHF 163).
# Check the 10 most expensive listings to see how unusual this price is before fitting the regression model.
listing_cols = ["id", "name", "room_type", "accommodates", "price"]

print("\n10 Highest-priced listings")
print(df.nlargest(10, "price")[listing_cols].to_string(index=False))

# Check how many values are missing in each candidate predictor before deciding which variables to include and how to handle missing values.
candidate_cols = ["accommodates",
                  "bedrooms",
                  "beds",
                  "room_type",
                  "neighbourhood_cleansed",]

print("\nMissing values")
print(df[candidate_cols].isna().sum())

import statsmodels.formula.api as smf
from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Start with guest capacity and room type because both are easy to interpret and have no missing values in this dataset.
model_df = df[["price", "accommodates", "room_type"]].copy()

# Keep the highest-priced listing in the baseline data because we have no evidence that its listed price is a data-entry error.
train_df, test_df = train_test_split(model_df,
                                     test_size=0.20,
                                     random_state=42,)

# Check where the highest price landed so we can assess its effect on the fitted model or on the test errors.
highest_price_index = model_df["price"].idxmax()
highest_price_split = "training" if highest_price_index in train_df.index else "test"
print("\nHighest-priced listing is in the", highest_price_split, "set.")

# Treat room type as a category when estimating price differences.
model = smf.ols("price ~ accommodates + C(room_type)", data=train_df).fit()

print("\nBaseline multiple regression")
print(model.summary())

# Refit after removing only the CHF 25,000 listing to measure its influence.
train_without_highest = train_df.drop(index=highest_price_index)
model_without_highest = smf.ols("price ~ accommodates + C(room_type)",
                                data=train_without_highest,).fit()

# Evaluate both models on the same listings for a fair comparison.
comparison = []

for label, fitted_model in [("Baseline", model),
                            ("Without CHF 25,000 training listing", model_without_highest),]:
    predictions = fitted_model.predict(test_df)
    comparison.append({"Model": label,
                       "Test MAE (CHF)": mean_absolute_error(test_df["price"], predictions),
                       "Test RMSE (CHF)": mean_squared_error(test_df["price"], predictions) ** 0.5,
                       "Accommodates coefficient": fitted_model.params["accommodates"],})

print("\nEffect of the highest-priced training listing")
print(pd.DataFrame(comparison).round(2).to_string(index=False))

# Fit the same predictors on log prices to reduce the influence of very large price differences while keeping all training listings.
log_model = smf.ols("np.log(price) ~ accommodates + C(room_type)",
                    data=train_df,).fit()

# Convert predictions back to CHF to compare errors on the same scale.
log_predictions_chf = np.exp(log_model.predict(test_df))

comparison.append({"Model": "Log-price regression",
                   "Test MAE (CHF)": mean_absolute_error(test_df["price"], 
                                                         log_predictions_chf),
                   "Test RMSE (CHF)": mean_squared_error(test_df["price"], 
                                                         log_predictions_chf) ** 0.5,})

print("\nTest performance comparison")
print(pd.DataFrame(comparison)[["Model", 
                                "Test MAE (CHF)", 
                                "Test RMSE (CHF)"]]
      .round(2)
      .to_string(index=False))

# Check how many training listings are available in each area before deciding whether to include individual areas in the regression.
train_areas = df.loc[train_df.index, "neighbourhood_cleansed"]
area_counts = train_areas.value_counts()

print("\nTraining listings by area")
print(area_counts.to_string())

# Reuse Simon's three area groups because several communes have too few training listings to estimate separate effects reliably.
inner_suburbs = ["Carouge", 
                 "Lancy", 
                 "Vernier", 
                 "Grand-Saconnex", 
                 "Meyrin",
                 "Onex", 
                 "Chêne-Bourg", 
                 "Chêne-Bougeries", 
                 "Thônex", 
                 "Plan-les-Ouates",]

cv_df = train_df.copy()
cv_df["area_group"] = "OUTER"
cv_df.loc[train_areas.isin(inner_suburbs), "area_group"] = "Inner suburbs"
cv_df.loc[train_areas.eq("Commune de Genève"), "area_group"] = "Genève city"

# Keep the same category labels across folds, including rare room types.
cv_df["room_type"] = cv_df["room_type"].astype("category")
cv_df["area_group"] = cv_df["area_group"].astype("category")

base_formula = "np.log(price) ~ accommodates + C(room_type)"
formulas = {"Without area group": base_formula,
            "With area group": base_formula + " + C(area_group)",}

# Compare both models on the same validation folds in CHF.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for train_positions, val_positions in cv.split(cv_df):
    fold_train = cv_df.iloc[train_positions]
    fold_val = cv_df.iloc[val_positions]

    for label, formula in formulas.items():
        fold_model = smf.ols(formula, data=fold_train).fit()
        predictions_chf = np.exp(fold_model.predict(fold_val))
        cv_results.append({"Model": label,
                           "MAE (CHF)": mean_absolute_error(fold_val["price"], 
                                                            predictions_chf),
                           "RMSE (CHF)": mean_squared_error(fold_val["price"], 
                                                            predictions_chf) ** 0.5,})

print("\nAverage 5-fold validation errors")
print(pd.DataFrame(cv_results).groupby("Model").mean().round(2))

print("\nSelected log-price regression summary")
print(log_model.summary())