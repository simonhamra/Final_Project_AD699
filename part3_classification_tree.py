from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data_path = Path(__file__).with_name("geneva_listings.csv")
df = pd.read_csv(data_path)

# Check how common each response-time category first.
target = df["host_response_time"]

print("Host response time counts")
print(target.value_counts(dropna=False))
print("\nMissing response times:", target.isna().sum())

# Keep listings with an observed response time so the tree learns to distinguish among the four recorded response-time categories.
target_col = "host_response_time"

# Listing and host characteristics that could be available when making a prediction. 
# And check their missing values before fitting.
feature_cols = ["accommodates",
                "room_type",
                "number_of_reviews",
                "host_is_superhost",
                "host_listings_count"]

tree_df = df.loc[df[target_col].notna(), feature_cols + [target_col]].copy()

print("\nListings available for classification:", len(tree_df))
print("\nResponse time proportions (%)")
print(tree_df[target_col].value_counts(normalize=True).mul(100).round(1))
print("\nMissing values in candidate predictors")
print(tree_df[feature_cols].isna().sum())

# Keep the five listings with missing Superhost status in the analysis.
tree_df["host_is_superhost"] = tree_df["host_is_superhost"].fillna("unknown")

X = tree_df[feature_cols]
y = tree_df[target_col]

# Preserve the response-time proportions in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=42, 
                                                    stratify=y)

# Encode the two categorical predictors within each training fold.
preprocess = ColumnTransformer([("categories", 
                                 OneHotEncoder(handle_unknown="ignore"),
                                 ["room_type", 
                                  "host_is_superhost"])],
                               remainder="passthrough")

tree_pipeline = Pipeline([("preprocess", 
                           preprocess),
                          ("tree", 
                           DecisionTreeClassifier(random_state=42))])

# Compare tree sizes on the training set.
cv = StratifiedKFold(n_splits=5, 
                     shuffle=True, 
                     random_state=42)

tree_search = GridSearchCV(tree_pipeline,
                           param_grid={"tree__max_depth": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]},
                           scoring="balanced_accuracy",
                           cv=cv)

tree_search.fit(X_train, y_train)

depth_results = pd.DataFrame({"Max depth": tree_search.cv_results_["param_tree__max_depth"],
                              "Mean CV balanced accuracy": tree_search.cv_results_["mean_test_score"]})

print("\nCross-validation results by tree depth")
print(depth_results.round(4).to_string(index=False))
print("\nSelected max depth:", tree_search.best_params_["tree__max_depth"])

# Evaluate the tree selected by cross-validation on the held-out test set.
final_tree = tree_search.best_estimator_
test_predictions = final_tree.predict(X_test)

# Compare with a baseline that always predicts the most common training label.
majority_class = y_train.mode().iloc[0]
baseline_predictions = [majority_class] * len(y_test)

print("\nTest set performance")
print(f"Tree accuracy: {accuracy_score(y_test, test_predictions):.2%}")
print(f"Tree balanced accuracy: {balanced_accuracy_score(y_test, test_predictions):.2%}")
print(f"Baseline accuracy ({majority_class}): {accuracy_score(y_test, baseline_predictions):.2%}")
print(f"Baseline balanced accuracy: {balanced_accuracy_score(y_test, baseline_predictions):.2%}")

labels = ["within an hour",
          "within a few hours",
          "within a day",
          "a few days or more"]

test_matrix = pd.DataFrame(confusion_matrix(y_test, 
                                            test_predictions, 
                                            labels=labels),
                           index=labels,
                           columns=labels)

test_matrix.index.name = "Actual"
test_matrix.columns.name = "Predicted"

print("\nTest confusion matrix")
print(test_matrix.to_string())

fitted_tree = final_tree.named_steps["tree"]

print("\nActual tree depth:", fitted_tree.get_depth())
print("Number of leaves:", fitted_tree.get_n_leaves())

# Show the upper splits of the tree.
feature_names = final_tree.named_steps["preprocess"].get_feature_names_out()

fig, ax = plt.subplots(figsize=(18, 9))

plot_tree(fitted_tree,
          max_depth=2,
          feature_names=feature_names,
          class_names=fitted_tree.classes_.tolist(),
          filled=True,
          rounded=True,
          fontsize=9,
          ax=ax)

ax.set_title("Host response time classification tree — upper splits")

figure_path = Path(__file__).with_name("part3_tree_upper_splits.png")
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print("\nSaved tree figure:", figure_path)