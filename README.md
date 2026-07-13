import numpy as np
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# --- Mocking Data for Demonstration (Replace with your actual data from Part 2) ---
# Assuming 20 features and binary classification
np.random.seed(42)
X_train = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
X_test = pd.DataFrame(np.random.randn(300, 20), columns=[f'feature_{i}' for i in range(20)])
y_clf_train = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])
y_clf_test = np.random.choice([0, 1], size=300, p=[0.5, 0.5])

# Scaled inputs for manual standalone models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Task 1 & 2: Decision Tree Baseline & Controlled Tree
# ==========================================
print("--- Tasks 1 & 2: Decision Trees ---")
dt_unconstrained = DecisionTreeClassifier(random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)
print(f"Unconstrained DT - Train Acc: {accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled)):.4f} | Test Acc: {accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled)):.4f}")

dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)
print(f"Controlled DT    - Train Acc: {accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled)):.4f} | Test Acc: {accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled)):.4f}\n")

# ==========================================
# Task 3: Gini vs Entropy Comparison
# ==========================================
print("--- Task 3: Gini vs Entropy ---")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42).fit(X_train_scaled, y_clf_train)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42).fit(X_train_scaled, y_clf_train)
print(f"Gini Test Acc: {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)):.4f}")
print(f"Entropy Test Acc: {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)):.4f}\n")

# ==========================================
# Task 4 & 4a: Random Forest & Gradient Boosting
# ==========================================
print("--- Tasks 4 & 4a: Ensembles ---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)
rf_train_auc = roc_auc_score(y_clf_train, rf_model.predict_proba(X_train_scaled)[:, 1])
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_test_scaled)[:, 1])
print(f"RF - Train Acc: {accuracy_score(y_clf_train, rf_model.predict(X_train_scaled)):.4f} | Test Acc: {accuracy_score(y_clf_test, rf_model.predict(X_test_scaled)):.4f} | Test AUC: {rf_test_auc:.4f}")

# Top 5 Features
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nTop 5 Features by Importance (Random Forest):")
for i in range(5):
    print(f"{i+1}. {X_train.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# 4a. Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_test_scaled)[:, 1])
print(f"\nGB - Train Acc: {accuracy_score(y_clf_train, gb_model.predict(X_train_scaled)):.4f} | Test Acc: {accuracy_score(y_clf_test, gb_model.predict(X_test_scaled)):.4f} | Test AUC: {gb_test_auc:.4f}\n")

# ==========================================
# Task 4b: Feature Ablation Study
# ==========================================
print("--- Task 4b: Feature Ablation Study ---")
lowest_5_indices = indices[-5:]
features_to_keep = [i for i in range(X_train.shape[1]) if i not in lowest_5_indices]

X_train_reduced = X_train_scaled[:, features_to_keep]
X_test_reduced = X_test_scaled[:, features_to_keep]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
rf_reduced_test_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"Full Model Test AUC (All Features): {rf_test_auc:.4f}")
print(f"Reduced Model Test AUC (5 Dropped):  {rf_reduced_test_auc:.4f}\n")

# ==========================================
# Task 5: Cross-Validated Comparison
# ==========================================
print("--- Task 5: Cross-Validated Comparison ---")
# Instantiate baseline logistic regression for parity mapping
lr_model = LogisticRegression(random_state=42)

models = {
    "Logistic Regression": lr_model,
    "Controlled Decision Tree": dt_controlled,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc')
    print(f"{name} -> 5-Fold CV Mean AUC: {cv_scores.mean():.4f} (std: {cv_scores.std():.4f})")
print("\n")

# ==========================================
# Task 6: Hyperparameter Tuning with GridSearchCV
# ==========================================
print("--- Task 6: Pipeline & GridSearchCV ---")
pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_clf_train)  # Pipeline scales unscaled data internally

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validated AUC Score: {grid_search.best_score_:.4f}\n")

best_pipeline = grid_search.best_estimator_

# ==========================================
# Task 7: Manual Learning Curve
# ==========================================
print("--- Task 7: Manual Learning Curve Table ---")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
print(f"{'Training Fraction':<18} | {'Training AUC':<12} | {'Test AUC':<8}")
print("-" * 46)

for f in fractions:
    n_samples = int(f * len(X_train))
    X_train_sub = X_train.iloc[:n_samples]
    y_train_sub = y_clf_train[:n_samples]
    
    # Fit the best tuned configuration pipeline
    best_pipeline.fit(X_train_sub, y_train_sub)
    
    # Evaluate using AUC
    train_auc = roc_auc_score(y_train_sub, best_pipeline.predict_proba(X_train_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    
    print(f"{f:<18.1f} | {train_auc:<12.4f} | {test_auc:.4f}")
print("\n")

# ==========================================
# Task 8: Serialize and Reload Validation
# ==========================================
print("--- Task 8: Model Serialization & Reload Validation ---")
# Save the pipeline
joblib.dump(best_pipeline, 'best_model.pkl')

# Short production integration code block confirmation
loaded_model = joblib.load('best_model.pkl')
hand_crafted_rows = pd.DataFrame(np.random.randn(2, 20), columns=[f'feature_{i}' for i in range(20)])
predictions = loaded_model.predict(hand_crafted_rows)
print(f"Successfully loaded model! Predictions for 2 sample rows: {predictions}")
