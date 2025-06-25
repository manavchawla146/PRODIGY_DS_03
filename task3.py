import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("🔷 Loading and preprocessing data...")

# Load dataset
try:
    df = pd.read_csv('bank.csv', sep=';')
    print("✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Log available columns
print(f"Available columns in dataset: {list(df.columns)}")

# Feature engineering with column existence checks
if 'balance' in df.columns:
    df['balance_log'] = np.log1p(df['balance'].clip(lower=0))
    df['balance_bin'] = pd.cut(df['balance'], bins=[-float('inf'), 0, 500, 2000, float('inf')],
                               labels=['negative', 'low', 'medium', 'high'])
else:
    print("Warning: 'balance' column not found, skipping balance-related features")

if 'age' in df.columns:
    df['age_binned'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], labels=['young', 'middle', 'senior', 'elderly'])
    if 'balance_log' in df.columns:
        df['age_balance_interaction'] = df['age'] * df['balance_log']
        df['balance_age_ratio'] = df['balance_log'] / (df['age'] + 1)
else:
    print("Warning: 'age' column not found, skipping age-related features")

if 'campaign' in df.columns and 'previous' in df.columns:
    df['campaign_previous_interaction'] = df['campaign'] * df['previous']
else:
    print("Warning: 'campaign' or 'previous' column not found, skipping interaction feature")

if 'pdays' in df.columns:
    df['pdays_binned'] = pd.cut(df['pdays'], bins=[-1, 0, 30, 90, 999], labels=['never', 'recent', 'medium', 'long'])
else:
    print("Warning: 'pdays' column not found, skipping pdays_binned feature")

if 'month' in df.columns:
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    if df['month'].isin(month_map.keys()).all():
        df['month_binned'] = pd.cut(df['month'].map(month_map),
                                    bins=[0, 3, 6, 9, 12], labels=['Q1', 'Q2', 'Q3', 'Q4'])
    else:
        print("Warning: Invalid 'month' values, skipping month_binned feature")
else:
    print("Warning: 'month' column not found, skipping month_binned feature")

if 'duration' in df.columns and 'campaign' in df.columns:
    df['campaign_duration_interaction'] = df['campaign'] * df['duration']
else:
    print("Warning: 'duration' or 'campaign' column not found, skipping campaign_duration_interaction")

if 'duration' in df.columns and 'poutcome' in df.columns:
    df['duration_poutcome_interaction'] = df['duration'] * df['poutcome'].astype('category').cat.codes
else:
    print("Warning: 'duration' or 'poutcome' column not found, skipping duration_poutcome_interaction")

if 'month_binned' in df.columns and 'duration' in df.columns:
    df['month_duration_interaction'] = df['duration'] * df['month_binned'].astype('category').cat.codes
else:
    print("Warning: 'month_binned' or 'duration' column not found, skipping month_duration_interaction")

# Define features dynamically based on existence
potential_features = ['age', 'job', 'marital', 'education', 'campaign', 'previous', 'poutcome',
                      'balance_log', 'age_balance_interaction', 'campaign_previous_interaction',
                      'balance_age_ratio', 'balance_bin', 'age_binned', 'month_binned',
                      'duration', 'campaign_duration_interaction', 'duration_poutcome_interaction',
                      'month_duration_interaction']
selected_features = [col for col in potential_features if col in df.columns]
if not selected_features:
    print("❌ No valid features selected, exiting")
    exit(1)

X = df[selected_features]
y = df['y'].map({'yes': 1, 'no': 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

print("\n🔷 Creating advanced features...")
print(f"Using {len(selected_features)} features: {selected_features}")

# Define columns for preprocessing
categorical_cols = [col for col in ['job', 'marital', 'education', 'poutcome', 'pdays_binned',
                                    'balance_bin', 'age_binned', 'month_binned']
                   if col in X.columns]
numeric_cols = [col for col in ['age', 'balance_log', 'campaign', 'previous', 'age_balance_interaction',
                                'campaign_previous_interaction', 'balance_age_ratio', 'duration',
                                'campaign_duration_interaction', 'duration_poutcome_interaction',
                                'month_duration_interaction']
                if col in X.columns]

if not categorical_cols and not numeric_cols:
    print("❌ No valid columns for preprocessing, exiting")
    exit(1)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

# Get feature names after preprocessing
try:
    feature_names = (numeric_cols +
                     list(preprocessor.fit(X_train).named_transformers_['cat'].get_feature_names_out(categorical_cols)))
except Exception as e:
    print(f"❌ Error in preprocessing: {e}")
    exit(1)

# LightGBM pipeline
lgb_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.3, random_state=42)),
    ('classifier', lgb.LGBMClassifier(n_estimators=300, max_depth=7, learning_rate=0.02,
                                      num_leaves=25, min_data_in_leaf=5, subsample=0.8,
                                      colsample_bytree=0.8, reg_alpha=2, reg_lambda=2,
                                      is_unbalance=True, random_state=42, metric='auc',
                                      force_col_wise=True))
])

print("\n🔷 Training LightGBM model with hyperparameter tuning...")

# Grid search for LightGBM
param_grid_lgb = {
    'classifier__max_depth': [6, 7, 8],
    'classifier__learning_rate': [0.02, 0.03],
    'classifier__num_leaves': [25, 30, 35]
}
grid_search_lgb = GridSearchCV(lgb_pipeline, param_grid_lgb, cv=5, scoring='roc_auc', n_jobs=-1)
try:
    grid_search_lgb.fit(X_train, y_train)
    print(f"Best LightGBM parameters: {grid_search_lgb.best_params_}")
    print(f"Best LightGBM cross-validation AUC: {grid_search_lgb.best_score_:.4f}")
except Exception as e:
    print(f"❌ Error in grid search: {e}")
    exit(1)

# Train final calibrated model
print("\n🔷 Training final calibrated LightGBM model...")
best_params = grid_search_lgb.best_params_
lgb_clf = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=best_params['classifier__max_depth'],
    learning_rate=best_params['classifier__learning_rate'],
    num_leaves=best_params['classifier__num_leaves'],
    min_data_in_leaf=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=2,
    reg_lambda=2,
    is_unbalance=True,
    random_state=42,
    metric='auc',
    force_col_wise=True
)
lgb_calibrated = CalibratedClassifierCV(lgb_clf, cv=5, method='sigmoid')

# Preprocess data and apply SMOTE
try:
    X_train_preprocessed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=feature_names)
    X_test_preprocessed = pd.DataFrame(preprocessor.transform(X_test), columns=feature_names)
    smote = SMOTE(sampling_strategy=0.3, random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
    lgb_calibrated.fit(X_train_smote, y_train_smote)
except Exception as e:
    print(f"❌ Error in final model training: {e}")
    exit(1)

# Feature importance
lgb_clf.fit(X_train_smote, y_train_smote)
importances_lgb = lgb_clf.feature_importances_
feature_importance_lgb = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances_lgb
}).sort_values('Importance', ascending=False)
print("\n🔍 LightGBM Feature Importance (Top 10):")
print(feature_importance_lgb.head(10))

# Evaluate model
print("\n📊 LightGBM Comprehensive model evaluation...")
y_test_pred_proba_lgb = lgb_calibrated.predict_proba(X_test_preprocessed)[:, 1]

# Find optimal threshold using precision-recall curve
precision_lgb, recall_lgb, thresholds_pr_lgb = precision_recall_curve(y_test, y_test_pred_proba_lgb)
optimal_idx_lgb = np.argmax(np.array(recall_lgb[:-1]) * np.array(precision_lgb[:-1]))
optimal_threshold_lgb = thresholds_pr_lgb[optimal_idx_lgb]
y_test_pred_lgb = (y_test_pred_proba_lgb >= optimal_threshold_lgb).astype(int)

# Performance metrics
val_auc_lgb = roc_auc_score(y_test, y_test_pred_proba_lgb)
val_acc_lgb = accuracy_score(y_test, y_test_pred_lgb)
val_precision_lgb = precision_score(y_test, y_test_pred_lgb, pos_label=1)
val_recall_lgb = recall_score(y_test, y_test_pred_lgb, pos_label=1)
val_f1_lgb = f1_score(y_test, y_test_pred_lgb, pos_label=1)
conf_matrix_lgb = confusion_matrix(y_test, y_test_pred_lgb)

print(f"\nLightGBM Performance at optimal threshold {optimal_threshold_lgb:.4f}:")
print(f"Val AUC: {val_auc_lgb:.4f}")
print(f"Val Accuracy: {val_acc_lgb:.4f}")
print(f"Val Precision (yes): {val_precision_lgb:.4f}")
print(f"Val Recall (yes): {val_recall_lgb:.4f}")
print(f"Val F1 (yes): {val_f1_lgb:.4f}")
print(f"Confusion Matrix:\n{conf_matrix_lgb}")

# Generate predictions
submission = pd.DataFrame({
    'predicted_subscription_prob': y_test_pred_proba_lgb
})
submission.to_csv('submission_lightgbm.csv', index=False)
print("\n✅ LightGBM submission saved to submission_lightgbm.csv")

print("\n🔍 Final Predictions Summary (predicted probability of subscription):")
print(submission['predicted_subscription_prob'].describe(percentiles=[0.1, 0.5, 0.9, 0.99]))