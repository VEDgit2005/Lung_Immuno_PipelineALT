# OPTIMIZED GENE EXPRESSION CLASSIFICATION PIPELINE
# Fixed: Creates binary response from PFS data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.pipeline import Pipeline
import joblib
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIG - OPTIMIZED FOR GENE EXPRESSION
# ============================
DATA_FILE = "merged_dataset.csv"
N_FEATURES_SELECT = 100  # Select top 100 most predictive genes
RANDOM_STATE = 42

# PFS threshold for defining response (in days)
# You can adjust this based on clinical knowledge
PFS_THRESHOLD_MONTHS = 6  # 6 months = 180 days
PFS_THRESHOLD_DAYS = PFS_THRESHOLD_MONTHS * 30

# ============================
# LOAD DATA
# ============================
print(f"🧬 Loading gene expression dataset from {DATA_FILE} …")
df = pd.read_csv(DATA_FILE)
print(f"Dataset shape: {df.shape}")

# ============================
# CREATE RESPONSE FROM PFS DATA
# ============================
print(f"\n🎯 Creating binary response from PFS data...")
print(f"Using PFS threshold: {PFS_THRESHOLD_MONTHS} months ({PFS_THRESHOLD_DAYS} days)")

# Check PFS data
pfs_data = df['pfs']
print(f"\nPFS Statistics:")
print(f"  Min PFS: {pfs_data.min()} days ({pfs_data.min()/30:.1f} months)")
print(f"  Max PFS: {pfs_data.max()} days ({pfs_data.max()/30:.1f} months)")
print(f"  Mean PFS: {pfs_data.mean():.1f} days ({pfs_data.mean()/30:.1f} months)")
print(f"  Median PFS: {pfs_data.median():.1f} days ({pfs_data.median()/30:.1f} months)")

# Create binary response: 1 = Good response (PFS > threshold), 0 = Poor response (PFS <= threshold)
df['response_binary'] = (df['pfs'] > PFS_THRESHOLD_DAYS).astype(int)

print(f"\nBinary Response Distribution:")
response_counts = df['response_binary'].value_counts()
print(f"  Poor Response (PFS ≤ {PFS_THRESHOLD_MONTHS} months): {response_counts.get(0, 0)} patients")
print(f"  Good Response (PFS > {PFS_THRESHOLD_MONTHS} months): {response_counts.get(1, 0)} patients")

# Check if we have balanced classes
if len(response_counts) < 2:
    print(f"\n⚠️  All patients fall into one category with current threshold.")
    print("Trying alternative: Use median PFS as threshold...")
    median_pfs = df['pfs'].median()
    df['response_binary'] = (df['pfs'] > median_pfs).astype(int)
    response_counts = df['response_binary'].value_counts()
    print(f"Using median PFS ({median_pfs:.0f} days = {median_pfs/30:.1f} months) as threshold:")
    print(f"  Below Median PFS: {response_counts.get(0, 0)} patients")
    print(f"  Above Median PFS: {response_counts.get(1, 0)} patients")

# ============================
# PREPARE FEATURES AND TARGET
# ============================
print(f"\n🧹 Preparing features and target...")

# Drop identifier columns and original empty response
columns_to_drop = ['expression_id', 'GSM', 'patient_id', 'pfs', 'response']  # Keep response_binary

X = df.drop(columns=columns_to_drop)
y = df['response_binary']

# Handle gender encoding
if 'gender' in X.columns:
    print("Encoding gender: male=1, female=0")
    X['gender'] = X['gender'].map({'male': 1, 'female': 0})

print(f"Final feature matrix: {X.shape}")
print(f"Target variable: {y.name}")
print(f"Class balance: {dict(y.value_counts())}")

# ============================
# FEATURE SELECTION PIPELINE
# ============================
print(f"\n🎯 Selecting top {N_FEATURES_SELECT} most predictive genes...")

# Create feature selection pipeline
feature_selector = SelectKBest(score_func=f_classif, k=min(N_FEATURES_SELECT, X.shape[1]-1))
scaler = StandardScaler()

# Fit feature selector
X_selected = feature_selector.fit_transform(X, y)
selected_features = X.columns[feature_selector.get_support()]

print(f"Selected {len(selected_features)} features")

# Show feature selection scores
feature_scores = pd.DataFrame({
    'feature': selected_features,
    'f_score': feature_selector.scores_[feature_selector.get_support()]
}).sort_values('f_score', ascending=False)

print(f"Top 10 most discriminative features:")
for i, row in feature_scores.head(10).iterrows():
    feature_type = "Clinical" if row['feature'] in ['age', 'gender'] else "Gene"
    print(f"  {row['feature']} ({feature_type}): F-score = {row['f_score']:.2f}")

# Scale the selected features
X_scaled = scaler.fit_transform(X_selected)

# ============================
# MODEL SELECTION AND TRAINING
# ============================
print(f"\n🤖 Training multiple models optimized for gene expression...")

models = {
    'Logistic_Ridge': LogisticRegression(penalty='l2', C=1.0, random_state=RANDOM_STATE, max_iter=1000),
    'Logistic_Lasso': LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=RANDOM_STATE),
    'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)
}

# Cross-validation setup (essential for small datasets)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Evaluate each model
model_results = {}
print("\n📊 Cross-validation results:")
print("-" * 60)

for name, model in models.items():
    # Cross-validation scores
    cv_scores_auc = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    cv_scores_acc = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    
    model_results[name] = {
        'auc_mean': cv_scores_auc.mean(),
        'auc_std': cv_scores_auc.std(),
        'acc_mean': cv_scores_acc.mean(),
        'acc_std': cv_scores_acc.std()
    }
    
    print(f"{name}:")
    print(f"  AUC: {cv_scores_auc.mean():.3f} ± {cv_scores_auc.std():.3f}")
    print(f"  Accuracy: {cv_scores_acc.mean():.3f} ± {cv_scores_acc.std():.3f}")

# Select best model based on AUC
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_mean'])
best_model = models[best_model_name]
print(f"\n🏆 Best model: {best_model_name}")
print(f"    Cross-validated AUC: {model_results[best_model_name]['auc_mean']:.3f}")

# ============================
# FINAL MODEL TRAINING AND EVALUATION
# ============================
print(f"\n🎯 Final model evaluation with holdout test...")

# Train/test split (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set: {len(y_train)} samples")
print(f"Test set: {len(y_test)} samples")
print(f"Training class balance: {dict(pd.Series(y_train).value_counts())}")

# Train the best model
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
if len(np.unique(y)) > 1:
    auc_score = roc_auc_score(y_test, y_pred_proba)
else:
    auc_score = 0.5

print(f"\n📈 Final Model Performance on Test Set:")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC-ROC: {auc_score:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Poor Response', 'Good Response']))
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("  [Poor Response, Good Response]")

# ============================
# FEATURE IMPORTANCE ANALYSIS
# ============================
print(f"\n🧬 Most Important Features for Predicting Treatment Response:")
print("-" * 70)

if hasattr(best_model, 'coef_'):
    # For logistic regression
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'coefficient': best_model.coef_[0],
        'abs_coefficient': np.abs(best_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 15 Most Predictive Features (Logistic Regression Coefficients):")
    for i, row in feature_importance.head(15).iterrows():
        direction = "↑ Good Response" if row['coefficient'] > 0 else "↓ Poor Response" 
        feature_type = "Clinical" if row['feature'] in ['age', 'gender'] else "Gene"
        print(f"  {row['feature']} ({feature_type}): {row['coefficient']:+.3f} ({direction})")

elif hasattr(best_model, 'feature_importances_'):
    # For random forest
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 15 Most Important Features (Random Forest):")
    for i, row in feature_importance.head(15).iterrows():
        feature_type = "Clinical" if row['feature'] in ['age', 'gender'] else "Gene"
        print(f"  {row['feature']} ({feature_type}): {row['importance']:.4f}")

# ============================
# CLINICAL INSIGHTS
# ============================
print(f"\n🏥 CLINICAL INSIGHTS:")
print("=" * 50)

# Analyze clinical variables
clinical_vars = ['age', 'gender']
clinical_in_top = [feat for feat in feature_importance.head(10)['feature'] if feat in clinical_vars]

if clinical_in_top:
    print(f"Clinical variables in top 10 predictors: {clinical_in_top}")
else:
    print("Gene expression more predictive than clinical variables")

# Show PFS distribution by predicted response
print(f"\nPFS Distribution by Response Group:")
for response_val in [0, 1]:
    mask = df['response_binary'] == response_val
    group_pfs = df.loc[mask, 'pfs']
    response_label = "Poor Response" if response_val == 0 else "Good Response"
    print(f"  {response_label}: {group_pfs.mean():.0f} ± {group_pfs.std():.0f} days "
          f"(n={len(group_pfs)})")

# ============================
# SAVE RESULTS
# ============================
print(f"\n💾 Saving results...")

# Save model and preprocessing pipeline
model_package = {
    'model': best_model,
    'feature_selector': feature_selector,
    'scaler': scaler,
    'selected_features': list(selected_features),
    'model_name': best_model_name,
    'pfs_threshold_days': PFS_THRESHOLD_DAYS,
    'feature_names': list(X.columns)
}
joblib.dump(model_package, 'immunotherapy_response_predictor.pkl')

# Save predictions with patient info
results_df = pd.DataFrame({
    'sample_id': df.loc[X_test.index if hasattr(X_test, 'index') else range(len(y_test)), 'expression_id'].values,
    'actual_pfs_days': df.loc[X_test.index if hasattr(X_test, 'index') else range(len(y_test)), 'pfs'].values,
    'actual_response': y_test,
    'predicted_response': y_pred,
    'predicted_probability': y_pred_proba
})
results_df.to_csv('immunotherapy_predictions.csv', index=False)

# Save feature importance with gene annotations
feature_importance_full = feature_importance.copy()
feature_importance_full['feature_type'] = feature_importance_full['feature'].apply(
    lambda x: 'Clinical' if x in ['age', 'gender'] else 'Gene_Expression'
)
feature_importance_full.to_csv('predictive_biomarkers.csv', index=False)

# Save model comparison
model_comparison = pd.DataFrame(model_results).T
model_comparison.to_csv('model_performance_comparison.csv')

print(f"✅ Saved complete model -> immunotherapy_response_predictor.pkl")
print(f"✅ Saved predictions -> immunotherapy_predictions.csv")
print(f"✅ Saved biomarkers -> predictive_biomarkers.csv")
print(f"✅ Saved model comparison -> model_performance_comparison.csv")

# ============================
# FINAL SUMMARY
# ============================
print(f"\n🎯 IMMUNOTHERAPY RESPONSE PREDICTION SUMMARY:")
print("=" * 60)
print(f"Dataset: {len(df)} NSCLC patients treated with anti-PD-1/PD-L1")
print(f"Response definition: PFS > {PFS_THRESHOLD_MONTHS} months ({PFS_THRESHOLD_DAYS} days)")
print(f"Best model: {best_model_name}")
print(f"Cross-validated performance: {model_results[best_model_name]['auc_mean']:.3f} AUC")
print(f"Test set performance: {auc_score:.3f} AUC, {accuracy:.3f} Accuracy")
print(f"Selected {N_FEATURES_SELECT} most predictive features from {X.shape[1]} total")

# Performance interpretation
if model_results[best_model_name]['auc_mean'] >= 0.8:
    print("🎉 EXCELLENT: Strong predictive performance!")
elif model_results[best_model_name]['auc_mean'] >= 0.7:
    print("✅ GOOD: Solid predictive performance for biomarker discovery")
elif model_results[best_model_name]['auc_mean'] >= 0.6:
    print("⚠️  MODERATE: Some predictive signal, consider more data/features")
else:
    print("❌ LIMITED: Weak predictive signal with current approach")

print(f"\n📊 Use the saved files for:")
print(f"   • Clinical validation of predicted biomarkers")
print(f"   • Gene pathway enrichment analysis")
print(f"   • Prospective patient stratification")

print(f"\n🧬 Pipeline completed successfully! 🧬")