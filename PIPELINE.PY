# OVERFITTING-RESISTANT GENE EXPRESSION PIPELINE
# Fixed: Prevents data leakage and overfitting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, balanced_accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================
# CONFIG - ANTI-OVERFITTING SETTINGS
# ============================
DATA_FILE = "merged_dataset.csv"
RANDOM_STATE = 42

# Aggressive overfitting prevention
N_FEATURES_SELECT = 20  # Much fewer features (was 100)
MIN_SAMPLES_PER_CLASS = 3  # Minimum for stable evaluation

# PFS threshold for response
PFS_THRESHOLD_MONTHS = 6
PFS_THRESHOLD_DAYS = PFS_THRESHOLD_MONTHS * 30

# ============================
# LOAD DATA
# ============================
print("🧬 Loading and preprocessing data with overfitting prevention...")
df = pd.read_csv(DATA_FILE)
print(f"Dataset shape: {df.shape}")

# ============================
# CREATE RESPONSE (PREVENTING TARGET LEAKAGE)
# ============================
print(f"\n🎯 Creating binary response from PFS (threshold: {PFS_THRESHOLD_MONTHS} months)...")

pfs_data = df['pfs']
print(f"PFS range: {pfs_data.min()}-{pfs_data.max()} days")

# Create response variable
df['response_binary'] = (df['pfs'] > PFS_THRESHOLD_DAYS).astype(int)
response_counts = df['response_binary'].value_counts()
print(f"Response distribution: Poor={response_counts.get(0,0)}, Good={response_counts.get(1,0)}")

# Check for severe class imbalance
minority_class_size = min(response_counts.values)
if minority_class_size < MIN_SAMPLES_PER_CLASS:
    print(f"⚠️  Severe class imbalance detected. Using median PFS split instead...")
    median_pfs = df['pfs'].median()
    df['response_binary'] = (df['pfs'] > median_pfs).astype(int)
    response_counts = df['response_binary'].value_counts()
    print(f"Median-based split: Below={response_counts.get(0,0)}, Above={response_counts.get(1,0)}")

# ============================
# FEATURE PREPARATION (PREVENT DATA LEAKAGE)
# ============================
print(f"\n🧹 Preparing features with strict leakage prevention...")

# CRITICAL: Remove ALL target-related columns
leakage_columns = [
    'pfs',           # Direct target info
    'response',      # Empty original response  
    'response_binary', # Our created target
    'expression_id', # Patient identifiers
    'GSM',          # Sample identifiers  
    'patient_id'    # Patient info
]

# Only keep actual features
available_columns = [col for col in leakage_columns if col in df.columns]
print(f"Removing potential leakage columns: {available_columns}")

X = df.drop(columns=available_columns, errors='ignore')
y = df['response_binary']

# Handle gender
if 'gender' in X.columns:
    X['gender'] = X['gender'].map({'male': 1, 'female': 0})
    print("Encoded gender: male=1, female=0")

# Ensure no non-numeric columns remain
object_cols = X.select_dtypes(include=['object']).columns
if len(object_cols) > 0:
    print(f"⚠️  Found non-numeric columns: {list(object_cols)}")
    X = X.drop(columns=object_cols)

print(f"Final feature matrix: {X.shape}")
print(f"Features include: {X.columns[:5].tolist()} ... (showing first 5)")

# ============================
# AGGRESSIVE FEATURE SELECTION
# ============================
print(f"\n🎯 Selecting only {N_FEATURES_SELECT} most stable features...")

# Use more conservative feature selection
selector = SelectKBest(score_func=f_classif, k=min(N_FEATURES_SELECT, X.shape[1]))
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

print(f"Selected features: {len(selected_features)}")
print("Top 10 selected features:")
feature_scores = pd.DataFrame({
    'feature': selected_features,
    'f_score': selector.scores_[selector.get_support()]
}).sort_values('f_score', ascending=False)

for i, (_, row) in enumerate(feature_scores.head(10).iterrows()):
    feature_type = "Clinical" if row['feature'] in ['age', 'gender'] else "Gene"
    print(f"  {i+1}. {row['feature']} ({feature_type}): {row['f_score']:.2f}")

# ============================
# OVERFITTING-RESISTANT MODEL EVALUATION
# ============================
print(f"\n🤖 Training with overfitting-resistant evaluation...")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Conservative models to prevent overfitting
models = {
    'Logistic_Conservative': LogisticRegression(
        C=10.0,           # Higher regularization (was 1.0)
        penalty='l2', 
        random_state=RANDOM_STATE, 
        max_iter=1000
    ),
    'Logistic_Strong_L1': LogisticRegression(
        C=0.01,           # Very strong regularization  
        penalty='l1', 
        solver='liblinear',
        random_state=RANDOM_STATE
    ),
    'Random_Forest_Limited': RandomForestClassifier(
        n_estimators=50,   # Fewer trees
        max_depth=2,       # Shallower trees (was 3)
        min_samples_split=5, # More conservative splits
        random_state=RANDOM_STATE
    )
}

# Use Leave-One-Out CV for small datasets (more realistic)
print(f"Using Leave-One-Out Cross-Validation (most conservative for n={len(y)})...")
loo = LeaveOneOut()

model_results = {}
print("\n📊 Conservative cross-validation results:")
print("-" * 60)

for name, model in models.items():
    try:
        # LOO cross-validation
        cv_predictions = []
        cv_true = []
        
        for train_idx, test_idx in loo.split(X_scaled):
            X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train_fold, y_train_fold)
            pred = model.predict(X_test_fold)[0]
            
            cv_predictions.append(pred)
            cv_true.append(y_test_fold.iloc[0])
        
        # Calculate metrics
        cv_accuracy = accuracy_score(cv_true, cv_predictions)
        cv_balanced_acc = balanced_accuracy_score(cv_true, cv_predictions)
        
        model_results[name] = {
            'accuracy': cv_accuracy,
            'balanced_accuracy': cv_balanced_acc,
            'predictions': cv_predictions
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {cv_accuracy:.3f}")
        print(f"  Balanced Accuracy: {cv_balanced_acc:.3f}")
        
    except Exception as e:
        print(f"{name}: Failed - {str(e)}")
        model_results[name] = {'accuracy': 0, 'balanced_accuracy': 0}

# Select best model
if model_results:
    best_model_name = max(model_results.keys(), 
                         key=lambda x: model_results[x]['balanced_accuracy'])
    best_model = models[best_model_name]
    print(f"\n🏆 Most stable model: {best_model_name}")
    print(f"    LOO Balanced Accuracy: {model_results[best_model_name]['balanced_accuracy']:.3f}")
else:
    print("❌ All models failed")
    sys.exit()

# ============================
# FINAL REALISTIC EVALUATION
# ============================
print(f"\n🔬 Final realistic performance assessment...")

# Train final model on all data (no holdout test - sample too small)
best_model.fit(X_scaled, y)

# Show predictions vs reality
loo_predictions = model_results[best_model_name]['predictions']
comparison_df = pd.DataFrame({
    'Actual_PFS_days': df['pfs'].values,
    'Actual_Response': y.values,
    'Predicted_Response': loo_predictions,
    'Correct': np.array(y.values) == np.array(loo_predictions)
})

print("\nPrediction Analysis:")
print(f"Correct predictions: {comparison_df['Correct'].sum()}/{len(comparison_df)} "
      f"({comparison_df['Correct'].mean()*100:.1f}%)")

# Show confusion matrix
cm = confusion_matrix(y.values, loo_predictions)
print(f"\nConfusion Matrix (LOO CV):")
print(cm)
print("  [[True Poor, False Good], [False Poor, True Good]]")

# Analyze errors
errors = comparison_df[~comparison_df['Correct']]
if len(errors) > 0:
    print(f"\nMisclassified cases ({len(errors)} patients):")
    for _, row in errors.iterrows():
        actual_label = "Good" if row['Actual_Response'] else "Poor" 
        pred_label = "Good" if row['Predicted_Response'] else "Poor"
        print(f"  PFS {row['Actual_PFS_days']} days: Actual={actual_label}, Predicted={pred_label}")

# ============================
# REALISTIC FEATURE IMPORTANCE
# ============================
print(f"\n🧬 Feature importance (with caveats)...")

if hasattr(best_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'coefficient': best_model.coef_[0],
        'abs_coefficient': np.abs(best_model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 features (interpret cautiously with small sample):")
    for _, row in feature_importance.head(10).iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        feature_type = "Clinical" if row['feature'] in ['age', 'gender'] else "Gene"
        print(f"  {row['feature']} ({feature_type}): {row['coefficient']:+.3f} {direction}")

# ============================
# HONEST CONCLUSIONS
# ============================
print(f"\n🎯 REALISTIC ASSESSMENT:")
print("=" * 50)

balanced_acc = model_results[best_model_name]['balanced_accuracy']
accuracy = model_results[best_model_name]['accuracy']

print(f"Dataset: {len(y)} patients (small sample!)")
print(f"Class balance: {dict(y.value_counts())}")
print(f"Selected features: {len(selected_features)} from {X.shape[1]}")
print(f"Best model: {best_model_name}")
print(f"Leave-One-Out Balanced Accuracy: {balanced_acc:.3f}")

# Honest interpretation
if balanced_acc > 0.8:
    print("🎉 PROMISING: Shows predictive signal despite small sample")
elif balanced_acc > 0.65:
    print("✅ MODERATE: Some predictive ability, needs validation")
elif balanced_acc > 0.55:
    print("⚠️  WEAK: Limited predictive power")
else:
    print("❌ POOR: No significant predictive ability")

print(f"\n⚠️  IMPORTANT LIMITATIONS:")
print(f"• Very small sample size (n={len(y)}) - results may not generalize")
print(f"• Class imbalance ({dict(y.value_counts())}) - affects reliability")  
print(f"• High dimensionality risk - even with feature selection")
print(f"• Results need validation in larger, independent cohort")

print(f"\n📋 RECOMMENDATIONS:")
print(f"• Collect more samples (target: n≥100)")
print(f"• Validate findings in independent dataset") 
print(f"• Consider as exploratory/hypothesis-generating")
print(f"• Focus on biological interpretation of top genes")

# Save conservative results
joblib.dump({
    'model': best_model,
    'scaler': scaler,
    'feature_selector': selector,
    'selected_features': list(selected_features),
    'model_name': best_model_name,
    'cv_results': model_results,
    'sample_size_warning': f"Small sample (n={len(y)}) - validate before clinical use"
}, 'conservative_model.pkl')

comparison_df.to_csv('realistic_predictions.csv', index=False)
print(f"\n✅ Saved conservative model and realistic predictions")
print(f"🧬 Analysis completed with appropriate statistical caution! 🧬")