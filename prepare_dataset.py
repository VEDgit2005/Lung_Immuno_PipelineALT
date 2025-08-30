import os
import pandas as pd
import numpy as np
import glob

BASE_DIR = r"C:\Users\jenav\OneDrive\Documents\STUDY\lung_immuno_ALT"

# ----- Auto-detect files -----
print("🔍 Auto-detecting files...")

# Look for expression files (TSV or series matrix)
expr_candidates = []
expr_candidates.extend(glob.glob(os.path.join(BASE_DIR, "*exp*.tsv")))
expr_candidates.extend(glob.glob(os.path.join(BASE_DIR, "*series_matrix*.txt*")))
expr_candidates.extend(glob.glob(os.path.join(BASE_DIR, "*.tsv")))

print(f"Expression file candidates: {expr_candidates}")

# Look for clinical files
clinical_candidates = glob.glob(os.path.join(BASE_DIR, "*clinical*.csv"))
print(f"Clinical file candidates: {clinical_candidates}")

if not expr_candidates:
    print("❌ No expression files found. Looking for files in directory...")
    all_files = os.listdir(BASE_DIR)
    print(f"All files: {all_files}")
    raise FileNotFoundError("No expression files found")

if not clinical_candidates:
    print("❌ No clinical files found")
    raise FileNotFoundError("No clinical files found")

# Use the first candidate
expr_path = expr_candidates[0]
clinical_path = clinical_candidates[0]

print(f"✅ Using expression file: {os.path.basename(expr_path)}")
print(f"✅ Using clinical file: {os.path.basename(clinical_path)}")

# ----- Load expression data -----
print("\n📊 Loading expression data...")
try:
    # Try TSV format first (for GSE135222_GEO_RNA-seq_omicslab_exp.tsv)
    if expr_path.endswith('.tsv'):
        expr = pd.read_csv(expr_path, sep="\t", index_col=0)
    else:
        # Try series matrix format
        expr = pd.read_csv(expr_path, sep="\t", comment="!", index_col=0)
    
    print(f"Expression shape (genes x samples): {expr.shape}")
except Exception as e:
    print(f"❌ Error loading expression: {e}")
    # Try different approaches
    print("Trying alternative loading methods...")
    expr = pd.read_csv(expr_path, sep="\t", index_col=0, low_memory=False)
    print(f"Expression shape: {expr.shape}")

# ----- Load clinical data -----
print("\n📋 Loading clinical data...")
clinical = pd.read_csv(clinical_path)
print(f"Clinical shape: {clinical.shape}")
print(f"Clinical columns: {list(clinical.columns)}")

# Auto-detect GSM column in clinical data
gsm_col = None
for col in clinical.columns:
    if 'gsm' in col.lower() or clinical[col].astype(str).str.contains('GSM', case=False).any():
        gsm_col = col
        break

if gsm_col is None:
    # Use first column as fallback
    gsm_col = clinical.columns[0]
    print(f"⚠️ No GSM column detected, using first column: {gsm_col}")

if gsm_col != "GSM":
    clinical = clinical.rename(columns={gsm_col: "GSM"})

# ----- Clean and align sample IDs -----
print("\n🔧 Cleaning sample IDs...")

# Clean expression column names (remove quotes, whitespace, prefixes)
expr_cols_raw = expr.columns.astype(str)
print(f"Raw expression columns (first 5): {list(expr_cols_raw[:5])}")

# Clean expression columns (basic cleaning first)
expr_cols_clean = (expr_cols_raw
                  .str.replace('"', '')  # Remove quotes
                  .str.replace("'", '')  # Remove single quotes
                  .str.strip())          # Remove whitespace

expr.columns = expr_cols_clean

# Clean clinical GSM column
clinical_gsms_raw = clinical["GSM"].astype(str)
print(f"Raw clinical GSMs (first 5): {list(clinical_gsms_raw[:5])}")

clinical_gsms_clean = (clinical_gsms_raw
                      .str.replace('"', '')
                      .str.replace("'", '')
                      .str.strip())

clinical["GSM"] = clinical_gsms_clean

# Check if we need to map between patient IDs and GSM IDs
print(f"\n🔍 Checking ID types...")
print(f"Expression columns look like patient IDs: {list(expr.columns[:5])}")
print(f"Clinical has GSM column: {list(clinical['GSM'][:5])}")

# Check if clinical has a patient_id column that might match expression columns
if 'patient_id' in clinical.columns:
    print(f"Clinical patient_ids: {list(clinical['patient_id'][:5])}")
    
    # Try to match expression columns to clinical patient_id instead of GSM
    expr_ids = set(expr.columns)
    patient_ids = set(clinical['patient_id'].astype(str).str.strip())
    
    print(f"Checking overlap with patient_id column...")
    common_patients = expr_ids.intersection(patient_ids)
    print(f"Common patient IDs: {len(common_patients)}")
    
    if len(common_patients) > 0:
        print("✅ Found matches using patient_id! Using patient_id for alignment...")
        # Use patient_id for alignment instead of GSM
        clinical["SAMPLE_ID"] = clinical["patient_id"]
        expr_sample_col = "SAMPLE_ID"
    else:
        print("❌ No matches with patient_id either")
        print("🔄 Trying positional mapping (assuming same order)...")
        
        # Check if we have same number of samples
        if len(expr.columns) == len(clinical):
            print(f"Both have {len(expr.columns)} samples - creating positional mapping")
            
            # Create a mapping dataframe
            mapping_df = pd.DataFrame({
                'expression_id': expr.columns,
                'gsm_id': clinical['GSM'].values,
                'order': range(len(expr.columns))
            })
            
            print("Sample mapping (first 10):")
            print(mapping_df.head(10))
            
            # Add expression IDs to clinical data for merging
            clinical['expression_id'] = clinical['GSM'].map(
                dict(zip(clinical['GSM'], mapping_df['expression_id']))
            )
            
            expr_sample_col = "expression_id"
            print("✅ Using positional mapping between GSM and expression IDs")
        else:
            print(f"❌ Different number of samples: expr={len(expr.columns)}, clinical={len(clinical)}")
            expr_sample_col = "GSM"
else:
    print("❌ No patient_id column found")
    print("🔄 Trying positional mapping (assuming same order)...")
    
    # Check if we have same number of samples
    if len(expr.columns) == len(clinical):
        print(f"Both have {len(expr.columns)} samples - creating positional mapping")
        
        # Create a mapping dataframe  
        mapping_df = pd.DataFrame({
            'expression_id': expr.columns,
            'gsm_id': clinical['GSM'].values,
            'order': range(len(expr.columns))
        })
        
        print("Sample mapping (first 10):")
        print(mapping_df.head(10))
        
        # Add expression IDs to clinical data for merging
        clinical['expression_id'] = mapping_df['expression_id'].values
        
        expr_sample_col = "expression_id"
        print("✅ Using positional mapping between GSM and expression IDs")
    else:
        print(f"❌ Different number of samples: expr={len(expr.columns)}, clinical={len(clinical)}")
        expr_sample_col = "GSM"

# ----- Debug: show cleaned IDs -----
print(f"\n🔍 After cleaning:")
print(f"Expression GSMs (first 10): {list(expr.columns[:10])}")
print(f"Clinical GSMs (first 10): {list(clinical['GSM'][:10])}")

# ----- Find common samples -----
if expr_sample_col == "SAMPLE_ID":
    # Using patient_id for matching
    expr_ids = set(expr.columns)
    clinical_ids = set(clinical["SAMPLE_ID"].astype(str).str.strip())
    common_ids = expr_ids.intersection(clinical_ids)
    id_type = "patient_id"
elif expr_sample_col == "expression_id":
    # Using positional mapping
    expr_ids = set(expr.columns)
    clinical_ids = set(clinical["expression_id"].astype(str).str.strip())
    common_ids = expr_ids.intersection(clinical_ids)
    id_type = "expression_id (positional mapping)"
else:
    # Using GSM for matching  
    expr_ids = set(expr.columns)
    clinical_ids = set(clinical["GSM"].astype(str).str.strip())
    common_ids = expr_ids.intersection(clinical_ids)
    id_type = "GSM"

print(f"\n📊 Sample overlap using {id_type}:")
print(f"Expression samples: {len(expr_ids)}")
print(f"Clinical samples: {len(clinical_ids)}")
print(f"Common samples: {len(common_ids)}")

if len(common_ids) == 0:
    print("❌ Still no overlap! Let's debug further...")
    print("Expression IDs:", sorted(list(expr_ids))[:10])
    print("Clinical IDs:", sorted(list(clinical_ids))[:10])
    
    # Try partial matching
    print("\n🔄 Attempting partial matching...")
    matches = []
    for c_id in clinical_ids:
        for e_id in expr_ids:
            if str(c_id) in str(e_id) or str(e_id) in str(c_id):
                matches.append((c_id, e_id))
    
    if matches:
        print(f"Found {len(matches)} potential partial matches:")
        for match in matches[:10]:
            print(f"  Clinical: {match[0]} <-> Expression: {match[1]}")
    
    raise RuntimeError(f"No overlapping {id_type} IDs found even after cleaning")

# ----- Subset to common samples -----
print(f"\n✅ Proceeding with {len(common_ids)} common samples using {id_type}")

# Sort for reproducibility
common_ids_sorted = sorted(common_ids)

# Subset expression data
expr_aligned = expr[common_ids_sorted]

# Subset clinical data based on the ID type being used
if expr_sample_col == "SAMPLE_ID":
    clinical_aligned = clinical[clinical["SAMPLE_ID"].isin(common_ids_sorted)].copy()
    clinical_aligned = clinical_aligned.sort_values("SAMPLE_ID").reset_index(drop=True)
    merge_col = "SAMPLE_ID"
elif expr_sample_col == "expression_id":
    clinical_aligned = clinical[clinical["expression_id"].isin(common_ids_sorted)].copy()
    clinical_aligned = clinical_aligned.sort_values("expression_id").reset_index(drop=True)
    merge_col = "expression_id"
else:
    clinical_aligned = clinical[clinical["GSM"].isin(common_ids_sorted)].copy() 
    clinical_aligned = clinical_aligned.sort_values("GSM").reset_index(drop=True)
    merge_col = "GSM"

print(f"Aligned expression shape: {expr_aligned.shape}")
print(f"Aligned clinical shape: {clinical_aligned.shape}")

# ----- Save aligned data -----
print("\n💾 Saving aligned datasets...")

# Save standalone aligned files
expr_aligned.to_csv(os.path.join(BASE_DIR, "expression_aligned.csv"))
clinical_aligned.to_csv(os.path.join(BASE_DIR, "clinical_aligned.csv"), index=False)

# ----- Create merged dataset (samples as rows) -----
print("\n🔄 Creating merged dataset...")

# Transpose expression (samples as rows, genes as columns)
expr_T = expr_aligned.T
expr_T.index.name = merge_col
expr_T = expr_T.reset_index()

# Merge with clinical data
merged = pd.merge(expr_T, clinical_aligned, on=merge_col, how="inner")
print(f"Merged dataset shape: {merged.shape}")

# Save merged dataset
merged.to_csv(os.path.join(BASE_DIR, "merged_dataset.csv"), index=False)

# ----- Create ML-ready dataset -----
print("\n🎯 Creating ML-ready dataset...")

# Look for response column
response_cols = [col for col in merged.columns if 'response' in col.lower()]
print(f"Potential response columns: {response_cols}")

if response_cols:
    response_col = response_cols[0]
    print(f"Using response column: {response_col}")
    
    # Convert to numeric and create binary target
    merged[response_col] = pd.to_numeric(merged[response_col], errors='coerce')
    merged["response_binary"] = (merged[response_col] > 0).astype(int)
    
    print(f"Response distribution: {merged['response_binary'].value_counts()}")
    
    # Create final dataset for ML (drop non-feature columns)
    non_feature_cols = [merge_col, "GSM", "patient_id", "gender", "age", "pfs", response_col]
    non_feature_cols = [col for col in non_feature_cols if col in merged.columns]
    
    ml_dataset = merged.drop(columns=non_feature_cols)
    ml_dataset.to_csv(os.path.join(BASE_DIR, "dataset_final.csv"), index=False)
    
    print(f"✅ Final ML dataset shape: {ml_dataset.shape}")
    print(f"   Features: {ml_dataset.shape[1] - 1}")
    print(f"   Samples: {ml_dataset.shape[0]}")
    print(f"   Target: response_binary")
else:
    print("⚠️ No response column found - saved merged dataset only")

print("\n🎉 Dataset preparation complete!")
print("\nFiles created:")
print("  📄 expression_aligned.csv - Aligned gene expression")
print("  📄 clinical_aligned.csv - Aligned clinical data") 
print("  📄 merged_dataset.csv - Combined dataset")
if response_cols:
    print("  📄 dataset_final.csv - ML-ready dataset")

print(f"\n💡 Next steps:")
print(f"   With {len(common_ids)} samples, consider:")
print(f"   • Feature selection (top variable genes)")
print(f"   • PCA + simple classifier") 
print(f"   • Cross-validation for robust evaluation")
print(f"   • Avoid complex models (risk of overfitting)")