import argparse
import os
import sys
import pandas as pd

def inspect_dataset(file_path):
    try:
        # 1. Dataset Provenance
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Load dataset
        df = pd.read_csv(file_path)
        
    except FileNotFoundError:
        print(f"Error: Dataset not found at path '{file_path}'. Please check the file path and try again.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty or not a valid CSV.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading the dataset: {e}")
        sys.exit(1)

    print("="*70)
    print("DATASET INSPECTION REPORT")
    print("="*70)
    
    print("\n--- 1. Dataset Provenance ---")
    print(f"File Path : {os.path.abspath(file_path)}")
    print(f"File Size : {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    
    # 2. Rows and Columns
    print("\n--- 2. Dataset Dimensions ---")
    print(f"Total Rows    : {df.shape[0]}")
    print(f"Total Columns : {df.shape[1]}")
    
    # 3, 4, 5. Column Details
    print("\n--- 3. Column Metadata ---")
    metadata = []
    for col in df.columns:
        dtype = df[col].dtype
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        metadata.append({
            "Column": col,
            "DType": str(dtype),
            "Missing Count": missing_count,
            "Missing %": f"{missing_pct:.2f}%",
            "Unique Count": unique_count
        })
        
    metadata_df = pd.DataFrame(metadata)
    # Print tabular metadata
    print(metadata_df.to_string(index=False))
    
    # 6. Potential Target Identification
    print("\n--- 4. Potential Target Identification ---")
    target_keywords = ['cond', 'sigma', 'temp', 'e_a', 'target']
    target_cols = [col for col in df.columns if any(kw in col.lower() for kw in target_keywords)]
    if target_cols:
        print(f"WARNING: Potential target variables detected: {', '.join(target_cols)}")
    else:
        print("No potential target variables detected based on keywords.")
        
    # 7. Potential Composition Identification
    print("\n--- 5. Potential Composition Identification ---")
    comp_keywords = ['comp', 'formula', 'material']
    comp_cols = [col for col in df.columns if any(kw in col.lower() for kw in comp_keywords)]
    if comp_cols:
        print(f"FLAG: Potential composition variables detected: {', '.join(comp_cols)}")
    else:
        print("No potential composition variables detected based on keywords.")
        
    print("\n" + "="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strictly inspect a raw dataset and generate a metadata report.")
    parser.add_argument("file_path", type=str, help="Path to the dataset CSV file")
    args = parser.parse_args()
    
    inspect_dataset(args.file_path)
