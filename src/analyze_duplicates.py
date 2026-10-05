import pandas as pd
import numpy as np

def analyze_duplicates():
    file_path = "data/raw/obelix_dataset.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
        
    # Identify duplicate True Compositions
    comp_counts = df['True Composition'].value_counts()
    dup_comps = comp_counts[comp_counts > 1].index
    
    df_dups = df[df['True Composition'].isin(dup_comps)].copy()
    
    if df_dups.empty:
        print("No duplicates found for True Composition.")
        return
        
    print("="*70)
    print("DUPLICATE COMPOSITION ANALYSIS REPORT")
    print("="*70)
    
    # 1. Polymorph Check
    polymorph_count = 0
    grouped = df_dups.groupby('True Composition')
    
    for comp, group in grouped:
        if group['Space group'].nunique(dropna=False) > 1:
            polymorph_count += 1
            
    print("\n--- 1. Polymorph Check ---")
    print(f"Number of duplicate composition groups with >1 unique Space group: {polymorph_count}")
    print(f"Total duplicate composition groups: {len(dup_comps)}")
    
    # 2. Inter-lab Variance Check
    most_duped_comp = comp_counts.index[0]
    most_duped_count = comp_counts.iloc[0]
    
    print(f"\n--- 2. Inter-lab Variance Check ---")
    print(f"Most duplicated composition: {most_duped_comp} (Occurrences: {most_duped_count})")
    
    cols_to_show = ['True Composition', 'Ionic conductivity (S cm-1)', 'Space group', 'Family', 'DOI']
    most_duped_df = df[df['True Composition'] == most_duped_comp][cols_to_show]
    print(most_duped_df.to_string(index=False))
    
    # 3. Conductivity Variance
    print("\n--- 3. Conductivity Variance (Max/Min Ratio) ---")
    ratios = []
    
    for comp, group in grouped:
        conds = group['Ionic conductivity (S cm-1)']
        min_cond = conds.min()
        max_cond = conds.max()
        
        # Add epsilon to prevent division by zero
        ratio = max_cond / (min_cond + 1e-10)
        ratios.append({
            'True Composition': comp,
            'Min Cond.': min_cond,
            'Max Cond.': max_cond,
            'Max/Min Ratio': ratio
        })
        
    ratios_df = pd.DataFrame(ratios).sort_values(by='Max/Min Ratio', ascending=False).head(3)
    
    print("Top 3 compositions with the most extreme conductivity variance:")
    print(ratios_df.to_string(index=False))
    print("\n" + "="*70)

if __name__ == "__main__":
    analyze_duplicates()
