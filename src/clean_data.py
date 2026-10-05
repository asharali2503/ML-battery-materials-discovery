import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_obelix_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by converting conductivity to log10 space, 
    grouping by True Composition, dropping high-variance duplicates 
    (>2.0 diff in log space), and taking the median of the remaining.
    """
    df = raw_df.copy()
    
    # Step 1: Create log10_conductivity
    df['log10_conductivity'] = np.log10(df['Ionic conductivity (S cm-1)'])
    
    # Step 2: Group by True Composition
    grouped = df.groupby('True Composition')
    
    # Step 3 & 5: Calculate min, max, median, and size
    agg_df = grouped.agg(
        min_log=('log10_conductivity', 'min'),
        max_log=('log10_conductivity', 'max'),
        median_log=('log10_conductivity', 'median'),
        count=('log10_conductivity', 'size')
    )
    
    agg_df['diff'] = agg_df['max_log'] - agg_df['min_log']
    
    # Step 4: Filter out groups with difference > 2.0
    high_variance = agg_df[agg_df['diff'] > 2.0]
    dropped_comps = high_variance.index.tolist()
    
    if dropped_comps:
        logging.info(f"Dropped {len(dropped_comps)} compositions due to >100x variance: {dropped_comps}")
    else:
        logging.info("No compositions dropped due to high variance.")
        
    clean_agg = agg_df[agg_df['diff'] <= 2.0].copy()
    
    # Step 6: Create new dataframe with specified columns
    clean_df = clean_agg.reset_index()
    clean_df = clean_df.rename(columns={
        'median_log': 'log10_conductivity', 
        'count': 'measurement_count'
    })
    
    return clean_df[['True Composition', 'log10_conductivity', 'measurement_count']]

if __name__ == "__main__":
    input_path = os.path.join("data", "raw", "obelix_dataset.csv")
    output_path = os.path.join("data", "interim", "cleaned_dataset.csv")
    
    logging.info(f"Loading raw data from {input_path}")
    raw_df = pd.read_csv(input_path)
    
    cleaned_df = clean_obelix_data(raw_df)
    
    logging.info(f"Final cleaned dataset shape: {cleaned_df.shape}")
    logging.info(f"Saving to {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
