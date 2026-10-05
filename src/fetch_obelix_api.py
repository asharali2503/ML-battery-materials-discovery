import sys
import subprocess
import os
import datetime
import pandas as pd

def main():
    print("Installing obelix-data...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "obelix-data"])
    
    # Import OBELiX after installing
    from obelix import OBELiX
    
    print("Initializing OBELiX dataset...")
    ob = OBELiX()
    
    try:
        df_train = ob.train_dataset.dataframe
        df_test = ob.test_dataset.dataframe
    except AttributeError as e:
        print(f"AttributeError: {e}")
        print("Inspecting ob.train_dataset:")
        print(dir(ob.train_dataset))
        return

    print("Concatenating datasets...")
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "obelix_dataset.csv")
    prov_path = os.path.join(output_dir, "obelix_dataset.csv.provenance.txt")
    
    print(f"Saving dataset of shape {df_full.shape} to {csv_path}")
    df_full.to_csv(csv_path, index=False)
    
    utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prov_content = f"Source: Fetched via the official obelix-data PyPI package.\nDownload Timestamp (UTC): {utc_now}\n"
    
    with open(prov_path, "w", encoding="utf-8") as f:
        f.write(prov_content)
        
    print("Saved dataset and provenance successfully.")

if __name__ == "__main__":
    main()
