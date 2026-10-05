import argparse
import datetime
import logging
import os
import requests

# Configure logging to output to the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_dataset(url, filename):
    output_dir = os.path.join("data", "raw")
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    provenance_path = os.path.join(output_dir, f"{filename}.provenance.txt")
    
    # Warn before overwriting as per scientific reproducibility requirements
    if os.path.exists(output_path):
        logging.warning(f"File '{output_path}' already exists. It will be overwritten.")
    
    logging.info(f"Downloading data from: {url}")
    
    try:
        response = requests.get(url, timeout=60)
        
        # Raise RuntimeError for non-200 status codes
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch dataset. HTTP Status Code: {response.status_code}")
            
        logging.info("Download completed successfully. Saving to disk...")
        
        # Write dataset in binary mode to avoid encoding corruption
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        logging.info(f"Saved dataset to: {output_path}")
        
        # Write provenance metadata establishing chain of custody
        utc_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        provenance_content = f"Source URL: {url}\nDownload Timestamp (UTC): {utc_now}\n"
        
        with open(provenance_path, 'w', encoding='utf-8') as f:
            f.write(provenance_content)
            
        logging.info(f"Provenance metadata saved to: {provenance_path}")
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"A network error occurred while attempting to download the data: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw datasets securely and track their provenance.")
    parser.add_argument("--url", type=str, required=True, help="Public URL of the dataset.")
    parser.add_argument("--filename", type=str, required=True, help="Destination filename to be saved in data/raw/.")
    
    args = parser.parse_args()
    
    download_dataset(args.url, args.filename)
