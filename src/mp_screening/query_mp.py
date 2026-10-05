import os
import pandas as pd
from mp_api.client import MPRester
from dotenv import load_dotenv

def query_stable_li_materials(limit: int = 100) -> pd.DataFrame:
    """
    Queries the Materials Project for thermodynamically stable lithium-containing 
    materials (energy_above_hull <= 0.05 eV/atom).
    Restricts to domains containing Oxygen or Sulfur to avoid out-of-domain errors.
    """
    load_dotenv()
    api_key = os.environ.get("MP_API_KEY")
    
    if not api_key:
        raise ValueError("MP_API_KEY environment variable not set. Please set it in a .env file.")
        
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Li"],
            energy_above_hull=(0.0, 0.05),
            fields=["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom", "elements"],
            num_chunks=1,
            chunk_size=500
        )
        
    data = []
    for doc in docs:
        el_symbols = [str(el) for el in doc.elements]
        # Domain Restriction: Must contain O or S
        if "O" in el_symbols or "S" in el_symbols:
            data.append({
                "material_id": str(doc.material_id),
                "formula_pretty": doc.formula_pretty,
                "energy_above_hull": doc.energy_above_hull,
                "formation_energy_per_atom": doc.formation_energy_per_atom
            })
            if limit and len(data) >= limit:
                break
                
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    load_dotenv()
    output_path = os.path.join("data", "raw", "mp_candidates.csv")
    
    try:
        # We limit to 50 for the test batch to prevent massive downloads right now
        df_candidates = query_stable_li_materials(limit=50)
        
        print(f"DataFrame Shape: {df_candidates.shape}")
        print("Data Preview (.head()):")
        print(df_candidates.head())
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to disk
        df_candidates.to_csv(output_path, index=False)
        print(f"Saved dataset to {output_path}")
    except Exception as e:
        print(f"Error querying MP: {e}")
