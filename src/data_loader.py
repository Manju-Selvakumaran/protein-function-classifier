"""
Data loader for fetching protein sequences from UniProt.
Fetches enzyme sequences with EC (Enzyme Commission) annotations.
"""

import requests
import pandas as pd
import time
import os
from tqdm import tqdm


def fetch_uniprot_enzymes(ec_class, limit=1000, reviewed_only=True):
    """
    Fetch protein sequences for a specific EC class from UniProt.
    
    Parameters
    ----------
    ec_class : int
        EC class number (1-7)
    limit : int
        Maximum number of sequences to fetch
    reviewed_only : bool
        If True, fetch only Swiss-Prot (reviewed) entries
    
    Returns
    -------
    pd.DataFrame
        DataFrame with protein data
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    
    # Build query
    review_filter = "reviewed:true" if reviewed_only else "*"
    query = f"({review_filter}) AND (ec:{ec_class}.*)"
    
    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,protein_name,ec,sequence,length,organism_name",
        "size": min(limit, 500)
    }
    
    all_results = []
    cursor = None
    fetched = 0
    header = None
    
    print(f"Fetching EC class {ec_class} enzymes...")
    
    with tqdm(total=limit, desc=f"EC {ec_class}") as pbar:
        while fetched < limit:
            if cursor:
                params["cursor"] = cursor
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    break
                
                lines = response.text.strip().split('\n')
                
                if len(lines) <= 1:
                    break
                
                if header is None:
                    header = lines[0].split('\t')
                
                for line in lines[1:]:
                    if fetched >= limit:
                        break
                    all_results.append(line.split('\t'))
                    fetched += 1
                    pbar.update(1)
                
                link_header = response.headers.get("Link", "")
                if 'rel="next"' in link_header:
                    try:
                        cursor = link_header.split('cursor=')[1].split('>')[0].split('&')[0]
                    except IndexError:
                        break
                else:
                    break
                
                time.sleep(0.5)
                
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                break
    
    if all_results and header:
        df = pd.DataFrame(all_results, columns=header)
        df['ec_class'] = ec_class
        return df
    
    return pd.DataFrame()


def fetch_all_ec_classes(samples_per_class=800, save_path=None):
    """
    Fetch enzyme sequences for all 7 EC classes.
    
    Parameters
    ----------
    samples_per_class : int
        Number of samples to fetch per EC class
    save_path : str
        Path to save the combined dataset
    
    Returns
    -------
    pd.DataFrame
        Combined dataset with all EC classes
    """
    if save_path is None:
        save_path = os.path.join("data", "raw", "uniprot_enzymes.csv")
    
    ec_classes = {
        1: "Oxidoreductases",
        2: "Transferases",
        3: "Hydrolases",
        4: "Lyases",
        5: "Isomerases",
        6: "Ligases",
        7: "Translocases"
    }
    
    all_data = []
    
    print("=" * 50)
    print("Fetching protein sequences from UniProt")
    print("=" * 50)
    
    for ec_num, ec_name in ec_classes.items():
        print(f"\nEC {ec_num}: {ec_name}")
        df = fetch_uniprot_enzymes(ec_num, limit=samples_per_class)
        
        if not df.empty:
            df['ec_name'] = ec_name
            all_data.append(df)
            print(f"  -> Fetched {len(df)} sequences")
        else:
            print(f"  -> Warning: No data fetched for EC {ec_num}")
        
        time.sleep(1)
    
    if not all_data:
        print("Error: No data was fetched!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = clean_dataset(combined_df)
    
    # Save to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_df.to_csv(save_path, index=False)
    print(f"\n✓ Saved {len(combined_df)} sequences to {save_path}")
    
    return combined_df


def clean_dataset(df):
    """
    Clean and preprocess the dataset.
    """
    print("\nCleaning dataset...")
    initial_size = len(df)
    
    # Rename columns
    column_mapping = {
        'Entry': 'accession',
        'Protein names': 'protein_name',
        'EC number': 'ec_number',
        'Sequence': 'sequence',
        'Length': 'length',
        'Organism': 'organism'
    }
    df = df.rename(columns=column_mapping)
    
    # Remove missing sequences
    df = df.dropna(subset=['sequence'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['sequence'])
    
    # Convert length to numeric
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    
    # Filter by length
    df = df[(df['length'] >= 50) & (df['length'] <= 2000)]
    
    # Remove non-standard amino acids
    standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
    df = df[df['sequence'].apply(lambda x: set(str(x).upper()).issubset(standard_aa))]
    
    # Uppercase sequences
    df['sequence'] = df['sequence'].str.upper()
    
    print(f"  -> Removed {initial_size - len(df)} invalid/duplicate entries")
    print(f"  -> Final dataset size: {len(df)} sequences")
    
    return df


def load_dataset(path=None):
    """Load the dataset from disk."""
    if path is None:
        path = os.path.join("data", "raw", "uniprot_enzymes.csv")
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run fetch_all_ec_classes() first."
        )
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} sequences from {path}")
    return df


def get_dataset_summary(df):
    """Print dataset summary."""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    
    print(f"\nTotal sequences: {len(df)}")
    
    if 'organism' in df.columns:
        print(f"Unique organisms: {df['organism'].nunique()}")
    
    print("\nClass distribution:")
    class_counts = df.groupby(['ec_class', 'ec_name']).size()
    for (ec_class, ec_name), count in class_counts.items():
        print(f"  EC {ec_class} ({ec_name}): {count}")
    
    print(f"\nSequence length statistics:")
    print(f"  Min: {df['length'].min()}")
    print(f"  Max: {df['length'].max()}")
    print(f"  Mean: {df['length'].mean():.1f}")
    print(f"  Median: {df['length'].median():.1f}")


if __name__ == "__main__":
    df = fetch_all_ec_classes(samples_per_class=800)
    if not df.empty:
        get_dataset_summary(df)