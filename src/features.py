"""
Feature extraction module for protein sequences.
Extracts various features used for enzyme classification.
"""

import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm


# Standard amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# Physicochemical properties of amino acids
# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

# Molecular weight of amino acids (Daltons)
MOLECULAR_WEIGHT = {
    'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
    'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
    'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
    'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2
}

# Amino acid polarity (1 = polar, 0 = nonpolar)
POLARITY = {
    'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
    'G': 0, 'H': 1, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 1, 'P': 0, 'Q': 1, 'R': 1,
    'S': 1, 'T': 1, 'V': 0, 'W': 0, 'Y': 1
}

# Amino acid charge at pH 7 (1 = positive, -1 = negative, 0 = neutral)
CHARGE = {
    'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
    'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
    'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
    'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# Secondary structure propensity (Chou-Fasman)
# H = helix, E = sheet, C = coil tendency
HELIX_PROPENSITY = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69
}

SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47
}


def compute_amino_acid_composition(sequence):
    """
    Calculate the frequency of each amino acid (20 features).
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of 20 amino acid frequencies
    """
    length = len(sequence)
    if length == 0:
        return [0.0] * 20
    
    return [sequence.count(aa) / length for aa in AMINO_ACIDS]


def compute_dipeptide_composition(sequence):
    """
    Calculate the frequency of each dipeptide pair (400 features).
    Dipeptides capture local sequence patterns.
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of 400 dipeptide frequencies
    """
    dipeptides = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
    
    total = len(sequence) - 1
    if total <= 0:
        return [0.0] * 400
    
    # Count all dipeptides
    dipep_counts = Counter(sequence[i:i+2] for i in range(total))
    
    return [dipep_counts.get(dp, 0) / total for dp in dipeptides]


def compute_physicochemical_properties(sequence):
    """
    Calculate global physicochemical properties (10 features).
    
    Features:
    - Average hydrophobicity
    - Hydrophobicity std
    - Average molecular weight
    - Total molecular weight
    - Fraction polar residues
    - Fraction nonpolar residues
    - Net charge
    - Fraction positive charged
    - Fraction negative charged
    - Sequence length (normalized)
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of 10 physicochemical features
    """
    length = len(sequence)
    if length == 0:
        return [0.0] * 10
    
    # Hydrophobicity
    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    avg_hydro = np.mean(hydro_values)
    std_hydro = np.std(hydro_values)
    
    # Molecular weight
    mw_values = [MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence]
    avg_mw = np.mean(mw_values)
    total_mw = sum(mw_values)
    
    # Polarity
    polar_count = sum(POLARITY.get(aa, 0) for aa in sequence)
    frac_polar = polar_count / length
    frac_nonpolar = 1 - frac_polar
    
    # Charge
    charges = [CHARGE.get(aa, 0) for aa in sequence]
    net_charge = sum(charges)
    frac_positive = sum(1 for c in charges if c > 0) / length
    frac_negative = sum(1 for c in charges if c < 0) / length
    
    # Normalized length (log scale to handle varying lengths)
    norm_length = np.log10(length)
    
    return [
        avg_hydro,
        std_hydro,
        avg_mw,
        total_mw / 10000,  # Scale down
        frac_polar,
        frac_nonpolar,
        net_charge / length,  # Normalize by length
        frac_positive,
        frac_negative,
        norm_length
    ]


def compute_secondary_structure_propensity(sequence):
    """
    Calculate secondary structure propensity features (4 features).
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of 4 secondary structure features
    """
    length = len(sequence)
    if length == 0:
        return [0.0] * 4
    
    helix_values = [HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence]
    sheet_values = [SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence]
    
    return [
        np.mean(helix_values),
        np.std(helix_values),
        np.mean(sheet_values),
        np.std(sheet_values)
    ]


def compute_sequence_complexity(sequence):
    """
    Calculate sequence complexity features (3 features).
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of 3 complexity features
    """
    length = len(sequence)
    if length == 0:
        return [0.0] * 3
    
    # Unique amino acids used
    unique_aa = len(set(sequence))
    frac_unique = unique_aa / 20  # Out of 20 possible
    
    # Shannon entropy (sequence complexity)
    aa_counts = Counter(sequence)
    probs = [count / length for count in aa_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(20)  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy
    
    # Repetitiveness (fraction of repeated dipeptides)
    if length > 1:
        dipeptides = [sequence[i:i+2] for i in range(length - 1)]
        dipep_counts = Counter(dipeptides)
        repeated = sum(1 for c in dipep_counts.values() if c > 1)
        frac_repeated = repeated / len(dipep_counts) if dipep_counts else 0
    else:
        frac_repeated = 0
    
    return [frac_unique, normalized_entropy, frac_repeated]


def extract_all_features(sequence):
    """
    Extract all features from a protein sequence.
    
    Total features: 20 + 400 + 10 + 4 + 3 = 437 features
    
    Parameters
    ----------
    sequence : str
        Protein sequence
    
    Returns
    -------
    list
        List of all features
    """
    features = []
    
    # Amino acid composition (20 features)
    features.extend(compute_amino_acid_composition(sequence))
    
    # Dipeptide composition (400 features)
    features.extend(compute_dipeptide_composition(sequence))
    
    # Physicochemical properties (10 features)
    features.extend(compute_physicochemical_properties(sequence))
    
    # Secondary structure propensity (4 features)
    features.extend(compute_secondary_structure_propensity(sequence))
    
    # Sequence complexity (3 features)
    features.extend(compute_sequence_complexity(sequence))
    
    return features


def get_feature_names():
    """
    Get names for all features.
    
    Returns
    -------
    list
        List of feature names
    """
    names = []
    
    # Amino acid composition
    names.extend([f'AAC_{aa}' for aa in AMINO_ACIDS])
    
    # Dipeptide composition
    names.extend([f'DPC_{a1}{a2}' for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS])
    
    # Physicochemical properties
    physchem_names = [
        'avg_hydrophobicity', 'std_hydrophobicity',
        'avg_molecular_weight', 'total_molecular_weight',
        'frac_polar', 'frac_nonpolar',
        'net_charge', 'frac_positive', 'frac_negative',
        'log_length'
    ]
    names.extend(physchem_names)
    
    # Secondary structure propensity
    ss_names = ['avg_helix_prop', 'std_helix_prop', 'avg_sheet_prop', 'std_sheet_prop']
    names.extend(ss_names)
    
    # Sequence complexity
    complexity_names = ['frac_unique_aa', 'sequence_entropy', 'dipeptide_repetitiveness']
    names.extend(complexity_names)
    
    return names


def process_sequences(sequences, show_progress=True):
    """
    Extract features from multiple sequences.
    
    Parameters
    ----------
    sequences : list or pd.Series
        List of protein sequences
    show_progress : bool
        Whether to show progress bar
    
    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, n_features)
    """
    if show_progress:
        iterator = tqdm(sequences, desc="Extracting features")
    else:
        iterator = sequences
    
    features_list = [extract_all_features(seq) for seq in iterator]
    
    return np.array(features_list)


def create_feature_dataframe(df, sequence_column='sequence'):
    """
    Create a DataFrame with all extracted features.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing sequences
    sequence_column : str
        Name of column containing sequences
    
    Returns
    -------
    pd.DataFrame
        DataFrame with features
    """
    print("Extracting features from sequences...")
    print(f"  - Amino acid composition: 20 features")
    print(f"  - Dipeptide composition: 400 features")
    print(f"  - Physicochemical properties: 10 features")
    print(f"  - Secondary structure propensity: 4 features")
    print(f"  - Sequence complexity: 3 features")
    print(f"  - Total: 437 features")
    print()
    
    # Extract features
    feature_matrix = process_sequences(df[sequence_column])
    
    # Create DataFrame
    feature_names = get_feature_names()
    feature_df = pd.DataFrame(feature_matrix, columns=feature_names)
    
    print(f"\n Extracted {feature_df.shape[1]} features from {feature_df.shape[0]} sequences")
    
    return feature_df


# Main execution for testing
if __name__ == "__main__":
    # Test with a sample sequence
    test_sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"
    
    print("Testing feature extraction...")
    print(f"Test sequence length: {len(test_sequence)}")
    
    features = extract_all_features(test_sequence)
    names = get_feature_names()
    
    print(f"\nTotal features extracted: {len(features)}")
    print(f"Feature names count: {len(names)}")
    
    print("\nSample features:")
    for name, value in zip(names[:10], features[:10]):
        print(f"  {name}: {value:.4f}")
    
    print("\n Feature extraction module is working")