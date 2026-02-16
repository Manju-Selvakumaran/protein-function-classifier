"""
Quick test script to verify project setup is working correctly.
"""

import sys
import os


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    
    packages = [
        "pandas",
        "numpy", 
        "sklearn",
        "xgboost",
        "matplotlib",
        "seaborn",
        "requests",
        "tqdm",
        "streamlit",
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"  [OK] {package}")
        except ImportError as e:
            print(f"  [FAIL] {package} - {e}")
            failed.append(package)
    
    return len(failed) == 0


def test_project_structure():
    """Verify project directory structure."""
    print("\nChecking project structure...")
    
    required_dirs = [
        "data",
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        "notebooks",
        "src",
        "models",
        "figures"
    ]
    
    required_files = [
        os.path.join("src", "__init__.py"),
        os.path.join("src", "data_loader.py"),
        "requirements.txt"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  [OK] {dir_path}\\")
        else:
            print(f"  [MISSING] {dir_path}\\")
            all_good = False
    
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  [OK] {file_path}")
        else:
            print(f"  [MISSING] {file_path}")
            all_good = False
    
    return all_good


def test_data_loader():
    """Test that our data loader module works."""
    print("\nTesting data loader module...")
    
    try:
        from src.data_loader import fetch_uniprot_enzymes
        print("  [OK] data_loader imports successfully")
        
        print("  Testing API connection (fetching 5 sequences)...")
        df = fetch_uniprot_enzymes(ec_class=1, limit=5)
        
        if len(df) > 0:
            print(f"  [OK] Successfully fetched {len(df)} sequences")
            return True
        else:
            print("  [FAIL] No data returned from API")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    print("=" * 50)
    print("PROTEIN FUNCTION CLASSIFIER - SETUP TEST")
    print("=" * 50)
    print(f"Working directory: {os.getcwd()}")
    print()
    
    results = {
        "Package Imports": test_imports(),
        "Project Structure": test_project_structure(),
        "Data Loader": test_data_loader(),
    }
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[PASSED]" if passed else "[FAILED]"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n" + "=" * 50)
        print("SUCCESS! All tests passed!")
        print("=" * 50)
        print("\nNext steps:")
        print("  1. Run: python src\\data_loader.py")
        print("     (This will download ~5000 protein sequences)")
        print("  2. Wait for download to complete (~5-10 minutes)")
    else:
        print("\n[WARNING] Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())