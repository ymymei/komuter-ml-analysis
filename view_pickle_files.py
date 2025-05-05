"""
Utility script to view the contents of pickle files in the KomuterPulse project.
This script helps you explore pickle files which cannot be directly opened in VS Code.
"""

import pickle
from pathlib import Path
import pandas as pd
import pprint

# Set up pretty printer for better output formatting
pp = pprint.PrettyPrinter(indent=4)

def view_pickle(file_path):
    """
    Load and display the contents of a pickle file
    """
    print(f"\n{'=' * 50}")
    print(f"EXAMINING PICKLE FILE: {file_path}")
    print(f"{'=' * 50}")
    
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            
        print("\nTYPE OF DATA:")
        print(f"  {type(data)}")
        
        # Handle different data types appropriately
        if isinstance(data, dict):
            print("\nKEYS IN DICTIONARY:")
            for key in data.keys():
                print(f"  - {key} ({type(data[key])})")
            
            # Ask which key to explore
            print("\nDo you want to explore a specific key? Enter key name or press Enter to skip:")
            key_name = input("> ")
            if key_name and key_name in data:
                print(f"\nCONTENTS OF KEY '{key_name}':")
                
                # Handle different value types
                if isinstance(data[key_name], pd.DataFrame):
                    print("\nDataFrame contents:")
                    print(data[key_name])
                elif isinstance(data[key_name], list) and len(data[key_name]) > 20:
                    print(f"List with {len(data[key_name])} items. First 10 items:")
                    pp.pprint(data[key_name][:10])
                    print("...")
                else:
                    pp.pprint(data[key_name])
        
        # Handle list data
        elif isinstance(data, list):
            print(f"\nLIST LENGTH: {len(data)}")
            print("\nSAMPLE CONTENTS (first 10 items):")
            if len(data) > 10:
                pp.pprint(data[:10])
                print("...")
            else:
                pp.pprint(data)
        
        # Handle DataFrame
        elif isinstance(data, pd.DataFrame):
            print(f"\nDATAFRAME SHAPE: {data.shape}")
            print("\nCOLUMNS:")
            print(data.columns.tolist())
            print("\nSAMPLE DATA (first 5 rows):")
            print(data.head())
        
        # Handle other data types
        else:
            print("\nCONTENTS:")
            pp.pprint(data)
            
    except Exception as e:
        print(f"ERROR: Could not read pickle file: {e}")

def main():
    """
    Main function to select and view pickle files
    """
    # Define paths to your pickle files
    pickle_files = {
        "1": Path("data/processed/feature_subsets.pkl"),
        "2": Path("models/lstm_model_summary.pkl"),
        "3": Path("models/lstm_preprocessing_info.pkl")
    }
    
    while True:
        print("\n\nAVAILABLE PICKLE FILES:")
        for num, path in pickle_files.items():
            print(f"{num}: {path}")
        
        print("\nSelect a file number to view (or 'q' to quit):")
        choice = input("> ")
        
        if choice.lower() == 'q':
            break
        
        if choice in pickle_files:
            view_pickle(pickle_files[choice])
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()