import pandas as pd
import os
import numpy as np

from dataset_manager import DatasetManager

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.manager = DatasetManager(data_dir)
        
    def load_experimental_data(self):
        """
        Loads experimentally validated datasets (GUIDE-seq, CIRCLE-seq) if available.
        Expected format: CSV files with columns ['grna_target', 'off_target_seq', 'label'] or similar.
        Returns:
            pd.DataFrame: Combined dataset
        """
        combined_data = []
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Attempt download/discovery via manager
        self.manager.download_data()
            
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"[INFO] No experimental CSV files found in '{self.data_dir}'. Will generate synthetic data.")
            return None
            
        print(f"Found {len(files)} experimental datasets: {files}")
        
        for file in files:
            file_path = os.path.join(self.data_dir, file)
            try:
                df = pd.read_csv(file_path)
                # Standardize columns (basic heuristic)
                df = self._standardize_columns(df)
                if df is not None:
                    combined_data.append(df)
            except Exception as e:
                print(f"[ERROR] Failed to load {file}: {e}")
                
        if not combined_data:
            return None
            
        return pd.concat(combined_data, ignore_index=True)

    def _standardize_columns(self, df):
        """
        Maps various dataset column names to standard 'target_seq', 'candidate_seq', 'label'.
        """
        # Map common column names to our standard
        col_map = {
            'grna': 'target_seq', 'sgRNA': 'target_seq', 'target': 'target_seq',
            'off_target': 'candidate_seq', 'off_target_sequence': 'candidate_seq', 'dna': 'candidate_seq',
            'cleavage_freq': 'label', 'read_count': 'label', 'score': 'label', 'risk': 'label'
        }
        
        df = df.rename(columns=col_map)
        
        required = ['target_seq', 'candidate_seq']
        if not all(col in df.columns for col in required):
            return None
            
        # Ensure label exists, default to 1 (positive class) if just a list of off-targets
        if 'label' not in df.columns:
            df['label'] = 1
            
        # Binarize label if it's counts/frequency (cutoff > 0 is off-target)
        # This is a simplification; for regression we'd keep it, but for classification we need 0/1
        # Ideally, we should also have negative samples (non-cleaved sites).
        df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)
        
        return df[['target_seq', 'candidate_seq', 'label']]
