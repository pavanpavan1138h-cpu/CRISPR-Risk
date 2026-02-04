import os
import requests
import pandas as pd

class DatasetManager:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_data(self):
        """
        Attempts to download public CRISPR benchmark datasets.
        Returns:
            list: List of paths to successfully downloaded CSV files.
        """
        # Real public URLs for processed CRISPR datasets
        # Using raw.githubusercontent links for reliability
        dataset_urls = {
            # CIRCLE-seq detected off-targets
            "circle_seq_1.csv": "https://raw.githubusercontent.com/dagrate/public_data_crisprCas9/master/data/CIRCLE_seq_10gRNA_HEK293T.csv",
            # GUIDE-seq detected off-targets (Benchmark)
            "guide_seq_benchmark.csv": "https://raw.githubusercontent.com/Epigenome/crispr-benchmarks/master/datasets/guideseq_all.csv"
        }
        
        downloaded_files = []
        
        print(f"Checking for datasets in {self.data_dir}...")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        for name, url in dataset_urls.items():
            file_path = os.path.join(self.data_dir, name)
            if not os.path.exists(file_path):
                print(f"Downloading {name}...")
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(r.content)
                        downloaded_files.append(file_path)
                        print(f"Successfully downloaded {name}")
                    else:
                        print(f"Failed to download {name}: Status {r.status_code}")
                except Exception as e:
                    print(f"Error downloading {name}: {e}")
            else:
                downloaded_files.append(file_path)
        
        return downloaded_files

if __name__ == "__main__":
    dm = DatasetManager()
    dm.download_data()
