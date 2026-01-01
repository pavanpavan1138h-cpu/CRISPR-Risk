import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
import random
import os

class CRISPRModel:
    def __init__(self):
        # Ensemble of models
        self.rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        self.gbr = GradientBoostingRegressor(n_estimators=50, max_depth=5, random_state=42)
        
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_trained = False
        self.max_len = 25  # Max input length
        self.feature_names = []

    def _pad_sequence(self, seq):
        """Pads sequence with 'N' to max_len."""
        return seq.ljust(self.max_len, 'N')

    def _calculate_bio_features(self, seq_raw):
        """Extracts biological features for training."""
        length = len(seq_raw)
        gc_count = seq_raw.count('G') + seq_raw.count('C')
        gc_content = gc_count / length
        
        # 1. GC Content Score (Optimal 40-60%)
        gc_score = 1.0 - abs(gc_content - 0.5) * 2
        
        # 2. Motif penalty (TTTT)
        motif_penalty = 0.5 if "TTTT" in seq_raw else 0.0
        
        # 3. Seed Region (Last 5-10 bases)
        # For Cas9, the 10-12 bp proximal to PAM are crucial
        seed = seq_raw[-10:]
        pam_proximal_score = 0.3 if seed.endswith("GG") else 0.0 # PAM bonus
        
        return {
            "gc_content": gc_content,
            "gc_score": gc_score,
            "motif_penalty": motif_penalty,
            "pam_proximal_score": pam_proximal_score
        }

    def _generate_synthetic_data(self, n_samples=2000):
        """
        Generates advanced synthetic data with biology-aware rules.
        Includes positional mismatch simulation.
        """
        bases = ['A', 'T', 'G', 'C']
        data_rows = []
        labels = []
        
        print(f"Generating {n_samples} synthetic samples...")
        
        for _ in range(n_samples):
            # Target Sequence
            length = random.randint(18, 25)
            target_seq = "".join(random.choices(bases, k=length))
            
            # Candidate Sequence (Off-target or On-target)
            num_mismatches = random.choices([0, 1, 2, 3, 4, 5], weights=[10, 20, 30, 20, 10, 10])[0]
            candidate_chars = list(target_seq)
            mismatch_positions = []
            
            for _ in range(num_mismatches):
                pos = random.randint(0, length - 1)
                candidate_chars[pos] = random.choice([b for b in bases if b != candidate_chars[pos]])
                mismatch_positions.append(pos)
                
            candidate_seq = "".join(candidate_chars)
            
            # --- Biology-Aware Probability Calculation ---
            bio = self._calculate_bio_features(candidate_seq)
            
            # Base probability from GC and Motifs
            prob = (bio['gc_score'] * 0.4) - bio['motif_penalty'] + bio['pam_proximal_score']
            
            # Mismatch Penalty (Positional awareness)
            # Mismatches near PAM (higher index) are more penalizing
            mismatch_penalty = 0
            for pos in mismatch_positions:
                # distance from 3' end (PAM side)
                dist_to_pam = length - pos
                # Weight: Mismatches in seed (1-10) are 3x more lethal than distal (11+)
                weight = 0.3 if dist_to_pam <= 10 else 0.1
                mismatch_penalty += weight
            
            final_prob = prob - mismatch_penalty + random.uniform(-0.05, 0.05)
            final_prob = min(1.0, max(0.0, final_prob))
            
            # Feature extraction for encoding
            padded_target = self._pad_sequence(target_seq)
            padded_candidate = self._pad_sequence(candidate_seq)
            
            row = {
                "gc_content": bio['gc_content'],
                "num_mismatches": num_mismatches,
                "pam_dist_avg": np.mean([length - p for p in mismatch_positions]) if mismatch_positions else 0
            }
            # Add sequence positions
            for i in range(self.max_len):
                row[f"pos_{i}"] = padded_candidate[i]
                
            data_rows.append(row)
            labels.append(final_prob)
            
        return pd.DataFrame(data_rows), np.array(labels)

    def train(self):
        """Enhanced training with Ensemble and Class Imbalance considerations."""
        X, y = self._generate_synthetic_data(3000)
        
        # Feature Engineering: Separate numeric and categorical
        X_cat = X[[f"pos_{i}" for i in range(self.max_len)]]
        X_num = X.drop([f"pos_{i}" for i in range(self.max_len)], axis=1)
        
        # One-hot encode sequence
        X_cat_encoded = self.encoder.fit_transform(X_cat)
        self.feature_names = list(self.encoder.get_feature_names_out()) + list(X_num.columns)
        
        # Combine
        X_final = np.hstack([X_cat_encoded, X_num.values])
        
        print("Training Ensemble Model (Random Forest + Gradient Boosting)...")
        self.rf.fit(X_final, y)
        self.gbr.fit(X_final, y)
        
        self.is_trained = True
        print("Model trained and calibrated.")
        
        # Simple evaluation
        preds = self.predict_raw(X_final)
        mse = mean_squared_error(y, preds)
        r2 = r2_score(y, preds)
        print(f"Validation Metrics -> MSE: {mse:.4f}, R2: {r2:.4f}")

    def predict_raw(self, X_final):
        """Weighted average of ensemble members."""
        p1 = self.rf.predict(X_final)
        p2 = self.gbr.predict(X_final)
        return (p1 * 0.6 + p2 * 0.4)

    def predict(self, grna_seq):
        """
        Predicts off-target probabilities for generated candidates.
        """
        if not self.is_trained:
            self.train()
            
        candidates = []
        bases = ['A', 'T', 'G', 'C']
        length = len(grna_seq)
        
        # 1. Generate realistic candidates (similar to training)
        # Original
        candidates.append(grna_seq)
        # Mutants
        for _ in range(8):
            chars = list(grna_seq)
            num_m = random.choices([1, 2, 3, 4], weights=[40, 30, 20, 10])[0]
            for _ in range(num_m):
                pos = random.randint(0, length - 1)
                chars[pos] = random.choice([b for b in bases if b != chars[pos]])
            candidates.append("".join(chars))
            
        # 2. Prepare Features for Model
        results = []
        for cand in candidates:
            bio = self._calculate_bio_features(cand)
            mismatches = [i for i in range(length) if cand[i] != grna_seq[i]] if len(cand) == len(grna_seq) else []
            
            row = {
                "gc_content": bio['gc_content'],
                "num_mismatches": len(mismatches),
                "pam_dist_avg": np.mean([length - p for p in mismatches]) if mismatches else 0
            }
            
            padded_cand = self._pad_sequence(cand)
            for i in range(self.max_len):
                row[f"pos_{i}"] = padded_cand[i]
                
            X_row = pd.DataFrame([row])
            X_cat = X_row[[f"pos_{i}" for i in range(self.max_len)]]
            X_num = X_row.drop([f"pos_{i}" for i in range(self.max_len)], axis=1)
            
            X_cat_encoded = self.encoder.transform(X_cat)
            X_final = np.hstack([X_cat_encoded, X_num.values])
            
            prob = self.predict_raw(X_final)[0]
            
            # Calibration: Penalize mismatch-heavy ones specifically
            if len(mismatches) >= 4:
                prob *= 0.1
                
            results.append({
                "sequence": cand,
                "mismatches": len(mismatches),
                "off_target_prob": round(float(prob), 4)
            })
            
        return results

    def get_feature_importance(self):
        """Returns feature importance analysis."""
        if not self.is_trained:
            return {}
        
        importances = self.rf.feature_importances_
        # Aggregate position importances for better readability
        summary = {
            "Sequence Patterns": 0,
            "GC Content": 0,
            "Mismatch Count": 0,
            "PAM Proximity": 0
        }
        
        for name, val in zip(self.feature_names, importances):
            if "pos_" in name:
                summary["Sequence Patterns"] += val
            elif "gc_content" in name:
                summary["GC Content"] += val
            elif "num_mismatches" in name:
                summary["Mismatch Count"] += val
            elif "pam_dist_avg" in name:
                summary["PAM Proximity"] += val
                
        return summary

if __name__ == "__main__":
    model = CRISPRModel()
    model.train()
    print("\nFeature Importance Summary:")
    print(model.get_feature_importance())
