import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import random
import os
from data_loader import DataLoader

class CRISPRModel:
    def __init__(self):
        # Ensemble of Classifiers for Probability Output
        # 1. Random Forest
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42)
        
        # 2. Gradient Boosting (Excellent for tabular/feature-heavy data)
        self.gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        
        # 3. Neural Network (MLP) for non-linear sequence interactions
        self.mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=500, random_state=42)
        
        # Feature Scaler/Encoder
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.is_trained = False
        self.max_len = 23  # Standard gRNA (20) + PAM (3)
        self.feature_names = []
        
        # Feature weights for explanation
        self.feature_importances_ = {}

    def _pad_sequence(self, seq):
        """Pads sequence with 'N' to max_len."""
        return seq.ljust(self.max_len, 'N')

    def _calculate_bio_features(self, target_seq, candidate_seq):
        """
        Extracts advanced biological features for off-target prediction.
        """
        length = min(len(target_seq), len(candidate_seq))
        
        # 1. Sequence Mismatches
        mismatches = [i for i in range(length) if target_seq[i] != candidate_seq[i]]
        num_mismatches = len(mismatches)
        
        # 2. Seed Region Sensitivity (PAM-proximal 10-12 bases)
        # Assuming 3' PAM, seed is the last 10-12 bases before PAM
        seed_region_size = 10
        seed_mismatches = [p for p in mismatches if (length - p) <= seed_region_size]
        
        # 3. PAM Check (NGG)
        # Check if candidate ends with GG (simplified PAM check)
        has_valid_pam = 1.0 if candidate_seq.endswith("GG") else 0.0
        
        # 4. GC Content & Enthalpy approximations
        gc_count = candidate_seq.count('G') + candidate_seq.count('C')
        gc_content = gc_count / len(candidate_seq) if len(candidate_seq) > 0 else 0
        
        # 5. Position-Weighted Mismatch Score
        # Mismatches closer to PAM (higher index) are more penalized (less likely to cleave = lower prob)
        # But for *features*, we want to capture the signal. 
        # Feature: Weighted Mismatch Score (Higher = Mismatches are in critical regions)
        weighted_mismatch_score = 0
        for pos in mismatches:
            dist_to_pam = length - pos
            # Weights: Seed (1-10) = 1.0, Distal = 0.3
            w = 1.0 if dist_to_pam <= 10 else 0.3
            weighted_mismatch_score += w
            
        return {
            "num_mismatches": num_mismatches,
            "seed_mismatches": len(seed_mismatches),
            "gc_content": gc_content,
            "has_valid_pam": has_valid_pam,
            "weighted_mismatch_score": weighted_mismatch_score,
            "is_distal_heavy": 1 if (num_mismatches > 0 and len(seed_mismatches) == 0) else 0
        }

    def _generate_synthetic_data(self, n_samples=3000):
        """
        Generates augmented synthetic data if no experimental data is found.
        Enforces biological constraints:
        - Exact matches = Label 1
        - Seed mismatches = Label 0 (mostly)
        - Distal mismatches = Label 1 (probability)
        """
        bases = ['A', 'T', 'G', 'C']
        data_rows = []
        labels = []
        
        print(f"Generating {n_samples} biologically-constrained synthetic samples...")
        
        for _ in range(n_samples):
            # Target (gRNA)
            target_seq = "".join(random.choices(bases, k=20)) + "NGG"
            
            # Case 1: Positive (On-target / High likely off-target)
            if random.random() < 0.3:
                # Perfect match (with PAM)
                cand_seq = target_seq
                label = 1
            elif random.random() < 0.5:
                 # Distal Mismatches (1-3 mismatches far from PAM) -> High Cleavage Prob
                cand_chars = list(target_seq)
                for _ in range(random.randint(1, 3)):
                    pos = random.randint(0, 8) # Distal region
                    cand_chars[pos] = random.choice([b for b in bases if b != cand_chars[pos]])
                cand_seq = "".join(cand_chars)
                label = 1 # Still likely to cleave
            else:
                # Negative (Off-target, no cleavage)
                # Seed mismatches or too many mismatches
                cand_chars = list(target_seq)
                
                # Force seed mismatch?
                if random.random() < 0.7:
                    # Mutate within seed (last 10 bases before PAM)
                    pos = random.randint(10, 19)
                    cand_chars[pos] = random.choice([b for b in bases if b != cand_chars[pos]])
                
                # Add random noise
                for _ in range(random.randint(2, 6)):
                    pos = random.randint(0, 19)
                    cand_chars[pos] = random.choice([b for b in bases if b != cand_chars[pos]])
                    
                cand_seq = "".join(cand_chars)
                label = 0
            
            # Extract features
            bio = self._calculate_bio_features(target_seq, cand_seq)
            padded_cand = self._pad_sequence(cand_seq)
            
            row = bio
            for i in range(self.max_len):
                row[f"pos_{i}"] = padded_cand[i]
                
            data_rows.append(row)
            labels.append(label)
            
        return pd.DataFrame(data_rows), np.array(labels)

    def train(self):
        """
        Trains the ensemble model using experimental data (if available) or synthetic data.
        """
        loader = DataLoader("data")
        raw_df = loader.load_experimental_data()
        
        if raw_df is not None:
             print("Training on EXPERIMENTAL DATA...")
             # Convert raw df to features
             data_rows = []
             labels = []
             for _, r in raw_df.iterrows():
                 t, c, l = r['target_seq'], r['candidate_seq'], r['label']
                 bio = self._calculate_bio_features(t, c)
                 padded_cand = self._pad_sequence(c)
                 row = bio
                 for i in range(self.max_len):
                     row[f"pos_{i}"] = padded_cand[i]
                 data_rows.append(row)
                 labels.append(l)
             X = pd.DataFrame(data_rows)
             y = np.array(labels)
        else:
             print("No experimental data found. Using Biology-Aware Synthetic Data.")
             X, y = self._generate_synthetic_data(5000)

        # Preprocessing
        X_cat = X[[f"pos_{i}" for i in range(self.max_len)]]
        X_num = X.drop([f"pos_{i}" for i in range(self.max_len)], axis=1)
        
        X_cat_encoded = self.encoder.fit_transform(X_cat)
        self.feature_names = list(self.encoder.get_feature_names_out()) + list(X_num.columns)
        
        X_final = np.hstack([X_cat_encoded, X_num.values])
        
        # Split for calibration/validation
        X_train, X_val, y_train, y_val = train_test_split(X_final, y, test_size=0.2, random_state=42)
        
        print("Training Ensemble Classifiers...")
        
        # Train RF
        self.rf.fit(X_train, y_train)
        
        # Train GBM
        # GBM doesn't accept 'class_weight' natively in this version usually, relying on sample_weight if needed
        # but gradient boosting naturally handles hard examples.
        self.gbm.fit(X_train, y_train)
        
        # Train MLP
        self.mlp.fit(X_train, y_train)
        
        # Calibration (Platt Scaling) - Optional but recommended for "Probabilities"
        # We can wrap one model or just trust the ensemble average. 
        # For simplicity in this demo, we'll use raw predict_proba from fitted models.
        
        self.is_trained = True
        
        # Evaluation
        preds_probs = self.predict_proba_ensemble(X_val)
        roc = roc_auc_score(y_val, preds_probs)
        acc = accuracy_score(y_val, (preds_probs > 0.5).astype(int))
        
        print(f"\nModel Evaluation (Validation Set):")
        print(f"ROC-AUC: {roc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        
        # Feature Importance Analysis
        self.feature_importances_ = dict(zip(self.feature_names, self.rf.feature_importances_))

    def predict_proba_ensemble(self, X_final):
        """
        Weighted average of probabilities from RF, GBM, and MLP.
        Weights: RF(0.4), GBM(0.4), MLP(0.2)
        """
        p1 = self.rf.predict_proba(X_final)[:, 1]
        p2 = self.gbm.predict_proba(X_final)[:, 1]
        p3 = self.mlp.predict_proba(X_final)[:, 1]
        
        return (p1 * 0.4) + (p2 * 0.4) + (p3 * 0.2)

    def predict(self, grna_seq):
        """
        Generates candidates and predicts cleavage probability.
        """
        if not self.is_trained:
            self.train()
            
        # 1. Generate hypothetical off-targets for demonstration
        candidates = []
        bases = ['A', 'T', 'G', 'C']
        
        # Add exact match
        candidates.append(grna_seq)
        
        # Add random mutants
        for _ in range(10):
            chars = list(grna_seq)
            # 1 to 4 mutations
            n_mut = random.randint(1, 4)
            for _ in range(n_mut):
                pos = random.randint(0, len(grna_seq)-1)
                chars[pos] = random.choice([b for b in bases if b != chars[pos]])
            candidates.append("".join(chars))
            
        results = []
        
        # 2. Process for prediction
        for cand in candidates:
            bio = self._calculate_bio_features(grna_seq, cand)
            padded_cand = self._pad_sequence(cand)
            
            row = bio
            for i in range(self.max_len):
                row[f"pos_{i}"] = padded_cand[i]
            
            # Transform
            X_row = pd.DataFrame([row])
            X_cat = X_row[[f"pos_{i}" for i in range(self.max_len)]]
            X_num = X_row.drop([f"pos_{i}" for i in range(self.max_len)], axis=1)
            
            X_cat_encoded = self.encoder.transform(X_cat)
            X_final = np.hstack([X_cat_encoded, X_num.values])
            
            # Predict
            prob = self.predict_proba_ensemble(X_final)[0]
            
            results.append({
                "sequence": cand,
                "mismatches": bio['num_mismatches'],
                "off_target_prob": round(float(prob), 4)
            })
            
        return results

    def get_feature_importance(self):
        """Aggregates feature importance for biology explanation."""
        if not self.is_trained:
            return {}
            
        summary = {
            "Seed Sensitivity (PAM-Proximal)": 0,
            "Gene Distal Mismatches": 0,
            "GC Content": 0,
            "PAM Validity": 0
        }
        
        for name, val in self.feature_importances_.items():
            if "seed_mismatches" in name or "weighted_mismatch" in name:
                summary["Seed Sensitivity (PAM-Proximal)"] += val
            elif "num_mismatches" in name:
                summary["Gene Distal Mismatches"] += val
            elif "gc_content" in name:
                summary["GC Content"] += val
            elif "has_valid_pam" in name:
                summary["PAM Validity"] += val
            elif "pos_" in name:
                # Assign positional importance roughly
                # pos_20, 21, 22 are PAM/Seed area usually
                if int(name.split("_")[1]) >= 15:
                    summary["Seed Sensitivity (PAM-Proximal)"] += val * 0.1
                else:
                    summary["Gene Distal Mismatches"] += val * 0.1
                    
        # Normalize
        total = sum(summary.values())
        if total > 0:
            for k in summary:
                summary[k] = round(summary[k] / total, 3)
                
        return summary

if __name__ == "__main__":
    model = CRISPRModel()
    model.train()
    print("\nFeature Analysis:")
    print(model.get_feature_importance())
