import random

class BioFeatureAnnotator:
    """
    Annotates off-target sites with Biological Impact Features for BRS calculation.
    
    Features:
    - E: Gene Essentiality (0.0 - 1.0)
    - C: Chromatin Accessibility (0.0 - 1.0)
    - F: Functional Region (Exon=1.0, Promoter=0.8, Intron=0.4, Intergenic=0.1)
    - D: Distance to TSS (0.0 - 1.0, proximity decay)
    - H: Disease Association (0.0 - 1.0)
    
    Note: Since real-time API queries to Ensembl/ENCODE for thousands of sites 
    are too slow for a demo, this class uses probabilistic rules based on 
    sequence context (e.g., GC content) and known gene lists to simulate 
    realistic annotations.
    """
    
    def __init__(self):
        # Small lookup of known essential genes for demo purposes
        self.essential_genes = {
            "TP53", "BRCA1", "BRCA2", "MYC", "GAPDH", "ACTB", "PCNA", "POLR2A"
        }
        
    def annotate(self, target_seq, candidate_seq):
        """
        Returns a dictionary of normalized biological features.
        """
        # 1. Functional Region (F)
        # Rule: High GC content (>60%) often correlates with Exons/Promoters (CpG islands)
        gc_content = self._calculate_gc(candidate_seq)
        if gc_content > 0.65:
            # Likely Exon or Promoter
            region_type = "Exon" if random.random() < 0.7 else "Promoter"
            f_score = 1.0 if region_type == "Exon" else 0.8
        elif gc_content > 0.45:
             # Likely Intron
            region_type = "Intron"
            f_score = 0.4
        else:
            # Likely Intergenic
            region_type = "Intergenic"
            f_score = 0.1
            
        # 2. Chromatin Accessibility (C)
        # Rule: Promoters/Exons are usually more accessible (Open Chromatin)
        if f_score >= 0.8:
            # Open chromatin
            c_score = random.uniform(0.7, 1.0)
        elif f_score == 0.4:
            # Heterochromatin or facultative
            c_score = random.uniform(0.2, 0.6)
        else:
            # Closed
            c_score = random.uniform(0.0, 0.3)
            
        # 3. Transcriptional Start Site Distance (D)
        # Rule: Promoters are close (1.0), Intergenic far (0.0)
        if region_type == "Promoter":
            d_score = random.uniform(0.9, 1.0)
        elif region_type == "Exon":
            d_score = random.uniform(0.4, 0.8)
        else:
            d_score = random.uniform(0.0, 0.3)
            
        # 4. Essentiality (E) & Disease (H)
        # Fix 1: Scaling - Ensure values represent full risk spectrum
        is_essential = False
        is_disease = False
        
        # Increased probability for demo purposes so "High Risk" is reachable
        # Rule: Exons/Promoters have 30% chance of being Essential, 40% chance of Disease link
        if f_score >= 0.8:
            if random.random() < 0.35: is_essential = True
            if random.random() < 0.40: is_disease = True
            
        # Fix 1: Ensure post-scaling ranges are > 0.2
        e_score = 0.95 if is_essential else 0.2 
        h_score = 0.90 if is_disease else 0.2
        
        # Correlation: Essential genes are often disease linked, but not always. 
        # Boost H if E is high to simulate "Critical Alignment"
        if is_essential and not is_disease:
            h_score += 0.3  # Bump to 0.5
        
        return {
            "gene_essentiality": e_score,      # E
            "chromatin_accessibility": c_score,# C
            "functional_region": f_score,      # F
            "tss_distance": d_score,           # D
            "disease_association": h_score     # H
        }

    def _calculate_gc(self, seq):
        if not seq: return 0
        g = seq.count('G') + seq.count('g')
        c = seq.count('C') + seq.count('c')
        return (g + c) / len(seq)
