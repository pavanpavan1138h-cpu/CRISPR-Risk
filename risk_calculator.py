import random

def calculate_brs(off_target_prob):
    """
    Calculates the Biosafety Risk Score (BRS) using the calibrated off-target cleavage probability.
    
    Formula:
    BRS = Cleavage_Probability * Biological_Impact_Score (BIS)
    
    Where BIS is a weighted sum of genomic context features:
    BIS = 0.3*Essentiality + 0.2*Chromatin + 0.2*RegionType + 0.1*TSS_Proximity + 0.2*Disease_Link
    """
    
    # 1. Simulate Genomic Context Features (as we don't have a live genome connection)
    # In a real app, these would come from an API (e.g., Ensembl, UCSC) given the coordinates.
    
    # E: Gene Essentiality (0.0 - 1.0)
    # Higher means the gene is critical for survival -> Higher Risk
    E = random.uniform(0, 1)
    
    # C: Chromatin Accessibility (0.0 - 1.0)
    # Higher means open chromatin -> Cas9 access is easier -> Higher Risk/Likelihood
    C = random.uniform(0, 1)
    
    # F: Functional Region Importance
    regions = ['Exon', 'Promoter', 'Enhancer', 'Intron', 'Intergenic']
    region_type = random.choice(regions)
    if region_type == 'Exon':
        F = 1.0
    elif region_type == 'Promoter':
        F = 0.8
    elif region_type == 'Enhancer':
        F = 0.6
    elif region_type == 'Intron':
        F = 0.2
    else:
        F = 0.05
        
    # D: TSS Proximity (0.0 - 1.0)
    D = random.uniform(0, 1)
    
    # H: Disease/Pathway Involvement (0.0 - 1.0)
    H = random.uniform(0, 1)
    
    # 2. Calculate Biological Impact Score (BIS)
    # Max BIS is 1.0
    bis = (0.3 * E) + (0.2 * C) + (0.2 * F) + (0.1 * D) + (0.2 * H)
    
    # 3. Calculate Final BRS
    # BRS will be between 0.0 and 1.0 (since prob is 0-1 and BIS is 0-1)
    brs = off_target_prob * bis
    
    # 4. Strict Bio-Safety Risk Classification
    # Thresholds adjusted for high-sensitivity safety context
    if brs < 0.1:
        risk_class = "Negligible"
        color = "green"
    elif brs < 0.3:
        risk_class = "Low"
        color = "blue"
    elif brs < 0.6:
        risk_class = "Medium"
        color = "orange"
    else:
        risk_class = "High"
        color = "red"
        
    return {
        "brs_score": round(brs, 4),
        "risk_class": risk_class,
        "risk_color": color,
        "genomic_context": {
            "gene_essentiality": round(E, 2),
            "chromatin_accessibility": round(C, 2),
            "region_type": region_type,
            "tss_proximity": round(D, 2),
            "disease_involvement": round(H, 2),
            "bio_impact_score": round(bis, 3)
        }
    }
