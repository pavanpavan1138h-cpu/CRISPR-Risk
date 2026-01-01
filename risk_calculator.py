import random

def calculate_brs(off_target_prob):
    """
    Calculates the Biosafety Risk Score (BRS) based on the formula:
    BRS = P * (0.3E + 0.2C + 0.2F + 0.1D + 0.2H)
    
    Since we don't have a real genomic database connected, we will
    simulate the biological features (E, C, F, D, H).
    """
    
    # Simulate feature scores (normalized 0-1)
    # E: Gene Essentiality (Higher = more essential)
    E = random.uniform(0, 1)
    
    # C: Chromatin Accessibility (Higher = open chromatin, more accessible)
    C = random.uniform(0, 1)
    
    # F: Functional Region Importance (1.0 for Exon, 0.7 Promoter, etc.)
    regions = ['Exon', 'Promoter', 'Intron', 'Intergenic']
    region_type = random.choice(regions)
    if region_type == 'Exon':
        F = 1.0
    elif region_type == 'Promoter':
        F = 0.7
    elif region_type == 'Intron':
        F = 0.3
    else:
        F = 0.1
        
    # D: TSS Proximity (Higher = closer to Transcription Start Site)
    D = random.uniform(0, 1)
    
    # H: Disease/Pathway Involvement (Higher = critical pathway)
    H = random.uniform(0, 1)
    
    # Calculate Biological Impact Score (BIS)
    bis = (0.3 * E) + (0.2 * C) + (0.2 * F) + (0.1 * D) + (0.2 * H)
    
    # Calculate BRS
    brs = off_target_prob * bis
    
    # Determine Risk Category
    if brs < 0.3:
        risk_class = "Low"
    elif brs < 0.6:
        risk_class = "Medium"
    else:
        risk_class = "High"
        
    return {
        "brs_score": round(brs, 4),
        "risk_class": risk_class,
        "features": {
            "gene_essentiality": round(E, 2),
            "chromatin_accessibility": round(C, 2),
            "functional_region": region_type,
            "tss_proximity": round(D, 2),
            "disease_involvement": round(H, 2)
        }
    }
