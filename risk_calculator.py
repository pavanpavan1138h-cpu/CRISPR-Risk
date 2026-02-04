def calculate_brs(prob, bio_features=None):
    """
    Calculates Biosafety Risk Score (BRS) using the formula:
    BRS = P * (alpha*E + beta*C + gamma*F + delta*D + eps*H)
    
    Weights:
    - E (Essentiality): 0.3
    - C (Chromatin): 0.2
    - F (Functional Region): 0.2
    - D (Distance to TSS): 0.1
    - H (Disease Association): 0.2
    """
    if bio_features is None:
        # Default fallback (should not happen in new flow)
        bio_features = {
            "gene_essentiality": 0.5,
            "chromatin_accessibility": 0.5,
            "functional_region": 0.5,
            "tss_distance": 0.5,
            "disease_association": 0.0
        }
        
    E = bio_features.get("gene_essentiality", 0)
    C = bio_features.get("chromatin_accessibility", 0)
    F = bio_features.get("functional_region", 0)
    D = bio_features.get("tss_distance", 0)
    H = bio_features.get("disease_association", 0)
    
    # Biological Impact Score (BIS)
    bis = (0.3 * E) + (0.2 * C) + (0.2 * F) + (0.1 * D) + (0.2 * H)
    
    brs_score = prob * bis
    
    risk_class = "Low"
    color = "blue"
    
    if brs_score >= 0.55:
        risk_class = "High"
        color = "red"
    elif brs_score >= 0.25:
        risk_class = "Medium"
        color = "orange"
    else: # BRS < 0.25
        risk_class = "Low"
        color = "green" # Merging Low/Negligible for cleaner UI or keep as Low
        if brs_score < 0.1:
            risk_class = "Negligible"
            color = "green"
        else:
            color = "blue" # Low is blue
            
    # Note: User requested "Low: BRS < 0.25", "Medium: 0.25 <= BRS < 0.55", "High: BRS >= 0.55"
        
    # Return structure matching what frontend expects
    return {
        "brs_score": round(brs_score, 4),
        "risk_class": risk_class,
        "risk_color": color,
        "genomic_context": {
            "gene_essentiality": round(E, 2),
            "chromatin_accessibility": round(C, 2),
            "region_type": "Annotated", # Simplified for now
            "tss_proximity": round(D, 2),
            "disease_involvement": round(H, 2),
            "bio_impact_score": round(bis, 3)
        }
    }
