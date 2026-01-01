import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_model import CRISPRModel
from risk_calculator import calculate_brs

def test_model():
    print("Testing ML Model...")
    model = CRISPRModel()
    model.train()
    
    test_seq = "AGCTAGCTAGCTAGCTAGCT"
    results = model.predict(test_seq)
    
    assert len(results) > 0, "Model should return candidates"
    print(f"✅ Prediction Successful. Generated {len(results)} off-target candidates.")
    
    first_res = results[0]
    print(f"   Sample: {first_res}")
    return results

def test_risk_calc(predictions):
    print("\nTesting Risk Calculator...")
    for pred in predictions:
        res = calculate_brs(pred['off_target_prob'])
        assert 'brs_score' in res
        assert 'risk_class' in res
        print(f"   Prob: {pred['off_target_prob']} -> BRS: {res['brs_score']} ({res['risk_class']})")
    print("✅ Risk Calculator Successful.")

if __name__ == "__main__":
    try:
        preds = test_model()
        test_risk_calc(preds)
        print("\n✅ ALL TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)
