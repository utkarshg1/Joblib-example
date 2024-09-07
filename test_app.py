import pytest
from model.train_model import load_model, predict_results

# Test model loading
def test_load_model():
    model = load_model()
    assert model is not None, "Model should not be None after loading"

# Test prediction function
def test_predict_results():
    model = load_model()
    
    # Define test inputs (e.g., typical iris measurements)
    sep_len = 5.1
    sep_wid = 3.5
    pet_len = 1.4
    pet_wid = 0.2
    
    # Perform prediction
    pred, prob = predict_results(model, sep_len, sep_wid, pet_len, pet_wid)
    
    # Check that the predicted species is a string (e.g., 'setosa')
    assert isinstance(pred, str), "Prediction should be a string representing the species"
    
    # Check that probabilities are a dictionary and sum to 1 (or very close)
    assert isinstance(prob, dict), "Probabilities should be a dictionary"
    assert sum(prob.values()) == pytest.approx(1.0), "Probabilities should sum to 1"

# Additional edge case tests
@pytest.mark.parametrize("sep_len, sep_wid, pet_len, pet_wid", [
    (0.0, 0.0, 0.0, 0.0), # extreme low values
    (10.0, 10.0, 10.0, 10.0), # extreme high values
])
def test_predict_edge_cases(sep_len, sep_wid, pet_len, pet_wid):
    model = load_model()
    pred, prob = predict_results(model, sep_len, sep_wid, pet_len, pet_wid)
    
    assert isinstance(pred, str), "Prediction should be a string"
    assert isinstance(prob, dict), "Probabilities should be a dictionary"
