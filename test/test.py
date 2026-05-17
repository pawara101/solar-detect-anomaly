import os
import sys
from PIL import Image
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import explainModel, detectAnomaly


def test_explainModel():
    # Load a sample image
    image_path = "sample_image.jpg"  # Replace with your test image path
    if not os.path.exists(image_path):
        pytest.skip("Test image not found")
    
    image_pil = Image.open(image_path)

    # Test the explainModel function
    cam_image = explainModel(image_pil)

    # Verify output
    assert cam_image is not None
    assert hasattr(cam_image, 'save'), "Output should be a PIL Image"
    
    # Save the CAM output for review
    cam_image.save("cam_output.jpg")


def test_detectAnomaly():
    # Load a sample image
    image_path = "sample_image.jpg"
    if not os.path.exists(image_path):
        pytest.skip("Test image not found")
    
    image_pil = Image.open(image_path)

    # Test the detectAnomaly function
    result = detectAnomaly(image_pil)

    # Verify output
    assert result is not None
    assert isinstance(result, (bool, int, float, dict)), "Result should be a valid detection output"


def test_explainModel_invalid_input():
    # Test with invalid input
    with pytest.raises((TypeError, AttributeError)):
        explainModel(None)


def test_detectAnomaly_invalid_input():
    # Test with invalid input
    with pytest.raises((TypeError, AttributeError)):
        detectAnomaly(None)