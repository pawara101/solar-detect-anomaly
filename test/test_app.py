from streamlit.testing.v1 import AppTest

def test_title():
    at = AppTest.from_file("app.py")
    at.run()

    assert at.title[0].value == "Solar Detector"

def test_text_input():
    at = AppTest.from_file("app.py")
    at.run()

    at.text_input[0].input("John")
    at.button[0].click()
    at.run()

    assert "Hello John" in at.markdown[0].value