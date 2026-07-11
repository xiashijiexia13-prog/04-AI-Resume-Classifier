from text_processing import clean_text


def test_clean_text_normalizes_resume_content():
    raw = "  <p>Python ENGINEER</p> https://example.com  C++   APIs 🚀 "

    assert clean_text(raw) == "python engineer c apis"


def test_clean_text_rejects_non_string_values():
    assert clean_text(None) == ""
    assert clean_text(123) == ""
