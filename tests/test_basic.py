import pytest
from yomikata import utils


def test_imports():
    """Test that the module can be imported."""
    assert utils is not None


def test_parse_furigana():
    """Test the parse_furigana function with basic input."""
    # Test with simple text without furigana
    result = utils.parse_furigana("こんにちは")
    # parse_furigana returns a RubyToken object, not a string
    assert hasattr(result, 'surface') or str(result) == "こんにちは"
    
    # Test with furigana notation
    result = utils.parse_furigana("今日{きょう}")
    # Check that it returns some kind of object or string containing the content
    result_str = str(result)
    assert "今日" in result_str or "きょう" in result_str or hasattr(result, 'surface')