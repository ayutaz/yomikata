import pytest
from yomikata import utils


def test_imports():
    """Test that the module can be imported."""
    assert utils is not None


def test_parse_furigana():
    """Test the parse_furigana function with basic input."""
    # Test with simple text without furigana
    result = utils.parse_furigana("こんにちは")
    assert result == "こんにちは"
    
    # Test with furigana notation
    result = utils.parse_furigana("今日{きょう}")
    assert "今日" in result or "きょう" in result