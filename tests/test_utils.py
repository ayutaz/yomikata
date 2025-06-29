import pytest
import json
import tempfile
import os
from pathlib import Path
from speach import ttlig
import numpy as np

from yomikata import utils


class TestTextProcessingFunctions:
    """Test text processing utility functions."""
    
    def test_standardize_text_normal(self):
        """Test normal text standardization."""
        assert utils.standardize_text("テスト") == "テスト"
        assert utils.standardize_text("  空白  ") == "空白"
        assert utils.standardize_text("Ｆｕｌｌｗｉｄｔｈ") == "Fullwidth"
    
    def test_standardize_text_old_kanji(self):
        """Test old kanji conversion during standardization."""
        # Testing with actual old kanji mappings from utils
        assert utils.standardize_text("醫者") == "医者"  # 醫 → 医
        assert utils.standardize_text("學校") == "学校"  # 學 → 学
    
    def test_standardize_text_edge_cases(self):
        """Test edge cases for text standardization."""
        assert utils.standardize_text("") == ""
        assert utils.standardize_text("   ") == ""
        assert utils.standardize_text("\n\t混合\r\n") == "混合"
    
    def test_convert_old_kanji(self):
        """Test old kanji conversion."""
        assert utils.convert_old_kanji("醫學會") == "医学会"
        assert utils.convert_old_kanji("普通の文") == "普通の文"
        assert utils.convert_old_kanji("") == ""
    
    def test_remove_furigana(self):
        """Test furigana removal."""
        assert utils.remove_furigana("{漢字/かんじ}") == "漢字"
        assert utils.remove_furigana("{今日/きょう}は{明日/あした}") == "今日は明日"
        assert utils.remove_furigana("普通のテキスト") == "普通のテキスト"
        assert utils.remove_furigana("") == ""
    
    def test_furigana_to_kana(self):
        """Test converting text to kana using furigana."""
        assert utils.furigana_to_kana("{漢字/かんじ}") == "かんじ"
        assert utils.furigana_to_kana("{今日/きょう}は") == "きょうは"
        assert utils.furigana_to_kana("テキスト") == "テキスト"
        assert utils.furigana_to_kana("") == ""
    
    def test_has_kanji(self):
        """Test kanji detection."""
        assert utils.has_kanji("漢字") == True
        assert utils.has_kanji("ひらがな") == False
        assert utils.has_kanji("カタカナ") == False
        assert utils.has_kanji("ABC123") == False
        assert utils.has_kanji("混合文字列") == True
        assert utils.has_kanji("") == False
    
    def test_parse_furigana_normal(self):
        """Test normal furigana parsing."""
        result = utils.parse_furigana("{漢字/かんじ}")
        assert isinstance(result, ttlig.RubyToken)
        # Verify the string representation contains the expected content
        assert "漢字" in str(result) or hasattr(result, 'surface')
    
    def test_parse_furigana_no_furigana(self):
        """Test parsing text without furigana."""
        result = utils.parse_furigana("普通のテキスト")
        assert isinstance(result, ttlig.RubyToken)
    
    def test_parse_furigana_edge_cases(self):
        """Test edge cases for furigana parsing."""
        # Empty string
        result = utils.parse_furigana("")
        assert isinstance(result, ttlig.RubyToken)
        
        # None should raise ValueError
        with pytest.raises(ValueError):
            utils.parse_furigana(None)
    
    def test_parse_furigana_malformed(self):
        """Test parsing malformed furigana."""
        # Missing closing brace - should handle gracefully
        result = utils.parse_furigana("壊れた{ふりがな")
        assert isinstance(result, ttlig.RubyToken)
        
        # Empty parts
        result = utils.parse_furigana("{/}")
        assert isinstance(result, ttlig.RubyToken)


class TestLabelEncoder:
    """Test LabelEncoder class."""
    
    def test_fit_and_encode(self):
        """Test fitting and encoding labels."""
        encoder = utils.LabelEncoder()
        labels = ["cat", "dog", "cat", "bird"]
        
        encoder.fit(labels)
        encoded = encoder.encode(labels)
        
        assert len(encoded) == len(labels)
        # Check that same labels get same encoding
        assert encoded[0] == encoded[2]  # Both "cat"
        # Check that different labels get different encoding
        assert len(set(encoded)) == 3  # 3 unique labels
    
    def test_decode(self):
        """Test decoding indices back to labels."""
        encoder = utils.LabelEncoder()
        labels = ["cat", "dog", "bird"]
        
        encoder.fit(labels)
        # Get actual indices from encoding
        encoded = encoder.encode(labels)
        # Decode them back
        decoded = encoder.decode(encoded)
        
        # Should get back the original labels
        assert decoded == labels
    
    def test_encode_unknown_label(self):
        """Test encoding with unknown label."""
        encoder = utils.LabelEncoder()
        encoder.fit(["cat", "dog"])
        
        # Should handle unknown labels gracefully
        with pytest.raises(Exception):  # Specific exception type depends on implementation
            encoder.encode(["cat", "unknown"])
    
    def test_decode_invalid_index(self):
        """Test decoding with invalid index."""
        encoder = utils.LabelEncoder()
        encoder.fit(["cat", "dog"])
        
        # Should handle invalid indices gracefully
        with pytest.raises(Exception):  # Specific exception type depends on implementation
            encoder.decode([0, 5])  # 5 is out of range
    
    def test_save_and_load(self):
        """Test saving and loading encoder."""
        encoder = utils.LabelEncoder()
        encoder.fit(["cat", "dog", "bird"])
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            encoder.save(f.name)
            temp_path = f.name
        
        try:
            # Load into new encoder
            new_encoder = utils.LabelEncoder.load(temp_path)
            
            # Verify it works the same
            test_labels = ["cat", "dog", "bird"]
            original_encoded = encoder.encode(test_labels)
            new_encoded = new_encoder.encode(test_labels)
            assert list(original_encoded) == list(new_encoded)
            
            # Verify decoding works
            assert new_encoder.decode(new_encoded) == test_labels
        finally:
            os.unlink(temp_path)
    
    def test_empty_labels(self):
        """Test with empty label list."""
        encoder = utils.LabelEncoder()
        encoder.fit([])
        
        assert len(encoder.encode([])) == 0
        assert len(encoder.decode([])) == 0


class TestFileIOFunctions:
    """Test file I/O utility functions."""
    
    def test_load_dict_valid_json(self):
        """Test loading valid JSON dictionary."""
        test_data = {"key1": "value1", "key2": 123}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_path = f.name
        
        try:
            loaded = utils.load_dict(temp_path)
            assert loaded == test_data
        finally:
            os.unlink(temp_path)
    
    def test_load_dict_invalid_json(self):
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                utils.load_dict(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_dict_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            utils.load_dict("/path/that/does/not/exist.json")
    
    def test_save_dict_basic(self):
        """Test saving dictionary to JSON."""
        test_data = {"key": "value", "number": 42}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            utils.save_dict(test_data, temp_path)
            
            # Verify saved content
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            assert loaded == test_data
        finally:
            os.unlink(temp_path)
    
    def test_save_dict_with_unicode(self):
        """Test saving dictionary with unicode characters."""
        test_data = {"japanese": "日本語", "emoji": "😀"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            utils.save_dict(test_data, temp_path)
            loaded = utils.load_dict(temp_path)
            assert loaded == test_data
        finally:
            os.unlink(temp_path)
    
    def test_save_dict_sorted_keys(self):
        """Test saving dictionary with sorted keys."""
        test_data = {"z": 1, "a": 2, "m": 3}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            utils.save_dict(test_data, temp_path, sortkeys=True)
            
            # Read raw content to verify key order
            with open(temp_path, 'r') as f:
                content = f.read()
            
            # Keys should appear in alphabetical order
            assert content.index('"a"') < content.index('"m"') < content.index('"z"')
        finally:
            os.unlink(temp_path)


class TestMLUtilities:
    """Test machine learning utility functions."""
    
    def test_set_seeds(self):
        """Test seed setting for reproducibility."""
        utils.set_seeds(42)
        
        # Generate random numbers
        np_random1 = np.random.rand()
        
        # Reset seeds
        utils.set_seeds(42)
        np_random2 = np.random.rand()
        
        # Should be the same
        assert np_random1 == np_random2
    
    def test_set_seeds_different_seeds(self):
        """Test that different seeds produce different results."""
        utils.set_seeds(42)
        random1 = np.random.rand()
        
        utils.set_seeds(123)
        random2 = np.random.rand()
        
        assert random1 != random2