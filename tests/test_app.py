"""Tests for Streamlit app functions"""
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


class TestAppFunctions:
    """Test suite for app.py functions"""

    def test_add_border(self):
        """Test add_border function"""
        from app import add_border
        
        # Test with simple HTML
        html = "<span>test</span>"
        result = add_border(html)
        assert "<div style=" in result
        assert "overflow-x: auto" in result
        assert html in result
        
        # Test with newlines
        html_with_newline = "<span>test\nwith\nnewline</span>"
        result = add_border(html_with_newline)
        assert "\n" not in result
        assert "test with newline" in result

    def test_furigana_to_spacy(self):
        """Test furigana_to_spacy conversion"""
        from app import furigana_to_spacy
        
        # Test with furigana text
        text = "これは{漢字/かんじ}です"
        result = furigana_to_spacy(text)
        
        assert result["text"] == "漢字"
        assert len(result["ents"]) == 1
        assert result["ents"][0]["label"] == "かんじ"
        assert result["ents"][0]["start"] == 0
        assert result["ents"][0]["end"] == 2
        
        # Test with multiple furigana
        text = "{人間/にんげん}は{自由/じゆう}だ"
        result = furigana_to_spacy(text)
        
        assert result["text"] == "人間, 自由"
        assert len(result["ents"]) == 2
        assert result["ents"][0]["label"] == "にんげん"
        assert result["ents"][1]["label"] == "じゆう"

    @patch('pandas.read_csv')
    def test_get_random_sentence(self, mock_read_csv):
        """Test get_random_sentence function"""
        from app import get_random_sentence
        
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'sentence': ['テスト文章1', 'テスト文章2', 'テスト文章3']
        })
        mock_read_csv.return_value = mock_df
        
        # Test that it returns a sentence from the DataFrame
        result = get_random_sentence()
        assert result in ['テスト文章1', 'テスト文章2', 'テスト文章3']

    @patch('app.load_dbert_model')
    def test_get_dbert_prediction_and_heteronym_list(self, mock_load_model):
        """Test get_dbert_prediction_and_heteronym_list function"""
        from app import get_dbert_prediction_and_heteronym_list
        
        # Mock dBert model
        mock_reader = Mock()
        mock_reader.furigana.return_value = "これは{漢字/かんじ}です"
        mock_reader.heteronyms = ["漢字"]
        mock_load_model.return_value = mock_reader
        
        # Test prediction
        result_furigana, result_heteronyms = get_dbert_prediction_and_heteronym_list("これは漢字です")
        
        assert result_furigana == "これは{漢字/かんじ}です"
        assert result_heteronyms == ["漢字"]
        mock_reader.furigana.assert_called_once_with("これは漢字です")

    @patch('yomikata.utils.load_dict')
    @patch('pathlib.Path.exists')
    def test_get_stats(self, mock_exists, mock_load_dict):
        """Test get_stats function"""
        from app import get_stats
        
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock stats data
        mock_stats = {
            "test": {
                "accuracy": 0.95,
                "heteronym_performance": {
                    "人間": {
                        "accuracy": 0.98,
                        "readings": {
                            "にんげん": {"found": {"にんげん": 50}, "n": 52},
                            "じんかん": {"found": {"じんかん": 2}, "n": 2}
                        }
                    },
                    "自由": {
                        "accuracy": 0.92,
                        "readings": {
                            "じゆう": {"found": {"じゆう": 46}, "n": 50}
                        }
                    }
                }
            }
        }
        mock_load_dict.return_value = mock_stats
        
        # Test stats loading
        global_accuracy, df = get_stats()
        
        assert global_accuracy == 0.95
        assert len(df) == 1  # Only "人間" has multiple readings
        assert "人間" in df["heteronym"].values

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('yomikata.dbert.dBert')
    def test_load_dbert_model(self, mock_dbert, mock_listdir, mock_exists):
        """Test load_dbert_model function"""
        from app import load_dbert_model
        
        # Mock environment
        mock_exists.return_value = True
        mock_listdir.return_value = ['model.pth', 'config.json']
        
        # Mock dBert initialization
        mock_reader = Mock()
        mock_dbert.return_value = mock_reader
        
        # Test model loading
        result = load_dbert_model()
        
        assert result == mock_reader
        mock_dbert.assert_called_once()

    def test_streamlit_app_structure(self):
        """Test basic Streamlit app structure"""
        # This test verifies that the app can be imported without errors
        try:
            import app
            assert hasattr(app, 'add_border')
            assert hasattr(app, 'furigana_to_spacy')
            assert hasattr(app, 'get_random_sentence')
            assert hasattr(app, 'load_dbert_model')
            assert hasattr(app, 'get_dbert_prediction_and_heteronym_list')
            assert hasattr(app, 'get_stats')
        except ImportError as e:
            pytest.fail(f"Failed to import app: {e}")


class TestStreamlitCaching:
    """Test suite for Streamlit caching decorators"""

    def test_cache_decorators(self):
        """Verify that functions use correct cache decorators"""
        import app
        import inspect
        
        # Check add_border uses @st.cache_data
        assert hasattr(app.add_border, '__wrapped__')
        
        # Check load_dbert_model uses @st.cache_resource
        assert hasattr(app.load_dbert_model, '__wrapped__')
        
        # Check get_dbert_prediction_and_heteronym_list uses @st.cache_data
        assert hasattr(app.get_dbert_prediction_and_heteronym_list, '__wrapped__')
        
        # Check get_stats uses @st.cache_data
        assert hasattr(app.get_stats, '__wrapped__')
        
        # Check furigana_to_spacy uses @st.cache_data
        assert hasattr(app.furigana_to_spacy, '__wrapped__')

    def test_no_deprecated_features(self):
        """Ensure no deprecated Streamlit features are used"""
        import app
        import inspect
        
        # Read the source code
        source = inspect.getsource(app)
        
        # Check for deprecated features
        assert "@st.cache" not in source.replace("@st.cache_data", "").replace("@st.cache_resource", "")
        assert "st.experimental_rerun" not in source
        
        # Verify new features are used
        assert "@st.cache_data" in source
        assert "@st.cache_resource" in source
        assert "st.rerun()" in source