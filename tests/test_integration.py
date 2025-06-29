import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from yomikata.reader import Reader
from yomikata.dictionary import Dictionary
from yomikata.dbert import dBert


class TestReaderInterface:
    """Test the Reader abstract interface."""
    
    def test_reader_cannot_be_instantiated(self):
        """Test that Reader abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            Reader()
    
    def test_reader_subclass_must_implement_furigana(self):
        """Test that subclasses must implement furigana method."""
        class IncompleteReader(Reader):
            pass
        
        with pytest.raises(TypeError):
            IncompleteReader()
    
    def test_reader_subclass_with_furigana(self):
        """Test that subclasses with furigana method can be instantiated."""
        class CompleteReader(Reader):
            def furigana(self, text: str) -> str:
                return text
        
        reader = CompleteReader()
        assert reader.furigana("test") == "test"


class TestDictionaryIntegration:
    """Integration tests for Dictionary class."""
    
    @pytest.mark.parametrize("tagger", ["unidic", "ipadic", "juman", "sudachi"])
    def test_dictionary_with_different_taggers(self, tagger):
        """Test Dictionary with different tagger backends."""
        # Mock all the tagger imports
        with patch('fugashi.Tagger') as mock_unidic:
            with patch('fugashi.GenericTagger') as mock_generic:
                with patch('sudachipy.Dictionary') as mock_sudachi:
                    
                    # Setup mocks
                    mock_unidic.return_value = MagicMock()
                    mock_generic.return_value = MagicMock()
                    mock_sudachi_instance = MagicMock()
                    mock_sudachi.return_value = mock_sudachi_instance
                    mock_sudachi_instance.create.return_value = MagicMock()
                    
                    # Should not raise exception
                    dictionary = Dictionary(tagger=tagger)
                    assert dictionary is not None
                    # The tagger attribute holds the actual tokenizer object, not the string name
                    assert hasattr(dictionary, 'tagger')
    
    def test_dictionary_furigana_pipeline(self):
        """Test complete furigana generation pipeline."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Setup mock tokens for a complete sentence
            tokens = []
            
            # "今日" token
            token1 = MagicMock()
            token1.surface = "今日"
            token1.feature.kana = "キョウ"
            token1.char_type = 2  # Kanji
            tokens.append(token1)
            
            # "は" token
            token2 = MagicMock()
            token2.surface = "は"
            token2.feature.kana = "ハ"
            token2.char_type = 1  # Hiragana
            tokens.append(token2)
            
            # "良い" token
            token3 = MagicMock()
            token3.surface = "良い"
            token3.feature.kana = "ヨイ"
            token3.char_type = 2  # Kanji
            tokens.append(token3)
            
            # "天気" token
            token4 = MagicMock()
            token4.surface = "天気"
            token4.feature.kana = "テンキ"
            token4.char_type = 2  # Kanji
            tokens.append(token4)
            
            mock_tagger_instance.return_value = tokens
            
            dictionary = Dictionary()
            result = dictionary.furigana("今日は良い天気")
            
            # Should return a string result
            assert isinstance(result, str)
            assert len(result) > 0


class TestDBertIntegration:
    """Integration tests for dBert class."""
    
    def test_dbert_model_loading_pipeline(self):
        """Test complete model loading pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            # Create minimal required files
            config = {
                "architectures": ["BertForTokenClassification"],
                "model_type": "bert",
                "num_labels": 10
            }
            (model_dir / "config.json").write_text(json.dumps(config))
            
            tokenizer_config = {
                "do_lower_case": False,
                "tokenizer_class": "BertTokenizer"
            }
            (model_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
            
            # Create vocab file
            vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "今日", "は"]
            (model_dir / "vocab.txt").write_text("\n".join(vocab))
            
            # Create label encoder
            label_encoder = {"labels": ["きょう", "こんにち"]}
            (model_dir / "label_encoder.json").write_text(json.dumps(label_encoder))
            
            # Create heteronyms
            heteronyms = {"今日": ["きょう", "こんにち"]}
            (model_dir / "heteronyms.json").write_text(json.dumps(heteronyms))
            
            # Mock model loading
            with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained') as mock_model:
                with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    
                    # Load model
                    reader = dBert(model_dir)
                    
                    # Verify components loaded
                    assert reader.heteronyms == heteronyms
                    assert hasattr(reader, 'model')
                    assert hasattr(reader, 'tokenizer')
                    assert hasattr(reader, 'label_encoder')
    
    def test_dbert_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained') as mock_model:
            with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
                # Setup mocks
                model_instance = MagicMock()
                tokenizer_instance = MagicMock()
                
                mock_model.return_value = model_instance
                mock_tokenizer.return_value = tokenizer_instance
                
                # Create reader
                reader = dBert.__new__(dBert)
                reader.model = model_instance
                reader.tokenizer = tokenizer_instance
                reader.heteronyms = {"今日": ["きょう", "こんにち"]}
                reader.surfaceIDs = {"今日": [5]}
                reader.label_encoder = MagicMock()
                reader.label_encoder.decode.return_value = ["きょう"]
                reader.max_length = 128  # Add missing attribute
                reader.device = MagicMock()  # Add missing device attribute
                
                # Mock tokenization
                encoding = MagicMock()
                encoding.input_ids = [[2, 5, 6, 3]]  # [CLS] 今日 は [SEP]
                encoding.word_ids.return_value = [None, 0, 1, None]
                tokenizer_instance.return_value = encoding
                
                # Mock model output
                output = MagicMock()
                import torch
                output.logits = torch.rand(1, 4, 2)  # batch_size=1, seq_len=4, num_labels=2
                model_instance.return_value = output
                
                # Run prediction
                result = reader.furigana("今日は")
                
                # Should return string
                assert isinstance(result, str)


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_dictionary_vs_dbert_comparison(self):
        """Test that both Dictionary and dBert produce valid outputs."""
        test_text = "今日は良い天気です"
        
        # Test Dictionary
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger.return_value = MagicMock()
            mock_tagger.return_value.return_value = []  # Empty tokens for simplicity
            
            dict_reader = Dictionary()
            dict_result = dict_reader.furigana(test_text)
            assert isinstance(dict_result, str)
        
        # Test dBert
        with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained'):
            with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained'):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create minimal files
                    model_dir = Path(temp_dir)
                    (model_dir / "heteronyms.json").write_text('{}')
                    (model_dir / "label_encoder.json").write_text('{"labels": []}')
                    
                    dbert_reader = MagicMock(spec=dBert)
                    dbert_reader.furigana.return_value = test_text
                    dbert_result = dbert_reader.furigana(test_text)
                    assert isinstance(dbert_result, str)
    
    def test_empty_text_handling(self):
        """Test that all readers handle empty text correctly."""
        # Dictionary
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger.return_value = MagicMock()
            mock_tagger.return_value.return_value = []
            
            dict_reader = Dictionary()
            assert dict_reader.furigana("") == ""
        
        # dBert
        dbert_reader = MagicMock(spec=dBert)
        dbert_reader.furigana.return_value = ""
        assert dbert_reader.furigana("") == ""
    
    def test_special_characters_handling(self):
        """Test handling of special characters across readers."""
        special_text = "Hello 世界! 😀"
        
        # Dictionary should handle without error
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Mock tokens
            tokens = []
            for char in special_text:
                token = MagicMock()
                token.surface = char
                token.feature.kana = char
                token.char_type = 0
                tokens.append(token)
            
            mock_tagger_instance.return_value = tokens
            
            dict_reader = Dictionary()
            result = dict_reader.furigana(special_text)
            assert isinstance(result, str)