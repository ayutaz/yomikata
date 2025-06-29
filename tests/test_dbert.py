import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from yomikata.dbert import dBert
from yomikata.utils import LabelEncoder


class TestDBertInitialization:
    """Test dBert class initialization."""
    
    def test_init_with_valid_directory(self):
        """Test initialization with valid model directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock model files
            model_dir = Path(temp_dir)
            
            # Create necessary files
            (model_dir / "config.json").write_text('{"model_type": "bert"}')
            (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
            (model_dir / "tokenizer_config.json").write_text('{}')
            (model_dir / "vocab.txt").write_text("fake\nvocab")
            (model_dir / "label_encoder.json").write_text('{"labels": ["label1", "label2"]}')
            (model_dir / "heteronyms.json").write_text('{"今日": ["きょう", "こんにち"]}')
            
            # Mock the actual model loading
            with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained') as mock_model:
                with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    
                    # Should not raise exception
                    reader = dBert(model_dir)
                    assert reader is not None
    
    def test_init_with_missing_files(self):
        """Test initialization with missing model files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Empty directory should raise error
            with pytest.raises(Exception):
                dBert(Path(temp_dir))
    
    def test_init_with_reinitialize(self):
        """Test initialization with reinitialize flag."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained') as mock_model:
                with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    
                    # Should create new model even if directory exists
                    reader = dBert(model_dir, reinitialize=True)
                    assert reader is not None


class TestDBertPrediction:
    """Test dBert prediction functionality."""
    
    @pytest.fixture
    def mock_dbert(self):
        """Create a mock dBert instance for testing."""
        with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained'):
            with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained'):
                reader = MagicMock(spec=dBert)
                reader.heteronyms = {"今日": ["きょう", "こんにち"], "表": ["ひょう", "おもて"]}
                reader.tokenizer = MagicMock()
                reader.model = MagicMock()
                reader.label_encoder = MagicMock()
                reader.surfaceIDs = {"今日": [100], "表": [101]}
                return reader
    
    def test_furigana_normal_text(self, mock_dbert):
        """Test furigana generation for normal text."""
        # Setup mock behavior
        mock_dbert.furigana = dBert.furigana.__get__(mock_dbert, dBert)
        mock_dbert.label_encoder.decode.return_value = ["きょう"]
        
        # Mock the standardize_text to return the input
        with patch('yomikata.utils.standardize_text', side_effect=lambda x: x):
            # Mock model inference
            mock_output = MagicMock()
            mock_output.logits = torch.tensor([[[0.1, 0.9]]])  # Mock logits
            mock_dbert.model.return_value = mock_output
            
            # Mock tokenizer
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [[101, 100, 102]]  # [CLS] 今日 [SEP]
            mock_encoding.word_ids.return_value = [None, 0, None]
            mock_dbert.tokenizer.return_value = mock_encoding
            
            result = mock_dbert.furigana("今日は良い天気")
            # Should contain furigana notation
            assert isinstance(result, str)
    
    def test_furigana_empty_text(self, mock_dbert):
        """Test furigana generation for empty text."""
        mock_dbert.furigana = dBert.furigana.__get__(mock_dbert, dBert)
        
        with patch('yomikata.utils.standardize_text', return_value=""):
            result = mock_dbert.furigana("")
            assert result == ""
    
    def test_furigana_no_heteronyms(self, mock_dbert):
        """Test furigana generation for text without heteronyms."""
        mock_dbert.furigana = dBert.furigana.__get__(mock_dbert, dBert)
        
        with patch('yomikata.utils.standardize_text', side_effect=lambda x: x):
            # Mock tokenizer to return tokens without heteronyms
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [[101, 200, 201, 102]]  # No heteronym IDs
            mock_encoding.word_ids.return_value = [None, 0, 1, None]
            mock_dbert.tokenizer.return_value = mock_encoding
            
            result = mock_dbert.furigana("普通のテキスト")
            assert isinstance(result, str)


class TestDBertTraining:
    """Test dBert training functionality."""
    
    def test_batch_preprocess_function(self):
        """Test batch preprocessing for training."""
        with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
            tokenizer_instance = MagicMock()
            mock_tokenizer.return_value = tokenizer_instance
            
            # Create minimal dBert instance
            reader = MagicMock(spec=dBert)
            reader.tokenizer = tokenizer_instance
            reader.heteronyms = {"今日": ["きょう", "こんにち"]}
            reader.surfaceIDs = {"今日": [100]}
            reader.label_encoder = LabelEncoder()
            reader.label_encoder.fit(["きょう", "こんにち"])
            
            # Bind the method
            reader.batch_preprocess_function = dBert.batch_preprocess_function.__get__(reader, dBert)
            
            # Test data
            entries = [{"sentence": "今日は", "furigana": "今日{きょう}は"}]
            
            # Mock tokenizer behavior
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [[101, 100, 200, 102]]
            mock_encoding.word_ids.return_value = [None, 0, 1, None]
            tokenizer_instance.return_value = mock_encoding
            
            result = reader.batch_preprocess_function(entries)
            
            assert "input_ids" in result
            assert "labels" in result
    
    def test_train_with_mock_dataset(self):
        """Test training with mock dataset."""
        with patch('yomikata.dbert.Trainer') as mock_trainer:
            reader = MagicMock(spec=dBert)
            reader.train = dBert.train.__get__(reader, dBert)
            reader.model = MagicMock()
            reader.tokenizer = MagicMock()
            reader.label_encoder = MagicMock()
            reader.heteronyms = {}
            reader.device = MagicMock()
            reader.max_length = 128
            reader.batch_preprocess_function = MagicMock()
            
            # Mock dataset
            mock_dataset = MagicMock()
            mock_dataset.train_test_split.return_value = mock_dataset
            
            # Mock trainer
            trainer_instance = MagicMock()
            trainer_instance.train.return_value = None
            trainer_instance.evaluate.return_value = {"eval_loss": 0.5}
            mock_trainer.return_value = trainer_instance
            
            # Should not raise exception
            result = reader.train(mock_dataset)
            assert isinstance(result, dict)


class TestDBertSaveLoad:
    """Test saving and loading dBert models."""
    
    def test_save_model(self):
        """Test saving model to directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "saved_model"
            
            # Create mock dBert instance
            reader = MagicMock(spec=dBert)
            reader.save = dBert.save.__get__(reader, dBert)
            reader.model = MagicMock()
            reader.tokenizer = MagicMock()
            reader.label_encoder = LabelEncoder()
            reader.label_encoder.fit(["label1", "label2"])
            reader.heteronyms = {"今日": ["きょう", "こんにち"]}
            
            # Create the directory first
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            reader.save(save_dir)
            
            # Verify files were created
            assert save_dir.exists()
            assert (save_dir / "label_encoder.json").exists()
            assert (save_dir / "heteronyms.json").exists()
    
    def test_load_model(self):
        """Test loading model from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            
            # Create mock files
            (model_dir / "config.json").write_text('{"model_type": "bert"}')
            (model_dir / "pytorch_model.bin").write_bytes(b"fake model")
            (model_dir / "tokenizer_config.json").write_text('{}')
            (model_dir / "vocab.txt").write_text("fake\nvocab")
            
            # Create label encoder file
            label_data = {"labels": ["label1", "label2"]}
            (model_dir / "label_encoder.json").write_text(json.dumps(label_data))
            
            # Create heteronyms file
            heteronyms_data = {"今日": ["きょう", "こんにち"]}
            (model_dir / "heteronyms.json").write_text(json.dumps(heteronyms_data))
            
            with patch('yomikata.dbert.AutoModelForTokenClassification.from_pretrained') as mock_model:
                with patch('yomikata.dbert.BertJapaneseTokenizer.from_pretrained') as mock_tokenizer:
                    mock_model.return_value = MagicMock()
                    mock_tokenizer.return_value = MagicMock()
                    
                    # Create reader instance
                    reader = MagicMock(spec=dBert)
                    reader.load = dBert.load.__get__(reader, dBert)
                    reader.label_encoder = LabelEncoder()
                    reader.heteronyms = {}
                    reader.surfaceIDs = {}
                    
                    # Load model
                    reader.load(model_dir)
                    
                    # Verify loaded correctly
                    assert reader.heteronyms == heteronyms_data


class TestDBertEdgeCases:
    """Test edge cases and error handling."""
    
    def test_long_text_truncation(self):
        """Test handling of text longer than max_length."""
        reader = MagicMock(spec=dBert)
        reader.furigana = dBert.furigana.__get__(reader, dBert)
        reader.heteronyms = {}
        reader.max_length = 128
        reader.device = MagicMock()
        
        # Create very long text
        long_text = "あ" * 1000
        
        with patch('yomikata.utils.standardize_text', side_effect=lambda x: x):
            # Mock tokenizer to handle truncation
            mock_encoding = MagicMock()
            mock_encoding.input_ids = [[101] + [100] * 510 + [102]]  # Max 512 tokens
            mock_encoding.word_ids.return_value = [None] + list(range(510)) + [None]
            reader.tokenizer = MagicMock(return_value=mock_encoding)
            reader.model = MagicMock()
            reader.surfaceIDs = {}
            
            # Should handle without error
            result = reader.furigana(long_text)
            assert isinstance(result, str)
    
    def test_special_characters(self):
        """Test handling of special characters."""
        reader = MagicMock(spec=dBert)
        reader.furigana = dBert.furigana.__get__(reader, dBert)
        reader.max_length = 128
        reader.device = MagicMock()
        
        special_text = "テスト\n\t🔥😀"
        
        with patch('yomikata.utils.standardize_text', side_effect=lambda x: x.strip()):
            reader.tokenizer = MagicMock()
            reader.model = MagicMock()
            reader.heteronyms = {}
            reader.surfaceIDs = {}
            
            # Should handle without error
            result = reader.furigana(special_text)
            assert isinstance(result, str)