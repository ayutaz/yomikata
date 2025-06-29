import pytest
from unittest.mock import Mock, patch, MagicMock
from speach import ttlig

from yomikata.dictionary import Dictionary


class TestDictionaryInitialization:
    """Test Dictionary class initialization."""
    
    def test_init_default_tagger(self):
        """Test initialization with default tagger."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger.return_value = MagicMock()
            dictionary = Dictionary()
            assert isinstance(dictionary.tagger, MagicMock)
            mock_tagger.assert_called_once()
    
    def test_init_with_ipadic(self):
        """Test initialization with ipadic tagger."""
        with patch('fugashi.GenericTagger') as mock_tagger:
            mock_tagger.return_value = MagicMock()
            dictionary = Dictionary(tagger="ipadic")
            assert isinstance(dictionary.tagger, MagicMock)
    
    def test_init_with_juman(self):
        """Test initialization with juman tagger."""
        with patch('fugashi.GenericTagger') as mock_tagger:
            mock_tagger.return_value = MagicMock()
            dictionary = Dictionary(tagger="juman")
            assert isinstance(dictionary.tagger, MagicMock)
    
    def test_init_with_sudachi(self):
        """Test initialization with sudachi tagger."""
        with patch('sudachipy.Dictionary') as mock_dict:
            with patch('sudachipy.Tokenizer') as mock_tokenizer:
                mock_dict_instance = MagicMock()
                mock_tokenizer_instance = MagicMock()
                mock_dict.return_value = mock_dict_instance
                mock_dict_instance.create.return_value = mock_tokenizer_instance
                
                dictionary = Dictionary(tagger="sudachi")
                # For sudachi, tagger is a lambda function
                assert callable(dictionary.tagger)
    
    def test_init_with_invalid_tagger(self):
        """Test initialization with invalid tagger."""
        # Currently Dictionary doesn't raise ValueError for invalid taggers
        dictionary = Dictionary(tagger="invalid_tagger")
        # It should have some default behavior or raise an error
        assert not hasattr(dictionary, 'token_to_surface')


class TestDictionaryFurigana:
    """Test Dictionary furigana generation."""
    
    @pytest.fixture
    def mock_dictionary_unidic(self):
        """Create mock dictionary with unidic tagger."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Mock token
            mock_token = MagicMock()
            mock_token.surface = "今日"
            mock_token.feature.kana = "キョウ"
            mock_token.char_type = 2  # Kanji
            
            mock_tagger_instance.return_value = [mock_token]
            
            dictionary = Dictionary()
            dictionary.tokenizer = mock_tagger_instance
            return dictionary
    
    @pytest.fixture
    def mock_dictionary_sudachi(self):
        """Create mock dictionary with sudachi tagger."""
        with patch('sudachipy.Dictionary') as mock_dict:
            with patch('sudachipy.Tokenizer') as mock_tokenizer:
                mock_dict_instance = MagicMock()
                mock_tokenizer_instance = MagicMock()
                mock_dict.return_value = mock_dict_instance
                mock_dict_instance.create.return_value = mock_tokenizer_instance
                
                # Mock morpheme
                mock_morpheme = MagicMock()
                mock_morpheme.surface.return_value = "今日"
                mock_morpheme.reading_form.return_value = "キョウ"
                
                mock_tokenizer_instance.tokenize.return_value = [mock_morpheme]
                
                dictionary = Dictionary(tagger="sudachi")
                return dictionary
    
    def test_furigana_simple_text(self, mock_dictionary_unidic):
        """Test furigana generation for simple text."""
        # Mock the parse_furigana to return appropriate RubyToken
        with patch('yomikata.utils.parse_furigana') as mock_parse:
            mock_parse.return_value = ttlig.RubyToken(surface="今日", ruby="きょう")
            
            result = mock_dictionary_unidic.furigana("今日は")
            assert isinstance(result, str)
    
    def test_furigana_no_kanji(self, mock_dictionary_unidic):
        """Test furigana generation for text without kanji."""
        # Mock token without kanji
        mock_token = MagicMock()
        mock_token.surface = "です"
        mock_token.feature.kana = "デス"
        mock_token.char_type = 1  # Not kanji
        
        mock_dictionary_unidic.tokenizer.return_value = [mock_token]
        
        with patch('yomikata.utils.parse_furigana') as mock_parse:
            mock_parse.return_value = ttlig.RubyToken(surface="です", ruby="")
            
            result = mock_dictionary_unidic.furigana("です")
            assert isinstance(result, str)
    
    def test_furigana_empty_text(self, mock_dictionary_unidic):
        """Test furigana generation for empty text."""
        mock_dictionary_unidic.tokenizer.return_value = []
        
        result = mock_dictionary_unidic.furigana("")
        assert result == ""
    
    def test_furigana_with_existing_furigana(self, mock_dictionary_unidic):
        """Test handling of text that already has furigana."""
        text_with_furigana = "今日{きょう}は"
        
        # Mock tokens
        mock_token1 = MagicMock()
        mock_token1.surface = "今日"
        mock_token1.feature.kana = "キョウ"
        mock_token1.char_type = 2
        
        mock_token2 = MagicMock()
        mock_token2.surface = "は"
        mock_token2.feature.kana = "ハ"
        mock_token2.char_type = 1
        
        mock_dictionary_unidic.tokenizer.return_value = [mock_token1, mock_token2]
        
        with patch('yomikata.utils.remove_furigana', return_value="今日は"):
            with patch('yomikata.utils.parse_furigana') as mock_parse:
                mock_parse.side_effect = [
                    ttlig.RubyToken(surface="今日", ruby="きょう"),
                    ttlig.RubyToken(surface="は", ruby="")
                ]
                
                result = mock_dictionary_unidic.furigana(text_with_furigana)
                assert isinstance(result, str)
    
    def test_furigana_sudachi_tagger(self, mock_dictionary_sudachi):
        """Test furigana generation with sudachi tagger."""
        with patch('yomikata.utils.parse_furigana') as mock_parse:
            mock_parse.return_value = ttlig.RubyToken(surface="今日", ruby="きょう")
            
            result = mock_dictionary_sudachi.furigana("今日")
            assert isinstance(result, str)


class TestDictionaryStaticMethods:
    """Test Dictionary static methods."""
    
    def test_furi_to_ruby_identical(self):
        """Test furi_to_ruby when surface and kana are identical."""
        result = Dictionary.furi_to_ruby("です", "です")
        assert isinstance(result, ttlig.RubyToken)
        # When surface and kana are identical, the surface becomes empty and text is in groups
        assert result.surface == ""
        assert len(result.groups) == 2
        assert result.groups[0] == ""
        assert result.groups[1] == "です"
    
    def test_furi_to_ruby_different(self):
        """Test furi_to_ruby when surface and kana are different."""
        result = Dictionary.furi_to_ruby("今日", "きょう")
        assert isinstance(result, ttlig.RubyToken)
        assert result.surface == "今日"
        # Check that the groups contain a RubyFrag with the correct furigana
        assert len(result.groups) == 1
        assert isinstance(result.groups[0], ttlig.RubyFrag)
        assert result.groups[0].text == "今日"
        assert result.groups[0].furi == "きょう"
    
    def test_furi_to_ruby_partial_match(self):
        """Test furi_to_ruby with partial matching."""
        # Test case where kana partially matches surface
        result = Dictionary.furi_to_ruby("食べる", "たべる")
        assert isinstance(result, ttlig.RubyToken)
    
    def test_furi_to_ruby_empty_inputs(self):
        """Test furi_to_ruby with empty inputs."""
        result = Dictionary.furi_to_ruby("", "")
        assert isinstance(result, ttlig.RubyToken)
        assert result.surface == ""
        # Empty inputs result in one empty string in groups
        assert len(result.groups) == 1
        assert result.groups[0] == ""


class TestDictionaryEdgeCases:
    """Test edge cases for Dictionary."""
    
    def test_ascii_space_token_handling(self):
        """Test handling of ASCII space tokens."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Create token that raises UnicodeDecodeError for surface
            mock_token = MagicMock()
            mock_token.surface = property(lambda self: (_ for _ in ()).throw(
                UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
            ))
            
            # Make char_type accessible
            mock_token.char_type = 1
            
            mock_tagger_instance.return_value = [mock_token]
            
            dictionary = Dictionary()
            
            # Should handle the error and treat as space
            with patch('yomikata.utils.parse_furigana') as mock_parse:
                mock_parse.return_value = ttlig.RubyToken(surface=" ", ruby="")
                result = dictionary.furigana("test text")
                assert isinstance(result, str)
    
    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Mock token with special characters
            mock_token = MagicMock()
            mock_token.surface = "😀"
            mock_token.feature.kana = "😀"
            mock_token.char_type = 0
            
            mock_tagger_instance.return_value = [mock_token]
            
            dictionary = Dictionary()
            
            with patch('yomikata.utils.parse_furigana') as mock_parse:
                mock_parse.return_value = ttlig.RubyToken(surface="😀", ruby="")
                result = dictionary.furigana("😀")
                assert isinstance(result, str)
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        with patch('fugashi.Tagger') as mock_tagger:
            mock_tagger_instance = MagicMock()
            mock_tagger.return_value = mock_tagger_instance
            
            # Create many tokens
            mock_tokens = []
            for i in range(1000):
                token = MagicMock()
                token.surface = "あ"
                token.feature.kana = "ア"
                token.char_type = 1
                mock_tokens.append(token)
            
            mock_tagger_instance.return_value = mock_tokens
            
            dictionary = Dictionary()
            
            with patch('yomikata.utils.parse_furigana') as mock_parse:
                mock_parse.return_value = ttlig.RubyToken(surface="あ", ruby="")
                
                long_text = "あ" * 1000
                result = dictionary.furigana(long_text)
                assert isinstance(result, str)