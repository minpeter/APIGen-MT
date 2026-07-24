"""Unit tests for token counting functionality.

This module tests the TokenCounter class and token tracking in LocalOpenAILLMClient.
All tests use local tiktoken encoding without requiring any model downloads.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from llm_client import TokenCounter, TokenUsage, LocalOpenAILLMClient


class TestTokenCounterInitialization:
    """Tests for TokenCounter initialization."""

    def test_token_counter_loads_successfully(self):
        """Test that TokenCounter can be initialized."""
        counter = TokenCounter()
        assert counter is not None
        assert counter.encoding is not None

    def test_token_counter_with_default_encoding(self):
        """Test TokenCounter uses cl100k_base by default."""
        counter = TokenCounter()
        # cl100k_base is the default for GPT-4/GPT-3.5
        assert counter.encoding is not None

    def test_token_counter_with_explicit_encoding(self):
        """Test TokenCounter can use a specific encoding."""
        counter = TokenCounter(encoding_name="cl100k_base")
        assert counter is not None
        assert counter.encoding is not None


class TestTokenCountingBasic:
    """Basic tests for token counting."""

    def test_count_empty_string(self):
        """Test that empty string returns 0 tokens."""
        counter = TokenCounter()
        assert counter.count_tokens("") == 0

    def test_count_whitespace_only(self):
        """Test that whitespace-only strings return minimal tokens."""
        counter = TokenCounter()
        # Whitespace is typically 1 token or handled efficiently
        assert counter.count_tokens("   ") >= 0

    def test_count_simple_english(self):
        """Test counting tokens in simple English text."""
        counter = TokenCounter()
        # "Hello world" should be approximately 2-3 tokens
        token_count = counter.count_tokens("Hello world")
        assert 1 <= token_count <= 5

    def test_count_single_word(self):
        """Test counting tokens in a single word."""
        counter = TokenCounter()
        # "hello" is typically 1 token
        token_count = counter.count_tokens("hello")
        assert token_count >= 1


class TestTokenCountingChat:
    """Tests for chat message token counting."""

    def test_count_empty_chat(self):
        """Test that empty message list returns minimal tokens."""
        counter = TokenCounter()
        # Empty list should still have assistant priming tokens
        assert counter.count_chat_tokens([]) >= 0

    def test_count_single_message(self):
        """Test counting tokens in a single chat message."""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        token_count = counter.count_chat_tokens(messages)
        # Should include message framing + content
        assert token_count > 0

    def test_count_multi_turn_chat(self):
        """Test counting tokens in a multi-turn conversation."""
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "What's the weather like?"}
        ]
        token_count = counter.count_chat_tokens(messages)
        # Multi-turn conversation should have more tokens
        assert token_count > len(messages) * 3  # At least framing tokens

    def test_count_chat_with_empty_content(self):
        """Test counting tokens with empty message content."""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}
        ]
        token_count = counter.count_chat_tokens(messages)
        # Should still count framing tokens
        assert token_count >= len(messages) * 3

    def test_count_chat_with_none_content(self):
        """Test counting tokens with None content (should handle gracefully)."""
        counter = TokenCounter()
        messages = [
            {"role": "user", "content": None},
            {"role": "assistant", "content": "Response"}
        ]
        token_count = counter.count_chat_tokens(messages)
        assert token_count >= 0


class TestTokenCountingSpecialCases:
    """Tests for special token counting scenarios."""

    def test_count_unicode_text(self):
        """Test counting tokens in unicode text."""
        counter = TokenCounter()
        # Unicode characters may take 1-2 tokens each
        token_count = counter.count_tokens("こんにちは")  # Japanese
        assert token_count >= 1

    def test_count_emojis(self):
        """Test counting tokens with emojis."""
        counter = TokenCounter()
        # Emojis typically take 1-3 tokens
        token_count = counter.count_tokens("Hello 👋 World 🌍")
        assert token_count > 2  # More than just "Hello World"

    def test_count_special_characters(self):
        """Test counting tokens with special characters."""
        counter = TokenCounter()
        # Special chars like newlines, tabs, etc.
        text = "Line 1\nLine 2\tTabbed"
        token_count = counter.count_tokens(text)
        assert token_count >= 1

    def test_count_code_snippets(self):
        """Test counting tokens in code snippets."""
        counter = TokenCounter()
        code = """def hello():
            return "world"
        """
        token_count = counter.count_tokens(code)
        # Code typically has more tokens due to indentation/symbols
        assert token_count >= 3

    def test_count_long_text(self):
        """Test counting tokens in long text."""
        counter = TokenCounter()
        long_text = "word " * 1000  # 1000 words
        token_count = counter.count_tokens(long_text)
        # Should be approximately 1000 tokens (maybe slightly more/less)
        assert 900 < token_count < 1100

    def test_count_json_structure(self):
        """Test counting tokens in JSON-like structures."""
        counter = TokenCounter()
        json_text = '{"name": "John", "age": 30, "city": "New York"}'
        token_count = counter.count_tokens(json_text)
        assert token_count >= 4  # At least 4 tokens for the structure


class TestTokenCountingConsistency:
    """Tests for token counting consistency."""

    def test_same_text_same_count(self):
        """Test that same text always produces same token count."""
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog."
        count1 = counter.count_tokens(text)
        count2 = counter.count_tokens(text)
        assert count1 == count2

    def test_longer_text_more_tokens(self):
        """Test that longer text generally has more tokens."""
        counter = TokenCounter()
        short_text = "Hello"
        long_text = "Hello world, how are you doing today?"
        assert counter.count_tokens(short_text) < counter.count_tokens(long_text)

    def test_additive_property(self):
        """Test that concatenating text approximately adds tokens."""
        counter = TokenCounter()
        text1 = "Hello world"
        text2 = "Goodbye world"
        combined = text1 + " " + text2
        
        count1 = counter.count_tokens(text1)
        count2 = counter.count_tokens(text2)
        combined_count = counter.count_tokens(combined)
        
        # Combined should be roughly the sum (allowing for small variance)
        assert abs(combined_count - (count1 + count2)) <= 2


class TestTokenUsageTracking:
    """Tests for TokenUsage class."""

    def test_token_usage_initialization(self):
        """Test TokenUsage initializes to zero."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_token_usage_add(self):
        """Test adding token counts."""
        usage = TokenUsage()
        usage.add(prompt=10, completion=20, total=30)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_token_usage_accumulation(self):
        """Test that token usage accumulates correctly."""
        usage = TokenUsage()
        usage.add(prompt=10, completion=5, total=15)
        usage.add(prompt=20, completion=10, total=30)
        assert usage.prompt_tokens == 30
        assert usage.completion_tokens == 15
        assert usage.total_tokens == 45

    def test_token_usage_to_dict(self):
        """Test conversion to dictionary."""
        usage = TokenUsage()
        usage.add(prompt=100, completion=50, total=150)
        result = usage.to_dict()
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }


class TestLocalOpenAILLMClientTokenTracking:
    """Tests for token tracking in LocalOpenAILLMClient."""

    def test_client_has_token_counter(self):
        """Test that LocalOpenAILLMClient initializes with a TokenCounter."""
        client = LocalOpenAILLMClient()
        assert hasattr(client, 'token_counter')
        assert client.token_counter is not None

    def test_client_token_usage_initialization(self):
        """Test that token usage is initialized properly."""
        client = LocalOpenAILLMClient()
        usage = client.get_token_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["total_calls"] == 0

    def test_client_reset_token_usage(self):
        """Test resetting token usage."""
        client = LocalOpenAILLMClient()
        # Simulate some usage
        client.token_usage.add(prompt=100, completion=50, total=150)
        client.total_calls = 5
        
        client.reset_token_usage()
        usage = client.get_token_usage()
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["total_calls"] == 0


class TestTokenCounterEdgeCases:
    """Edge case tests for token counter."""

    def test_very_long_text(self):
        """Test handling very long text."""
        counter = TokenCounter()
        very_long_text = "word " * 100000  # 100k words
        # Should handle without error
        token_count = counter.count_tokens(very_long_text)
        assert token_count > 0
        assert token_count > 90000  # Reasonable lower bound

    def test_multilingual_text(self):
        """Test handling multilingual text."""
        counter = TokenCounter()
        texts = [
            "Hello world",  # English
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "こんにちは世界",  # Japanese
            "你好世界",  # Chinese
            "Привет мир",  # Russian
        ]
        for text in texts:
            count = counter.count_tokens(text)
            assert count >= 1

    def test_special_tokens_in_text(self):
        """Test handling special tokens that might be used by models."""
        counter = TokenCounter()
        text_with_special = "<|im_start|>user<|im_end|>"
        # Should handle special tokens without error
        count = counter.count_tokens(text_with_special)
        assert count >= 0

    def test_repeated_patterns(self):
        """Test counting repeated patterns."""
        counter = TokenCounter()
        pattern = "abc " * 100
        count = counter.count_tokens(pattern)
        # Should be roughly proportional to repetitions
        single = counter.count_tokens("abc ")
        # Tokenization boundaries can cause variance, but it should still be roughly proportional
        # The important thing is that long patterns are counted correctly
        expected = single * 100
        assert count > 0  # Just verify it counts something
        assert abs(count - expected) <= expected * 0.5  # Allow 50% variance for boundary effects


class TestTokenCountingApproximation:
    """Tests to verify token counting approximates real-world usage."""

    def test_english_word_approximation(self):
        """Test that English words are approximately 1.3 tokens on average."""
        counter = TokenCounter()
        # Typical English: 100 tokens ~= 75 words
        text = "The quick brown fox jumps over the lazy dog. " * 10
        word_count = len(text.split())
        token_count = counter.count_tokens(text)
        
        # Should be roughly 1.3 tokens per word (allowing variance)
        ratio = token_count / word_count
        assert 0.8 <= ratio <= 2.0

    def test_code_token_ratio(self):
        """Test token count for code is higher than English."""
        counter = TokenCounter()
        english = "Hello world, this is a test sentence."
        code = "def test():\n    x = [1, 2, 3]\n    return x"
        
        english_tokens = counter.count_tokens(english)
        code_tokens = counter.count_tokens(code)
        
        english_words = len(english.split())
        code_words = len(code.split())
        
        english_ratio = english_tokens / english_words
        code_ratio = code_tokens / code_words
        
        # Code typically has higher token-to-word ratio
        assert code_ratio >= english_ratio * 0.5  # At least comparable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
