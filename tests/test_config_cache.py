"""Tests for configuration caching system."""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from metareason.config.cache import (
    ConfigCache,
    clear_global_cache,
    disable_caching,
    get_global_cache,
    is_caching_enabled,
    set_global_cache,
)
from metareason.config.models import EvaluationConfig


class TestConfigCache:
    """Test ConfigCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with default settings."""
        cache = ConfigCache()
        assert cache.size() == 0
        stats = cache.get_stats()
        assert stats["total_entries"] == 0
        assert stats["ttl_seconds"] == 300
        assert stats["hot_reload_enabled"] is True

    def test_cache_initialization_custom_settings(self):
        """Test cache initialization with custom settings."""
        cache = ConfigCache(ttl_seconds=600, enable_hot_reload=False)
        stats = cache.get_stats()
        assert stats["ttl_seconds"] == 600
        assert stats["hot_reload_enabled"] is False

    def test_cache_set_and_get(self, sample_config):
        """Test caching and retrieving a configuration."""
        cache = ConfigCache()
        test_path = Path("test.yaml")

        # Set configuration
        cache.set(test_path, sample_config)
        assert cache.size() == 1

        # Get configuration
        cached_config = cache.get(test_path)
        assert cached_config is not None
        assert cached_config.prompt_id == sample_config.prompt_id

    def test_cache_miss(self):
        """Test cache miss for non-existent file."""
        cache = ConfigCache()
        cached_config = cache.get("nonexistent.yaml")
        assert cached_config is None

    def test_cache_ttl_expiry(self, sample_config):
        """Test cache TTL expiry."""
        cache = ConfigCache(ttl_seconds=0.1)  # Very short TTL
        test_path = Path("test.yaml")

        # Cache configuration
        cache.set(test_path, sample_config)
        assert cache.get(test_path) is not None

        # Wait for TTL to expire
        time.sleep(0.2)

        # Should be expired and return None
        assert cache.get(test_path) is None
        assert cache.size() == 0

    def test_cache_invalidate(self, sample_config):
        """Test cache invalidation."""
        cache = ConfigCache()
        test_path = Path("test.yaml")

        # Cache configuration
        cache.set(test_path, sample_config)
        assert cache.size() == 1

        # Invalidate
        result = cache.invalidate(test_path)
        assert result is True
        assert cache.size() == 0

        # Try to invalidate again
        result = cache.invalidate(test_path)
        assert result is False

    def test_cache_clear(self, sample_config):
        """Test clearing the entire cache."""
        cache = ConfigCache()

        # Add multiple configurations
        cache.set("test1.yaml", sample_config)
        cache.set("test2.yaml", sample_config)
        assert cache.size() == 2

        # Clear cache
        cache.clear()
        assert cache.size() == 0

    def test_cache_hot_reload_detection(self, sample_config):
        """Test hot reload detection with file modification."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test: content")

        try:
            cache = ConfigCache(enable_hot_reload=True)

            # Cache configuration
            cache.set(temp_path, sample_config)
            assert cache.get(temp_path) is not None

            # Modify file (update mtime)
            time.sleep(0.1)  # Ensure different mtime
            temp_path.touch()

            # Should detect modification and return None
            assert cache.get(temp_path) is None
        finally:
            temp_path.unlink()

    def test_cache_hot_reload_disabled(self, sample_config):
        """Test that hot reload can be disabled."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test: content")

        try:
            cache = ConfigCache(enable_hot_reload=False)

            # Cache configuration
            cache.set(temp_path, sample_config)
            assert cache.get(temp_path) is not None

            # Modify file (update mtime)
            time.sleep(0.1)
            temp_path.touch()

            # Should still return cached config (hot reload disabled)
            assert cache.get(temp_path) is not None
        finally:
            temp_path.unlink()

    def test_cache_file_deleted(self, sample_config):
        """Test behavior when cached file is deleted."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test: content")

        cache = ConfigCache(enable_hot_reload=True)

        # Cache configuration
        cache.set(temp_path, sample_config)
        assert cache.get(temp_path) is not None

        # Delete file
        temp_path.unlink()

        # Should detect deletion and return None
        assert cache.get(temp_path) is None

    def test_cache_cleanup_expired(self, sample_config):
        """Test cleanup of expired cache entries."""
        cache = ConfigCache(ttl_seconds=0.1)

        # Add configurations
        cache.set("test1.yaml", sample_config)
        cache.set("test2.yaml", sample_config)
        assert cache.size() == 2

        # Wait for expiry
        time.sleep(0.2)

        # Cleanup expired entries
        removed_count = cache.cleanup_expired()
        assert removed_count == 2
        assert cache.size() == 0

    def test_cache_stats(self, sample_config):
        """Test cache statistics."""
        cache = ConfigCache(ttl_seconds=0.1)

        # Add some entries
        cache.set("test1.yaml", sample_config)
        cache.set("test2.yaml", sample_config)

        # Check stats immediately (should be active)
        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["expired_entries"] == 0

        # Wait for expiry
        time.sleep(0.2)

        # Check stats after expiry
        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 0
        assert stats["expired_entries"] == 2


class TestGlobalCache:
    """Test global cache functions."""

    def setup_method(self):
        """Reset global cache before each test."""
        set_global_cache(None)

    def test_get_global_cache(self):
        """Test getting global cache instance."""
        cache = get_global_cache()
        assert isinstance(cache, ConfigCache)

        # Should return same instance
        cache2 = get_global_cache()
        assert cache is cache2

    def test_set_global_cache(self):
        """Test setting global cache instance."""
        custom_cache = ConfigCache(ttl_seconds=600)
        set_global_cache(custom_cache)

        cache = get_global_cache()
        assert cache is custom_cache
        assert cache.get_stats()["ttl_seconds"] == 600

    def test_clear_global_cache(self, sample_config):
        """Test clearing global cache."""
        cache = get_global_cache()
        cache.set("test.yaml", sample_config)
        assert cache.size() == 1

        clear_global_cache()
        assert cache.size() == 0

    def test_disable_caching(self):
        """Test disabling global caching."""
        # Initially enabled
        assert is_caching_enabled() is True

        # Disable caching
        disable_caching()
        assert is_caching_enabled() is False

        # Re-enable by getting cache
        get_global_cache()
        assert is_caching_enabled() is True

    def test_is_caching_enabled(self):
        """Test checking if caching is enabled."""
        assert is_caching_enabled() is True

        disable_caching()
        assert is_caching_enabled() is False

        set_global_cache(ConfigCache())
        assert is_caching_enabled() is True


class TestCacheThreadSafety:
    """Test cache thread safety."""

    def test_concurrent_access(self, sample_config):
        """Test concurrent cache access."""
        import threading
        import time

        cache = ConfigCache()
        errors = []

        def worker(worker_id):
            """Worker function for concurrent access."""
            try:
                # Set configuration
                cache.set(f"test_{worker_id}.yaml", sample_config)

                # Get configuration multiple times
                for _ in range(10):
                    cached = cache.get(f"test_{worker_id}.yaml")
                    assert cached is not None
                    time.sleep(0.001)  # Small delay

                # Invalidate
                cache.invalidate(f"test_{worker_id}.yaml")
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0
        assert cache.size() == 0  # All entries should be invalidated


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config_data = {
        "prompt_id": "test_prompt",
        "prompt_template": "Hello {{name}}, this is a test template",
        "axes": {
            "name": {"type": "categorical", "values": ["Alice", "Bob", "Charlie"]}
        },
        "oracles": {
            "accuracy": {
                "type": "embedding_similarity",
                "canonical_answer": "This is a comprehensive test answer for validation",
                "threshold": 0.8,
            }
        },
    }
    return EvaluationConfig(**config_data)
