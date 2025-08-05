"""Configuration caching system for MetaReason."""

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .models import EvaluationConfig


class ConfigCache:
    """Thread-safe configuration cache with TTL and file watching."""

    def __init__(self, ttl_seconds: int = 300, enable_hot_reload: bool = True):
        """Initialize the configuration cache.

        Args:
            ttl_seconds: Time-to-live for cached configurations in seconds
            enable_hot_reload: Whether to enable hot-reload detection
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._ttl_seconds = ttl_seconds
        self._enable_hot_reload = enable_hot_reload

    def get(self, file_path: Union[str, Path]) -> Optional[EvaluationConfig]:
        """Get a cached configuration if available and valid.

        Args:
            file_path: Path to the configuration file

        Returns:
            Cached EvaluationConfig if valid, None otherwise
        """
        path = Path(file_path).resolve()
        cache_key = str(path)

        with self._lock:
            if cache_key not in self._cache:
                return None

            cache_entry = self._cache[cache_key]
            current_time = time.time()

            # Check TTL expiry
            if current_time - cache_entry["timestamp"] > self._ttl_seconds:
                del self._cache[cache_key]
                return None

            # Check file modification if hot-reload is enabled
            if self._enable_hot_reload and path.exists():
                try:
                    current_mtime = path.stat().st_mtime
                    if current_mtime != cache_entry["mtime"]:
                        del self._cache[cache_key]
                        return None
                except (OSError, IOError):
                    # File might have been deleted or become inaccessible
                    del self._cache[cache_key]
                    return None
            elif (
                self._enable_hot_reload
                and not path.exists()
                and cache_entry["mtime"] != 0.0
            ):
                # File was deleted
                del self._cache[cache_key]
                return None

            return cache_entry["config"]

    def set(self, file_path: Union[str, Path], config: EvaluationConfig) -> None:
        """Cache a configuration.

        Args:
            file_path: Path to the configuration file
            config: Configuration to cache
        """
        path = Path(file_path).resolve()
        cache_key = str(path)

        try:
            mtime = (
                path.stat().st_mtime
                if self._enable_hot_reload and path.exists()
                else 0.0
            )
        except (OSError, IOError):
            mtime = 0.0

        with self._lock:
            self._cache[cache_key] = {
                "config": config,
                "timestamp": time.time(),
                "mtime": mtime,
            }

    def invalidate(self, file_path: Union[str, Path]) -> bool:
        """Invalidate a specific cached configuration.

        Args:
            file_path: Path to the configuration file

        Returns:
            True if an entry was removed, False if it wasn't cached
        """
        path = Path(file_path).resolve()
        cache_key = str(path)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cached configurations."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get the current number of cached configurations."""
        with self._lock:
            return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for cache_key, cache_entry in self._cache.items():
                if current_time - cache_entry["timestamp"] > self._ttl_seconds:
                    expired_keys.append(cache_key)

            for key in expired_keys:
                del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        with self._lock:
            current_time = time.time()
            active_entries = 0
            expired_entries = 0

            for cache_entry in self._cache.values():
                if current_time - cache_entry["timestamp"] > self._ttl_seconds:
                    expired_entries += 1
                else:
                    active_entries += 1

            return {
                "total_entries": len(self._cache),
                "active_entries": active_entries,
                "expired_entries": expired_entries,
                "ttl_seconds": self._ttl_seconds,
                "hot_reload_enabled": self._enable_hot_reload,
            }


# Global cache instance
_global_cache: Optional[ConfigCache] = None
_cache_lock = threading.Lock()


def get_global_cache() -> ConfigCache:
    """Get the global configuration cache instance."""
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = ConfigCache()

    return _global_cache


def set_global_cache(cache: Optional[ConfigCache]) -> None:
    """Set the global configuration cache instance.

    Args:
        cache: Cache instance to set, or None to create a new default cache
    """
    global _global_cache

    with _cache_lock:
        _global_cache = cache or ConfigCache()


def clear_global_cache() -> None:
    """Clear the global configuration cache."""
    cache = get_global_cache()
    cache.clear()


def disable_caching() -> None:
    """Disable global caching by setting cache to None."""
    global _global_cache

    with _cache_lock:
        _global_cache = None


def is_caching_enabled() -> bool:
    """Check if global caching is enabled."""
    return _global_cache is not None
