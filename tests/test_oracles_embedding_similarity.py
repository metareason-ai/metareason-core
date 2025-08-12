"""Tests for embedding similarity oracle implementation."""

import math
from unittest.mock import AsyncMock, patch

import pytest

from metareason.adapters.base import CompletionResponse, LLMAdapter
from metareason.config.oracles import (
    EmbeddingSimilarityOracle as EmbeddingSimilarityConfig,
)
from metareason.oracles.base import OracleError
from metareason.oracles.embedding_similarity import EmbeddingSimilarityOracle


class TestEmbeddingSimilarityOracle:
    """Test embedding similarity oracle implementation."""

    @pytest.fixture
    def cosine_config(self):
        """Create cosine similarity configuration."""
        return EmbeddingSimilarityConfig(
            canonical_answer="The capital of France is Paris, a major European city.",
            method="cosine_similarity",
            threshold=0.85,
            embedding_model="text-embedding-3-small",
        )

    @pytest.fixture
    def euclidean_config(self):
        """Create Euclidean distance configuration."""
        return EmbeddingSimilarityConfig(
            canonical_answer="Machine learning is a subset of artificial intelligence.",
            method="euclidean",
            threshold=0.80,
            embedding_model="text-embedding-ada-002",
            batch_size=64,
        )

    @pytest.fixture
    def dot_product_config(self):
        """Create dot product similarity configuration."""
        return EmbeddingSimilarityConfig(
            canonical_answer="Climate change affects global temperatures and weather patterns.",
            method="dot_product",
            threshold=0.75,
            embedding_model="text-embedding-3-large",
            use_vectorized=True,
            parallel_workers=4,
        )

    @pytest.fixture
    def semantic_entropy_config(self):
        """Create semantic entropy configuration."""
        return EmbeddingSimilarityConfig(
            canonical_answer="Quantum computing uses quantum mechanics principles.",
            method="semantic_entropy",
            threshold=0.70,
            embedding_model="text-embedding-3-small",
            confidence_passthrough=True,
            distribution_analysis=True,
        )

    @pytest.fixture
    def mock_adapter(self):
        """Create mock LLM adapter."""
        adapter = AsyncMock(spec=LLMAdapter)
        adapter.complete.return_value = CompletionResponse(
            content="[0.1, 0.2, 0.3, 0.4, 0.5]",
            model="text-embedding-3-small",
        )
        return adapter

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        return {
            "canonical": [0.1, 0.2, 0.3, 0.4, 0.5],
            "similar": [0.15, 0.25, 0.35, 0.45, 0.55],  # Similar to canonical
            "different": [0.9, 0.8, 0.7, 0.6, 0.5],  # Different from canonical
            "identical": [0.1, 0.2, 0.3, 0.4, 0.5],  # Identical to canonical
        }

    async def test_initialization_with_primary_adapter(
        self, cosine_config, mock_adapter
    ):
        """Test oracle initialization with primary adapter."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        await oracle.initialize()
        assert oracle.adapter == mock_adapter
        assert oracle.get_name() == "embedding_similarity_cosine_similarity"

    async def test_initialization_without_adapter_fails(self, cosine_config):
        """Test oracle initialization fails without adapter."""
        oracle = EmbeddingSimilarityOracle(config=cosine_config)

        with pytest.raises(OracleError, match="No adapter available"):
            await oracle.initialize()

    @pytest.mark.parametrize(
        "method,expected_similarity",
        [
            ("cosine_similarity", 0.9998),  # Very similar vectors
            ("dot_product", 0.55),  # Sum of products
            ("euclidean", 1.0),  # Same vectors = max similarity
        ],
    )
    async def test_similarity_methods_identical_vectors(
        self, method, expected_similarity, mock_adapter, sample_embeddings
    ):
        """Test similarity calculations with identical vectors."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method=method,
            threshold=0.9,  # High threshold so dot product doesn't pass
            embedding_model="test-model",
            confidence_passthrough=True,  # Get raw similarity scores
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["identical"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            # With confidence passthrough, score equals similarity_score
            assert abs(result.score - expected_similarity) < 0.001
            assert result.score == result.metadata["similarity_score"]
            assert result.metadata["method"] == method

    @pytest.mark.parametrize(
        "method", ["cosine_similarity", "euclidean", "dot_product", "semantic_entropy"]
    )
    async def test_all_similarity_methods(
        self, method, mock_adapter, sample_embeddings
    ):
        """Test all similarity methods produce valid scores."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method=method,
            threshold=0.5,
            embedding_model="test-model",
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["similar"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.metadata["similarity_score"] <= 1.0
            assert result.metadata["method"] == method

    async def test_threshold_application(self, mock_adapter, sample_embeddings):
        """Test threshold application for binary classification."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.9,  # High threshold
            embedding_model="test-model",
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            # Test above threshold
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["identical"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            assert result.score == 1.0  # Above threshold
            assert result.metadata["similarity_score"] > 0.9

            # Test below threshold
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["different"],
            ]

            result = await oracle.evaluate("Different response", {})

            # Should be 0.0 due to threshold
            assert result.score == 0.0
            assert result.metadata["similarity_score"] < 0.9

    async def test_confidence_passthrough(self, mock_adapter, sample_embeddings):
        """Test confidence passthrough mode."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.9,
            embedding_model="test-model",
            confidence_passthrough=True,
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["similar"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            # Score should be raw similarity, not thresholded
            assert result.score == result.metadata["similarity_score"]
            assert 0.0 < result.score < 1.0  # Not binary

    async def test_distribution_analysis(self, mock_adapter, sample_embeddings):
        """Test similarity distribution analysis."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.8,
            embedding_model="test-model",
            distribution_analysis=True,
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["similar"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            assert "distribution_analysis" in result.metadata
            analysis = result.metadata["distribution_analysis"]

            # Check analysis contains expected fields
            assert "component_mean" in analysis
            assert "component_std" in analysis
            assert "component_max" in analysis
            assert "component_min" in analysis
            assert "magnitude_ratio" in analysis
            assert "confidence_estimate" in analysis

    @pytest.mark.parametrize("use_vectorized", [True, False])
    async def test_vectorized_vs_sequential(
        self, use_vectorized, mock_adapter, sample_embeddings
    ):
        """Test vectorized vs sequential similarity calculations."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.8,
            embedding_model="test-model",
            use_vectorized=use_vectorized,
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["similar"],
            ]

            await oracle.initialize()
            result = await oracle.evaluate("Test response", {})

            # Both should produce similar results
            assert 0.0 <= result.score <= 1.0
            assert result.metadata["method"] == "cosine_similarity"

    async def test_batch_evaluation(
        self, cosine_config, mock_adapter, sample_embeddings
    ):
        """Test batch evaluation functionality."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        responses = ["Response 1", "Response 2", "Response 3"]

        with patch.object(oracle, "_get_embedding") as mock_embed:
            # Return embeddings for canonical + 3 responses
            mock_embed.side_effect = [
                sample_embeddings["canonical"],  # For canonical (cached)
                sample_embeddings["similar"],  # Response 1
                sample_embeddings["canonical"],  # For canonical again
                sample_embeddings["different"],  # Response 2
                sample_embeddings["canonical"],  # For canonical again
                sample_embeddings["identical"],  # Response 3
            ]

            await oracle.initialize()
            results = await oracle.batch_evaluate(responses, {})

            assert len(results) == 3
            for result in results:
                assert isinstance(result.score, (int, float))
                assert 0.0 <= result.score <= 1.0
                assert "similarity_score" in result.metadata

    async def test_embedding_generation_failure(self, cosine_config, mock_adapter):
        """Test handling of embedding generation failures."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = Exception("Embedding failed")

            await oracle.initialize()

            with pytest.raises(OracleError, match="Embedding generation failed"):
                await oracle.evaluate("Test response", {})

    async def test_dimension_mismatch_error(self, cosine_config, mock_adapter):
        """Test handling of embedding dimension mismatch."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                [0.1, 0.2, 0.3],  # 3 dimensions
                [0.1, 0.2, 0.3, 0.4],  # 4 dimensions
            ]

            await oracle.initialize()

            with pytest.raises(OracleError, match="Embedding dimensions mismatch"):
                await oracle.evaluate("Test response", {})

    async def test_mock_embedding_generation(self, cosine_config):
        """Test mock embedding generation for consistent testing."""
        oracle = EmbeddingSimilarityOracle(config=cosine_config)

        # Test deterministic mock embeddings
        text1 = "Hello world"
        text2 = "Hello world"
        text3 = "Goodbye world"

        embedding1 = oracle._mock_embedding(text1)
        embedding2 = oracle._mock_embedding(text2)
        embedding3 = oracle._mock_embedding(text3)

        # Same text should produce identical embeddings
        assert embedding1 == embedding2

        # Different text should produce different embeddings
        assert embedding1 != embedding3

        # All embeddings should be normalized (unit vectors)
        magnitude1 = math.sqrt(sum(x * x for x in embedding1))
        magnitude3 = math.sqrt(sum(x * x for x in embedding3))

        assert abs(magnitude1 - 1.0) < 0.001
        assert abs(magnitude3 - 1.0) < 0.001

    async def test_context_manager(self, cosine_config, mock_adapter):
        """Test async context manager functionality."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        async with oracle:
            assert oracle.adapter == mock_adapter

            # Should be able to evaluate within context
            with patch.object(oracle, "_get_embedding") as mock_embed:
                mock_embed.side_effect = [[0.1, 0.2], [0.1, 0.2]]
                result = await oracle.evaluate("Test", {})
                assert result.score >= 0.0

    @pytest.mark.parametrize("parallel_workers", [None, 1, 4])
    async def test_parallel_processing_config(
        self, parallel_workers, mock_adapter, sample_embeddings
    ):
        """Test different parallel processing configurations."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.8,
            embedding_model="test-model",
            parallel_workers=parallel_workers,
        )

        oracle = EmbeddingSimilarityOracle(config=config, primary_adapter=mock_adapter)

        with patch.object(oracle, "_get_embedding") as mock_embed:
            mock_embed.side_effect = [
                sample_embeddings["canonical"],
                sample_embeddings["similar"],
            ]

            await oracle.initialize()

            if parallel_workers:
                assert oracle._thread_pool is not None
                assert oracle._thread_pool._max_workers == parallel_workers
            else:
                assert oracle._thread_pool is None

            result = await oracle.evaluate("Test response", {})
            assert 0.0 <= result.score <= 1.0

    async def test_cleanup_behavior(self, cosine_config, mock_adapter):
        """Test proper cleanup of resources."""
        oracle = EmbeddingSimilarityOracle(
            config=cosine_config, primary_adapter=mock_adapter
        )

        await oracle.initialize()
        await oracle.cleanup()

        # Should not cleanup primary adapter (not owned by oracle)
        mock_adapter.cleanup.assert_not_called()

    async def test_embedding_adapter_cleanup(self, cosine_config):
        """Test cleanup when oracle owns the embedding adapter."""
        mock_embedding_adapter = AsyncMock(spec=LLMAdapter)

        config = EmbeddingSimilarityConfig(
            canonical_answer="Test answer",
            method="cosine_similarity",
            threshold=0.8,
            embedding_model="test-model",
            embedding_adapter="openai",  # Indicates separate embedding adapter
        )

        oracle = EmbeddingSimilarityOracle(config=config)
        oracle.adapter = mock_embedding_adapter  # Simulate owned adapter

        await oracle.cleanup()

        # Should cleanup the embedding adapter when oracle owns it
        mock_embedding_adapter.cleanup.assert_called_once()


class TestEmbeddingSimilarityConfigValidation:
    """Test embedding similarity oracle configuration validation."""

    def test_valid_configurations(self):
        """Test valid configuration creation."""
        # Minimal valid config
        config = EmbeddingSimilarityConfig(
            canonical_answer="This is a sufficiently long canonical answer for testing.",
            threshold=0.85,
        )
        assert config.method == "cosine_similarity"  # Default
        assert config.embedding_model == "text-embedding-3-small"  # Default
        assert config.batch_size == 32  # Default
        assert config.use_vectorized is True  # Default
        assert config.confidence_passthrough is False  # Default

    def test_all_similarity_methods(self):
        """Test all supported similarity methods."""
        methods = ["cosine_similarity", "euclidean", "dot_product", "semantic_entropy"]

        for method in methods:
            config = EmbeddingSimilarityConfig(
                canonical_answer="Valid canonical answer for testing purposes.",
                method=method,
                threshold=0.8,
            )
            assert config.method == method

    def test_invalid_canonical_answer(self):
        """Test validation of canonical answer."""
        # Empty canonical answer
        with pytest.raises(ValueError, match="Canonical answer cannot be empty"):
            EmbeddingSimilarityConfig(
                canonical_answer="",
                threshold=0.8,
            )

        # Too short canonical answer
        with pytest.raises(ValueError, match="Canonical answer seems too short"):
            EmbeddingSimilarityConfig(
                canonical_answer="Short",
                threshold=0.8,
            )

    def test_invalid_threshold(self):
        """Test threshold validation."""
        from pydantic import ValidationError

        # Threshold too low
        with pytest.raises(ValidationError):
            EmbeddingSimilarityConfig(
                canonical_answer="Valid canonical answer for testing purposes.",
                threshold=-0.1,
            )

        # Threshold too high
        with pytest.raises(ValidationError):
            EmbeddingSimilarityConfig(
                canonical_answer="Valid canonical answer for testing purposes.",
                threshold=1.5,
            )

    def test_invalid_embedding_adapter(self):
        """Test embedding adapter validation."""
        with pytest.raises(
            ValueError, match="Embedding adapter cannot be empty string"
        ):
            EmbeddingSimilarityConfig(
                canonical_answer="Valid canonical answer for testing purposes.",
                threshold=0.8,
                embedding_adapter="",  # Empty string
            )

    @pytest.mark.parametrize("batch_size", [1, 32, 100, 1000])
    def test_valid_batch_sizes(self, batch_size):
        """Test valid batch size configurations."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Valid canonical answer for testing purposes.",
            threshold=0.8,
            batch_size=batch_size,
        )
        assert config.batch_size == batch_size

    @pytest.mark.parametrize("parallel_workers", [1, 8, 16, 32])
    def test_valid_parallel_workers(self, parallel_workers):
        """Test valid parallel worker configurations."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Valid canonical answer for testing purposes.",
            threshold=0.8,
            parallel_workers=parallel_workers,
        )
        assert config.parallel_workers == parallel_workers

    def test_performance_options(self):
        """Test performance-related configuration options."""
        config = EmbeddingSimilarityConfig(
            canonical_answer="Valid canonical answer for testing purposes.",
            threshold=0.8,
            use_vectorized=False,
            parallel_workers=8,
            confidence_passthrough=True,
            distribution_analysis=True,
        )

        assert config.use_vectorized is False
        assert config.parallel_workers == 8
        assert config.confidence_passthrough is True
        assert config.distribution_analysis is True
