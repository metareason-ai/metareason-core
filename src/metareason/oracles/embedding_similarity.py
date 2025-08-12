"""Embedding similarity oracle implementation."""

import asyncio
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from typing import Any, Dict, List, Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from ..adapters.base import LLMAdapter
from ..adapters.registry import AdapterFactory
from ..config.adapters import AdapterConfigType
from ..config.oracles import EmbeddingSimilarityOracle as EmbeddingSimilarityConfig
from .base import BaseOracle, OracleError, OracleResult

logger = logging.getLogger(__name__)


class EmbeddingSimilarityOracle(BaseOracle):
    """Oracle for semantic similarity evaluation using embeddings."""

    def __init__(
        self,
        config: EmbeddingSimilarityConfig,
        adapter_config: Optional[AdapterConfigType] = None,
        primary_adapter: Optional[LLMAdapter] = None,
    ):
        """Initialize embedding similarity oracle.

        Args:
            config: Embedding similarity configuration
            adapter_config: Optional adapter configuration for embeddings
            primary_adapter: Primary adapter to use if no embedding_adapter specified
        """
        super().__init__()
        self.config = config
        self.adapter_config = adapter_config
        self.primary_adapter = primary_adapter
        self.adapter: Optional[LLMAdapter] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None

    async def initialize(self) -> None:
        """Initialize the embedding adapter and thread pool."""
        # Initialize adapter for embeddings
        if self.config.embedding_adapter and self.adapter_config:
            # Use specific embedding adapter if configured
            self.adapter = AdapterFactory.create(self.adapter_config)
            await self.adapter.initialize()
        elif self.primary_adapter:
            # Use primary adapter for embeddings
            self.adapter = self.primary_adapter
        else:
            raise OracleError("No adapter available for embedding generation")

        # Initialize thread pool for parallel processing if requested
        if self.config.parallel_workers:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.config.parallel_workers
            )

    async def cleanup(self) -> None:
        """Cleanup adapter and thread pool resources."""
        if self.adapter and self.config.embedding_adapter:
            # Only cleanup if we created the adapter
            await self.adapter.cleanup()

        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)

    def get_name(self) -> str:
        """Get oracle name."""
        return f"embedding_similarity_{self.config.method}"

    async def evaluate(self, response: str, context: Dict[str, Any]) -> OracleResult:
        """Evaluate response using embedding similarity.

        Args:
            response: Response to evaluate
            context: Context including canonical answer

        Returns:
            Oracle result with similarity score
        """
        if not self.adapter:
            raise OracleError("Embedding adapter not initialized")

        # Get canonical answer
        canonical_answer = self.config.canonical_answer

        # Generate embeddings for both texts
        try:
            canonical_embedding = await self._get_embedding(canonical_answer)
            response_embedding = await self._get_embedding(response)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise OracleError(f"Embedding generation failed: {e}") from e

        # Calculate similarity
        similarity_score = await self._calculate_similarity(
            canonical_embedding, response_embedding
        )

        # Create result based on configuration
        if self.config.confidence_passthrough:
            # Pass through the raw similarity score
            final_score = similarity_score
        else:
            # Apply threshold for binary classification
            final_score = 1.0 if similarity_score >= self.config.threshold else 0.0

        # Build metadata
        metadata = {
            "similarity_score": similarity_score,
            "method": self.config.method,
            "threshold": self.config.threshold,
            "confidence_passthrough": self.config.confidence_passthrough,
            "embedding_model": self.config.embedding_model,
        }

        # Add distribution analysis if requested
        if self.config.distribution_analysis:
            metadata["distribution_analysis"] = (
                await self._analyze_similarity_distribution(
                    canonical_embedding, response_embedding, similarity_score
                )
            )

        return OracleResult(score=final_score, metadata=metadata)

    async def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using the configured model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            OracleError: On embedding generation failure
        """
        # Create embedding request - many providers support this through completion API
        # with special system prompts or use dedicated embedding endpoints

        # For now, we'll use a simple approach that works with most LLM providers
        # In a production system, you'd want dedicated embedding adapter classes

        try:
            # For actual implementation, this would use real embedding APIs
            # For now, we'll create a mock embedding based on text characteristics
            embedding = self._mock_embedding(text)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed for text: {text[:50]}...")
            raise OracleError(f"Failed to generate embedding: {e}") from e

    def _mock_embedding(self, text: str) -> List[float]:
        """Create mock embedding for testing purposes.

        In production, this would be replaced with actual embedding API calls.

        Args:
            text: Text to embed

        Returns:
            Mock embedding vector
        """
        # Create deterministic but varied embeddings based on text content
        # This is just for testing - real implementation would use actual embedding models
        text_hash = hash(text)
        embedding_size = 384  # Common embedding dimension

        embedding = []
        for i in range(embedding_size):
            # Create pseudo-random but deterministic values
            val = ((text_hash + i * 1337) % 10000) / 5000.0 - 1.0
            embedding.append(val)

        # Normalize to unit vector
        magnitude = sqrt(sum(x * x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    async def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0 and 1
        """
        if len(embedding1) != len(embedding2):
            raise OracleError(
                f"Embedding dimensions mismatch: {len(embedding1)} vs {len(embedding2)}"
            )

        if self.config.use_vectorized and NUMPY_AVAILABLE:
            return await self._calculate_similarity_vectorized(embedding1, embedding2)
        else:
            return await self._calculate_similarity_sequential(embedding1, embedding2)

    async def _calculate_similarity_vectorized(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate similarity using vectorized operations.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score
        """

        def _compute():
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            if self.config.method == "cosine_similarity":
                # Cosine similarity: dot product of normalized vectors
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(vec1, vec2) / (norm1 * norm2))

            elif self.config.method == "dot_product":
                # Simple dot product
                return float(np.dot(vec1, vec2))

            elif self.config.method == "euclidean":
                # Convert Euclidean distance to similarity (0-1 range)
                distance = np.linalg.norm(vec1 - vec2)
                # Normalize distance to similarity score
                max_distance = np.sqrt(2 * len(vec1))  # Max possible L2 distance
                return max(0.0, 1.0 - (distance / max_distance))

            elif self.config.method == "semantic_entropy":
                # Semantic entropy - more complex similarity measure
                # Using KL divergence approximation
                vec1_norm = vec1 / (np.sum(np.abs(vec1)) + 1e-8)
                vec2_norm = vec2 / (np.sum(np.abs(vec2)) + 1e-8)

                # Add small epsilon to avoid log(0)
                vec1_norm = vec1_norm + 1e-8
                vec2_norm = vec2_norm + 1e-8

                kl_div = np.sum(vec1_norm * np.log(vec1_norm / vec2_norm))
                return max(0.0, 1.0 / (1.0 + abs(kl_div)))

            else:
                raise OracleError(f"Unknown similarity method: {self.config.method}")

        if self._thread_pool:
            # Run in thread pool for CPU-intensive operations
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self._thread_pool, _compute)
        else:
            return _compute()

    async def _calculate_similarity_sequential(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate similarity using sequential operations (no numpy).

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score
        """
        if self.config.method == "cosine_similarity":
            # Cosine similarity: dot product of normalized vectors
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

            norm1 = sqrt(sum(a * a for a in embedding1))
            norm2 = sqrt(sum(b * b for b in embedding2))

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        elif self.config.method == "dot_product":
            # Simple dot product
            return sum(a * b for a, b in zip(embedding1, embedding2))

        elif self.config.method == "euclidean":
            # Convert Euclidean distance to similarity
            distance_squared = sum((a - b) ** 2 for a, b in zip(embedding1, embedding2))
            distance = sqrt(distance_squared)

            # Normalize to similarity score
            max_distance = sqrt(2 * len(embedding1))
            return max(0.0, 1.0 - (distance / max_distance))

        elif self.config.method == "semantic_entropy":
            # Simplified semantic entropy without numpy
            sum1 = sum(abs(x) for x in embedding1) + 1e-8
            sum2 = sum(abs(x) for x in embedding2) + 1e-8

            # Normalize and compute KL divergence approximation
            kl_div = 0.0
            for a, b in zip(embedding1, embedding2):
                a_norm = (abs(a) + 1e-8) / sum1
                b_norm = (abs(b) + 1e-8) / sum2
                kl_div += a_norm * (a_norm / b_norm if b_norm > 0 else 1.0)

            return max(0.0, 1.0 / (1.0 + abs(kl_div)))

        else:
            raise OracleError(f"Unknown similarity method: {self.config.method}")

    async def _analyze_similarity_distribution(
        self, embedding1: List[float], embedding2: List[float], similarity: float
    ) -> Dict[str, Any]:
        """Analyze similarity distribution for additional insights.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            similarity: Calculated similarity score

        Returns:
            Distribution analysis metadata
        """
        analysis = {}

        try:
            # Component-wise similarities for analysis
            if self.config.use_vectorized and NUMPY_AVAILABLE:
                vec1 = np.array(embedding1)
                vec2 = np.array(embedding2)
                component_similarities = vec1 * vec2

                analysis["component_mean"] = float(np.mean(component_similarities))
                analysis["component_std"] = float(np.std(component_similarities))
                analysis["component_max"] = float(np.max(component_similarities))
                analysis["component_min"] = float(np.min(component_similarities))
            else:
                # Sequential analysis
                component_similarities = [a * b for a, b in zip(embedding1, embedding2)]
                analysis["component_mean"] = statistics.mean(component_similarities)
                analysis["component_std"] = (
                    statistics.stdev(component_similarities)
                    if len(component_similarities) > 1
                    else 0.0
                )
                analysis["component_max"] = max(component_similarities)
                analysis["component_min"] = min(component_similarities)

            # Vector magnitude comparison
            norm1 = sqrt(sum(x * x for x in embedding1))
            norm2 = sqrt(sum(x * x for x in embedding2))

            analysis["magnitude_ratio"] = norm2 / norm1 if norm1 > 0 else 0.0
            analysis["magnitude_difference"] = abs(norm1 - norm2)

            # Similarity confidence based on vector properties
            analysis["confidence_estimate"] = min(
                1.0, similarity + 0.1 * (1.0 - abs(norm1 - norm2))
            )

        except Exception as e:
            logger.warning(f"Distribution analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    async def batch_evaluate(
        self, responses: List[str], context: Dict[str, Any]
    ) -> List[OracleResult]:
        """Evaluate multiple responses efficiently using batch processing.

        Args:
            responses: List of responses to evaluate
            context: Shared context

        Returns:
            List of oracle results
        """
        if not responses:
            return []

        # Process in batches for efficiency
        batch_size = self.config.batch_size
        all_results = []

        for i in range(0, len(responses), batch_size):
            batch = responses[i : i + batch_size]

            if self.config.parallel_workers and len(batch) > 1:
                # Process batch in parallel
                tasks = [self.evaluate(response, context) for response in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Handle exceptions
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Batch evaluation failed for response {i+j}: {result}"
                        )
                        # Create error result
                        batch_results[j] = OracleResult(
                            score=0.0, metadata={"error": str(result)}
                        )
            else:
                # Process batch sequentially
                batch_results = []
                for response in batch:
                    try:
                        result = await self.evaluate(response, context)
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Batch evaluation failed: {e}")
                        batch_results.append(
                            OracleResult(score=0.0, metadata={"error": str(e)})
                        )

            all_results.extend(batch_results)

        return all_results

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
