"""Tests for real posterior samples flowing through to report generators."""

import numpy as np
import pytest

from metareason.config.models import (
    AdapterConfig,
    AxisConfig,
    BayesianAnalysisConfig,
    CalibrateConfig,
    OracleConfig,
    PipelineConfig,
    SamplingConfig,
    SpecConfig,
)
from metareason.config.models import CalibrateConfig

# Import oracle_base directly to avoid adapter import chain
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.runner import SampleResult
from metareason.reporting.calibration_report import CalibrationReportGenerator
from metareason.reporting.report_generator import ReportGenerator


def _make_results(n=5, score=3.5):
    return [
        SampleResult(
            sample_params={"tone": "formal"},
            original_prompt="test prompt",
            final_response="test response",
            evaluations={
                "test_oracle": EvaluationResult(score=score, explanation="ok")
            },
        )
        for _ in range(n)
    ]


def _make_spec():
    return SpecConfig(
        spec_id="test_spec",
        pipeline=[
            PipelineConfig(
                template="t",
                adapter=AdapterConfig(name="ollama"),
                model="m",
                temperature=0.7,
                top_p=0.9,
                max_tokens=100,
            )
        ],
        sampling=SamplingConfig(method="latin_hypercube", optimization="maximin"),
        oracles={
            "test_oracle": OracleConfig(
                type="llm_judge",
                model="m",
                adapter=AdapterConfig(name="ollama"),
                rubric="test",
            )
        },
        axes=[
            AxisConfig(name="tone", type="categorical", values=["formal", "casual"]),
        ],
    )


class TestReportGeneratorPosteriorSamples:
    """Test that ReportGenerator uses real posterior samples when available."""

    def test_uses_real_posterior_samples(self, tmp_path):
        results = _make_results()
        spec = _make_spec()

        # Provide real posterior samples
        rng = np.random.default_rng(123)
        real_quality_samples = rng.normal(4.0, 0.3, 8000).tolist()
        real_noise_samples = np.abs(rng.normal(0.3, 0.1, 8000)).tolist()

        analysis_results = {
            "test_oracle": {
                "population_mean": 4.0,
                "population_median": 4.1,
                "population_std": 0.3,
                "hdi_lower": 3.5,
                "hdi_upper": 4.5,
                "hdi_prob": 0.94,
                "oracle_noise_mean": 0.3,
                "oracle_noise_hdi": (0.1, 0.5),
                "n_samples": 5,
                "posterior_samples": real_quality_samples,
                "noise_posterior_samples": real_noise_samples,
            }
        }

        gen = ReportGenerator(results, spec, analysis_results)
        output = tmp_path / "report.html"
        gen.generate_html(output)

        html = output.read_text()
        assert "MetaReason" in html
        assert len(html) > 1000

    def test_fallback_without_posterior_samples(self, tmp_path):
        """Legacy JSON without posterior_samples should still work."""
        results = _make_results()
        spec = _make_spec()

        # No posterior_samples or noise_posterior_samples keys
        analysis_results = {
            "test_oracle": {
                "population_mean": 4.0,
                "population_median": 4.1,
                "population_std": 0.3,
                "hdi_lower": 3.5,
                "hdi_upper": 4.5,
                "hdi_prob": 0.94,
                "oracle_noise_mean": 0.3,
                "oracle_noise_hdi": (0.1, 0.5),
                "n_samples": 5,
            }
        }

        gen = ReportGenerator(results, spec, analysis_results)
        output = tmp_path / "report.html"
        gen.generate_html(output)

        html = output.read_text()
        assert "MetaReason" in html

    def test_fallback_is_reproducible(self):
        """Fallback with seeded RNG should produce consistent results."""
        results = _make_results()
        spec = _make_spec()

        analysis_results = {
            "test_oracle": {
                "population_mean": 4.0,
                "population_median": 4.1,
                "population_std": 0.3,
                "hdi_lower": 3.5,
                "hdi_upper": 4.5,
                "hdi_prob": 0.94,
                "oracle_noise_mean": 0.3,
                "oracle_noise_hdi": (0.1, 0.5),
                "n_samples": 5,
            }
        }

        gen1 = ReportGenerator(results, spec, analysis_results)
        gen2 = ReportGenerator(results, spec, analysis_results)

        data1 = gen1._generate_chart_data()
        data2 = gen2._generate_chart_data()

        assert data1["test_oracle"]["posterior_x"] == data2["test_oracle"]["posterior_x"]
        assert data1["test_oracle"]["posterior_y"] == data2["test_oracle"]["posterior_y"]


class TestCalibrationReportPosteriorSamples:
    """Test that CalibrationReportGenerator uses real posterior samples."""

    def _make_config(self):
        return CalibrateConfig(
            spec_id="cal_test",
            prompt="test prompt",
            response="test response",
            expected_score=4.0,
            repeats=10,
            oracle=OracleConfig(
                type="llm_judge",
                model="m",
                adapter=AdapterConfig(name="ollama"),
                rubric="test",
            ),
        )

    def test_uses_real_bias_and_noise_samples(self, tmp_path):
        config = self._make_config()
        scores = [3.5, 4.0, 4.5, 3.8, 4.2]

        rng = np.random.default_rng(99)
        analysis = {
            "noise_mean": 0.3,
            "noise_hdi": (0.1, 0.5),
            "n_samples": 5,
            "hdi_prob": 0.94,
            "raw_score_mean": 4.0,
            "raw_score_std": 0.35,
            "expected_score": 4.0,
            "bias_mean": -0.1,
            "bias_median": -0.09,
            "bias_hdi": (-0.3, 0.1),
            "bias_posterior_samples": rng.normal(-0.1, 0.1, 8000).tolist(),
            "noise_posterior_samples": np.abs(rng.normal(0.3, 0.1, 8000)).tolist(),
        }

        gen = CalibrationReportGenerator(config, scores, analysis)
        output = tmp_path / "cal_report.html"
        gen.generate_html(output)

        html = output.read_text()
        assert "MetaReason" in html

    def test_fallback_without_posterior_samples(self, tmp_path):
        config = self._make_config()
        scores = [3.5, 4.0, 4.5]

        # Legacy format — no posterior sample keys
        analysis = {
            "noise_mean": 0.3,
            "noise_hdi": (0.1, 0.5),
            "n_samples": 3,
            "hdi_prob": 0.94,
            "raw_score_mean": 4.0,
            "raw_score_std": 0.4,
            "expected_score": 4.0,
            "bias_mean": -0.1,
            "bias_median": -0.09,
            "bias_hdi": (-0.3, 0.1),
        }

        gen = CalibrationReportGenerator(config, scores, analysis)
        output = tmp_path / "cal_report.html"
        gen.generate_html(output)

        html = output.read_text()
        assert "MetaReason" in html
