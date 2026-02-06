import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from metareason.cli.main import metareason
from metareason.oracles.oracle_base import EvaluationResult
from metareason.pipeline.runner import SampleResult

VALID_SPEC_YAML = """\
spec_id: "test"
pipeline:
  - template: "Hello {{ name }}"
    adapter:
      name: "ollama"
    model: "test-model"
    temperature: 0.7
    top_p: 0.9
    max_tokens: 100
sampling:
  method: "latin_hypercube"
  optimization: "maximin"
n_variants: 1
oracles:
  coherence_judge:
    type: "llm_judge"
    model: "test-model"
    adapter:
      name: "ollama"
    rubric: "Rate 1-5"
"""

INVALID_SPEC_YAML = """\
spec_id: "test"
pipeline: []
sampling:
  method: "latin_hypercube"
  optimization: "maximin"
oracles: {}
"""


def make_sample_result(**overrides):
    defaults = dict(
        sample_params={"tone": "formal", "complexity_level": 5.0},
        original_prompt="test prompt",
        final_response="test response",
        evaluations={
            "coherence_judge": EvaluationResult(score=4.0, explanation="good")
        },
    )
    defaults.update(overrides)
    return SampleResult(**defaults)


def make_results_data():
    return [
        {
            "sample_params": {"tone": "formal", "complexity_level": 5.0},
            "original_prompt": "test prompt",
            "final_response": "test response",
            "evaluations": {"coherence_judge": {"score": 4.0, "explanation": "good"}},
        }
    ]


def mock_population_quality():
    return {
        "population_mean": 4.0,
        "population_median": 4.1,
        "hdi_lower": 3.5,
        "hdi_upper": 4.5,
        "hdi_prob": 0.94,
        "oracle_noise_mean": 0.3,
        "oracle_noise_hdi": (0.1, 0.5),
        "n_samples": 5,
    }


@pytest.fixture
def cli_runner():
    return CliRunner()


# --- validate command ---


class TestValidateCommand:
    def test_validate_valid_spec(self, cli_runner, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)

        result = cli_runner.invoke(metareason, ["validate", str(spec_file)])

        assert result.exit_code == 0
        assert "is valid" in result.output

    def test_validate_invalid_spec(self, cli_runner, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text(INVALID_SPEC_YAML)

        result = cli_runner.invoke(metareason, ["validate", str(spec_file)])

        # validate catches all exceptions and logs, so exit_code is 0
        assert "Failed to load spec" in result.output

    def test_validate_missing_file(self, cli_runner, tmp_path):
        missing = str(tmp_path / "nonexistent.yaml")

        result = cli_runner.invoke(metareason, ["validate", missing])

        # File doesn't exist, load_spec raises, caught by except block
        assert "Failed to load spec" in result.output


# --- run command ---


class TestRunCommand:
    @patch("metareason.cli.main.runner")
    @patch("metareason.cli.main.load_spec")
    def test_run_basic(self, mock_load_spec, mock_runner, cli_runner, tmp_path):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        output_file = tmp_path / "results.json"

        mock_load_spec.return_value = real_load_spec(spec_file)

        mock_run = AsyncMock(return_value=[make_sample_result()])
        mock_runner.run = mock_run

        result = cli_runner.invoke(
            metareason, ["run", str(spec_file), "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert "Completed" in result.output
        assert output_file.exists()

    @patch("metareason.cli.main.runner")
    @patch("metareason.cli.main.load_spec")
    def test_run_with_output_flag(
        self, mock_load_spec, mock_runner, cli_runner, tmp_path
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        output_file = tmp_path / "output.json"

        mock_load_spec.return_value = real_load_spec(spec_file)
        mock_runner.run = AsyncMock(return_value=[make_sample_result()])

        result = cli_runner.invoke(
            metareason, ["run", str(spec_file), "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["original_prompt"] == "test prompt"
        assert "coherence_judge" in data[0]["evaluations"]

    @patch("metareason.cli.main.runner")
    @patch("metareason.cli.main.load_spec")
    def test_run_spec_not_found(
        self, mock_load_spec, mock_runner, cli_runner, tmp_path
    ):
        mock_load_spec.side_effect = FileNotFoundError("spec not found")

        result = cli_runner.invoke(
            metareason, ["run", str(tmp_path / "nonexistent.yaml")]
        )

        assert result.exit_code != 0


# --- analyze command ---


class TestAnalyzeCommand:
    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.load_spec")
    def test_analyze_basic(
        self, mock_load_spec, mock_analyzer_class, cli_runner, tmp_path
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(make_results_data()))

        mock_load_spec.return_value = real_load_spec(spec_file)

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(
            metareason,
            ["analyze", str(results_file), "--spec", str(spec_file)],
        )

        assert result.exit_code == 0
        mock_analyzer.estimate_population_quality.assert_called_once()

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.load_spec")
    def test_analyze_with_oracle_filter(
        self, mock_load_spec, mock_analyzer_class, cli_runner, tmp_path
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(make_results_data()))

        mock_load_spec.return_value = real_load_spec(spec_file)

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(
            metareason,
            [
                "analyze",
                str(results_file),
                "--spec",
                str(spec_file),
                "--oracle",
                "coherence_judge",
            ],
        )

        assert result.exit_code == 0
        mock_analyzer.estimate_population_quality.assert_called_once()

    def test_analyze_invalid_json(self, cli_runner, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        results_file = tmp_path / "bad.json"
        results_file.write_text("not valid json {{{")

        result = cli_runner.invoke(
            metareason,
            ["analyze", str(results_file), "--spec", str(spec_file)],
        )

        assert "Invalid JSON" in result.output

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.load_spec")
    def test_analyze_oracle_not_found(
        self, mock_load_spec, mock_analyzer_class, cli_runner, tmp_path
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(make_results_data()))

        mock_load_spec.return_value = real_load_spec(spec_file)

        result = cli_runner.invoke(
            metareason,
            [
                "analyze",
                str(results_file),
                "--spec",
                str(spec_file),
                "--oracle",
                "nonexistent",
            ],
        )

        assert "not found" in result.output
