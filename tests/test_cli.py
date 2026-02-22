import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from click.testing import CliRunner
from rich.console import Console

from metareason.cli.main import (
    _run_bayesian_analysis,
    display_bayesian_analysis,
    metareason,
)
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
        "population_std": 0.3,
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


# --- report command ---


class TestReportCommand:
    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.load_spec")
    def test_report_standalone_command(
        self, mock_load_spec, mock_analyzer_class, cli_runner, tmp_path
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        results_file = tmp_path / "results.json"
        results_file.write_text(json.dumps(make_results_data()))
        report_file = tmp_path / "report.html"

        mock_load_spec.return_value = real_load_spec(spec_file)

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        with patch("metareason.reporting.ReportGenerator") as mock_report_gen:
            mock_generator = MagicMock()
            mock_report_gen.return_value = mock_generator

            result = cli_runner.invoke(
                metareason,
                [
                    "report",
                    str(results_file),
                    "--spec",
                    str(spec_file),
                    "--output",
                    str(report_file),
                ],
            )

            assert result.exit_code == 0
            assert "HTML report saved" in result.output
            mock_generator.generate_html.assert_called_once()

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.load_spec")
    def test_analyze_with_report_flag(
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

        with patch("metareason.reporting.ReportGenerator") as mock_report_gen:
            mock_generator = MagicMock()
            mock_report_gen.return_value = mock_generator

            result = cli_runner.invoke(
                metareason,
                [
                    "analyze",
                    str(results_file),
                    "--spec",
                    str(spec_file),
                    "--report",
                ],
            )

            assert result.exit_code == 0
            assert "HTML report saved" in result.output
            mock_generator.generate_html.assert_called_once()

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.runner")
    @patch("metareason.cli.main.load_spec")
    def test_run_with_report_flag(
        self,
        mock_load_spec,
        mock_runner,
        mock_analyzer_class,
        cli_runner,
        tmp_path,
    ):
        from metareason.pipeline.loader import load_spec as real_load_spec

        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        output_file = tmp_path / "results.json"

        mock_load_spec.return_value = real_load_spec(spec_file)
        mock_runner.run = AsyncMock(return_value=[make_sample_result()])

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        with patch("metareason.reporting.ReportGenerator") as mock_report_gen:
            mock_generator = MagicMock()
            mock_report_gen.return_value = mock_generator

            result = cli_runner.invoke(
                metareason,
                [
                    "run",
                    str(spec_file),
                    "--output",
                    str(output_file),
                    "--analyze",
                    "--report",
                ],
            )

            assert result.exit_code == 0
            assert "HTML report saved" in result.output
            mock_report_gen.assert_called_once()

    def test_report_missing_results_file(self, cli_runner, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text(VALID_SPEC_YAML)
        missing_results = str(tmp_path / "nonexistent.json")

        result = cli_runner.invoke(
            metareason,
            ["report", missing_results, "--spec", str(spec_file)],
        )

        # Click's exists=True validation catches missing file
        assert result.exit_code != 0


# --- calibrate command ---

VALID_CALIBRATE_YAML = """\
spec_id: "cal-test"
type: calibrate
prompt: "Test prompt"
response: "Test response"
repeats: 5
oracle:
  type: "llm_judge"
  model: "test-model"
  adapter:
    name: "ollama"
  rubric: "Rate 1-5"
analysis:
  hdi_probability: 0.94
  mcmc_draws: 200
  mcmc_chains: 2
  mcmc_tune: 100
"""

VALID_CALIBRATE_WITH_EXPECTED_YAML = """\
spec_id: "cal-expected"
type: calibrate
prompt: "Test prompt"
response: "Test response"
expected_score: 4.0
repeats: 5
oracle:
  type: "llm_judge"
  model: "test-model"
  adapter:
    name: "ollama"
  rubric: "Rate 1-5"
analysis:
  hdi_probability: 0.94
  mcmc_draws: 200
  mcmc_chains: 2
  mcmc_tune: 100
"""


class TestCalibrateCommand:
    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_basic(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(
            return_value=EvaluationResult(score=4.0, explanation="good")
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(metareason, ["calibrate", str(spec_file)])

        assert result.exit_code == 0
        assert "Judge Calibration" in result.output
        assert "confident" in result.output
        assert mock_judge.evaluate.call_count == 5

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_with_expected_score(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_WITH_EXPECTED_YAML)

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(
            return_value=EvaluationResult(score=4.0, explanation="good")
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(metareason, ["calibrate", str(spec_file)])

        assert result.exit_code == 0
        assert "Accuracy" in result.output
        assert "Bias" in result.output

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_with_output(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)
        output_file = tmp_path / "cal_results.json"

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(
            return_value=EvaluationResult(score=4.0, explanation="good")
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(
            metareason, ["calibrate", str(spec_file), "-o", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        data = json.loads(output_file.read_text())
        assert data["spec_id"] == "cal-test"
        assert data["repeats"] == 5
        assert len(data["scores"]) == 5
        assert "analysis" in data

    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_all_evaluations_fail(
        self, mock_judge_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        mock_judge_class.return_value = mock_judge

        result = cli_runner.invoke(metareason, ["calibrate", str(spec_file)])

        assert result.exit_code == 0  # graceful exit, not a crash
        assert "All evaluations failed" in result.output

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_partial_failures(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)

        mock_judge = MagicMock()
        # 3 successes, 2 failures out of 5 repeats
        mock_judge.evaluate = AsyncMock(
            side_effect=[
                EvaluationResult(score=4.0, explanation="good"),
                RuntimeError("timeout"),
                EvaluationResult(score=3.5, explanation="ok"),
                EvaluationResult(score=4.5, explanation="great"),
                RuntimeError("timeout"),
            ]
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        result = cli_runner.invoke(metareason, ["calibrate", str(spec_file)])

        assert result.exit_code == 0
        assert "3/5 evaluations" in result.output
        assert "2 failed" in result.output

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_with_report(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(
            return_value=EvaluationResult(score=4.0, explanation="good")
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        with patch(
            "metareason.reporting.CalibrationReportGenerator"
        ) as mock_report_gen:
            mock_generator = MagicMock()
            mock_report_gen.return_value = mock_generator

            result = cli_runner.invoke(
                metareason, ["calibrate", str(spec_file), "--report"]
            )

            assert result.exit_code == 0
            assert "HTML report saved" in result.output
            mock_report_gen.assert_called_once()
            mock_generator.generate_html.assert_called_once()

    @patch("metareason.cli.main.BayesianAnalyzer")
    @patch("metareason.cli.main.LLMJudge")
    def test_calibrate_with_report_and_output(
        self, mock_judge_class, mock_analyzer_class, cli_runner, tmp_path
    ):
        spec_file = tmp_path / "cal.yaml"
        spec_file.write_text(VALID_CALIBRATE_YAML)
        output_file = tmp_path / "cal_results.json"

        mock_judge = MagicMock()
        mock_judge.evaluate = AsyncMock(
            return_value=EvaluationResult(score=4.0, explanation="good")
        )
        mock_judge_class.return_value = mock_judge

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        with patch(
            "metareason.reporting.CalibrationReportGenerator"
        ) as mock_report_gen:
            mock_generator = MagicMock()
            mock_report_gen.return_value = mock_generator

            result = cli_runner.invoke(
                metareason,
                [
                    "calibrate",
                    str(spec_file),
                    "-o",
                    str(output_file),
                    "--report",
                ],
            )

            assert result.exit_code == 0
            assert "HTML report saved" in result.output
            # Report path should derive from output path
            call_args = mock_generator.generate_html.call_args[0]
            assert str(call_args[0]).endswith(".html")

    def test_calibrate_invalid_spec(self, cli_runner, tmp_path):
        spec_file = tmp_path / "bad_cal.yaml"
        spec_file.write_text("spec_id: bad\ntype: calibrate\n")

        result = cli_runner.invoke(metareason, ["calibrate", str(spec_file)])

        assert result.exit_code != 0


# --- _run_bayesian_analysis helper ---


class TestRunBayesianAnalysis:
    """Tests for the extracted _run_bayesian_analysis helper function."""

    @patch("metareason.cli.main.BayesianAnalyzer")
    def test_returns_analysis_results_for_all_oracles(self, mock_analyzer_class):
        import tempfile
        from pathlib import Path

        from metareason.pipeline.loader import load_spec as real_load_spec

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(VALID_SPEC_YAML)
            spec_path = Path(f.name)

        spec_config = real_load_spec(spec_path)
        results = [make_sample_result()]

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        test_console = Console(file=open("/dev/null", "w"))
        analysis_results = _run_bayesian_analysis(spec_config, results, test_console)

        assert isinstance(analysis_results, dict)
        assert "coherence_judge" in analysis_results
        assert analysis_results["coherence_judge"]["population_mean"] == 4.0
        mock_analyzer.estimate_population_quality.assert_called_once_with(
            "coherence_judge", hdi_prob=0.94
        )

    @patch("metareason.cli.main.BayesianAnalyzer")
    def test_handles_oracle_analysis_error_gracefully(self, mock_analyzer_class):
        import tempfile
        from pathlib import Path

        from metareason.pipeline.loader import load_spec as real_load_spec

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(VALID_SPEC_YAML)
            spec_path = Path(f.name)

        spec_config = real_load_spec(spec_path)
        results = [make_sample_result()]

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.side_effect = RuntimeError(
            "MCMC failed"
        )
        mock_analyzer_class.return_value = mock_analyzer

        test_console = Console(file=open("/dev/null", "w"))
        analysis_results = _run_bayesian_analysis(spec_config, results, test_console)

        # Should return empty dict, not raise
        assert isinstance(analysis_results, dict)
        assert len(analysis_results) == 0

    @patch("metareason.cli.main.BayesianAnalyzer")
    def test_uses_default_hdi_when_no_analysis_config(self, mock_analyzer_class):
        import tempfile
        from pathlib import Path

        from metareason.pipeline.loader import load_spec as real_load_spec

        # Spec without analysis section
        spec_yaml = VALID_SPEC_YAML
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(spec_yaml)
            spec_path = Path(f.name)

        spec_config = real_load_spec(spec_path)
        # Ensure no analysis config
        spec_config.analysis = None
        results = [make_sample_result()]

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        test_console = Console(file=open("/dev/null", "w"))
        _run_bayesian_analysis(spec_config, results, test_console)

        mock_analyzer.estimate_population_quality.assert_called_once_with(
            "coherence_judge", hdi_prob=0.94
        )

    @patch("metareason.cli.main.BayesianAnalyzer")
    def test_accepts_custom_oracle_names(self, mock_analyzer_class):
        import tempfile
        from pathlib import Path

        from metareason.pipeline.loader import load_spec as real_load_spec

        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(VALID_SPEC_YAML)
            spec_path = Path(f.name)

        spec_config = real_load_spec(spec_path)
        results = [make_sample_result()]

        mock_analyzer = MagicMock()
        mock_analyzer.estimate_population_quality.return_value = (
            mock_population_quality()
        )
        mock_analyzer_class.return_value = mock_analyzer

        test_console = Console(file=open("/dev/null", "w"))
        analysis_results = _run_bayesian_analysis(
            spec_config, results, test_console, oracle_names=["coherence_judge"]
        )

        assert "coherence_judge" in analysis_results


# --- display_bayesian_analysis with non-default HDI ---


class TestDisplayBayesianAnalysis:
    def _make_summary_df(self, hdi_low_col, hdi_high_col):
        """Build a mock ArviZ summary DataFrame with given HDI column names."""
        index = ["oracle_noise", "true_quality[0]"]
        data = {
            "mean": [0.3, 4.0],
            "sd": [0.1, 0.2],
            hdi_low_col: [0.1, 3.5],
            hdi_high_col: [0.5, 4.5],
            "r_hat": [1.0, 1.0],
            "ess_bulk": [1000.0, 1000.0],
            "ess_tail": [900.0, 900.0],
        }
        return pd.DataFrame(data, index=index)

    def _make_results(self):
        return [
            SampleResult(
                sample_params={"tone": "formal"},
                original_prompt="test",
                final_response="test",
                evaluations={
                    "test_oracle": EvaluationResult(score=4.0, explanation="ok")
                },
            )
        ]

    @patch("metareason.cli.main.az")
    def test_display_with_90_hdi(self, mock_az):
        """display_bayesian_analysis should not crash with hdi_5%/hdi_95% columns (90% HDI)."""
        summary_df = self._make_summary_df("hdi_5%", "hdi_95%")
        mock_az.summary.return_value = summary_df

        mock_idata = MagicMock()
        results = self._make_results()

        # Should not raise KeyError
        display_bayesian_analysis(mock_idata, "test_oracle", results, hdi_prob=0.90)

    @patch("metareason.cli.main.az")
    def test_display_with_94_hdi(self, mock_az):
        """display_bayesian_analysis should work with default 94% HDI columns."""
        summary_df = self._make_summary_df("hdi_3%", "hdi_97%")
        mock_az.summary.return_value = summary_df

        mock_idata = MagicMock()
        results = self._make_results()

        display_bayesian_analysis(mock_idata, "test_oracle", results, hdi_prob=0.94)

    @patch("metareason.cli.main.az")
    def test_display_with_80_hdi(self, mock_az):
        """display_bayesian_analysis should work with 80% HDI columns."""
        summary_df = self._make_summary_df("hdi_10%", "hdi_90%")
        mock_az.summary.return_value = summary_df

        mock_idata = MagicMock()
        results = self._make_results()

        display_bayesian_analysis(mock_idata, "test_oracle", results, hdi_prob=0.80)
