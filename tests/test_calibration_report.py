from metareason.config.models import (
    AdapterConfig,
    BayesianAnalysisConfig,
    CalibrateConfig,
    OracleConfig,
)
from metareason.reporting.calibration_report import CalibrationReportGenerator


def make_calibrate_config(**overrides):
    defaults = dict(
        spec_id="cal-test",
        type="calibrate",
        prompt="What is 2+2?",
        response="The answer is 4.",
        repeats=10,
        oracle=OracleConfig(
            type="llm_judge",
            model="test-model",
            adapter=AdapterConfig(name="ollama"),
            rubric="Rate 1-5",
        ),
        analysis=BayesianAnalysisConfig(hdi_probability=0.94),
    )
    defaults.update(overrides)
    return CalibrateConfig(**defaults)


def make_analysis_result(expected_score=None):
    """Create mock judge calibration result (new structure)."""
    result = {
        "noise_mean": 0.3,
        "noise_hdi": (0.1, 0.5),
        "n_samples": 10,
        "hdi_prob": 0.94,
        "raw_score_mean": 4.0,
        "raw_score_std": 0.3,
    }
    if expected_score is not None:
        result["expected_score"] = expected_score
        result["bias_mean"] = 4.0 - expected_score
        result["bias_median"] = 4.0 - expected_score
        result["bias_hdi"] = (
            4.0 - expected_score - 0.2,
            4.0 - expected_score + 0.2,
        )
    else:
        result["estimated_quality_mean"] = 4.0
        result["estimated_quality_hdi"] = (3.5, 4.5)
    return result


class TestCalibrationReportGenerator:
    def test_generates_html(self, tmp_path):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        result = generator.generate_html(output_path)

        assert result == output_path
        assert output_path.exists()

        html = output_path.read_text()
        assert "MetaReason" in html
        assert "Calibration Report" in html
        assert "cal-test" in html
        assert "Judge Assessment" in html
        assert "histogramChart" in html
        assert "noiseChart" in html
        assert "test-model" in html

    def test_with_expected_score(self, tmp_path):
        config = make_calibrate_config(expected_score=4.0)
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result(expected_score=4.0)

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        generator.generate_html(output_path)

        html = output_path.read_text()
        assert "Judge Assessment" in html
        assert "Accuracy (bias)" in html
        assert "biasChart" in html

    def test_without_expected_score(self, tmp_path):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        generator.generate_html(output_path)

        html = output_path.read_text()
        assert "biasChart" not in html  # No bias chart without expected_score
        assert "Estimated Quality" in html

    def test_chart_data_structure_without_expected(self):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        chart_data = generator._generate_chart_data()

        # Score histogram data
        assert "histogram_labels" in chart_data
        assert "histogram_counts" in chart_data
        assert chart_data["histogram_labels"] == ["1", "2", "3", "4", "5"]
        assert len(chart_data["histogram_counts"]) == 5
        assert "score_mean" in chart_data

        # Noise KDE data
        assert "noise_x" in chart_data
        assert "noise_y" in chart_data
        assert len(chart_data["noise_x"]) == 80
        assert len(chart_data["noise_y"]) == 80

        # Noise annotation values
        assert "noise_mean" in chart_data
        assert "noise_hdi_lower" in chart_data
        assert "noise_hdi_upper" in chart_data

        # No bias data without expected_score
        assert "bias_x" not in chart_data
        assert chart_data["has_expected"] is False

    def test_chart_data_structure_with_expected(self):
        config = make_calibrate_config(expected_score=4.0)
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result(expected_score=4.0)

        generator = CalibrationReportGenerator(config, scores, analysis)
        chart_data = generator._generate_chart_data()

        # Bias KDE data should be present
        assert "bias_x" in chart_data
        assert "bias_y" in chart_data
        assert "bias_mean" in chart_data
        assert "bias_hdi_lower" in chart_data
        assert "bias_hdi_upper" in chart_data
        assert chart_data["has_expected"] is True

    def test_collect_data_with_expected(self):
        config = make_calibrate_config(expected_score=4.0)
        scores = [4.0, 3.5, 4.5]
        analysis = make_analysis_result(expected_score=4.0)

        generator = CalibrationReportGenerator(config, scores, analysis)
        data = generator._collect_data()

        assert data["spec_id"] == "cal-test"
        assert data["repeats"] == 10
        assert data["hdi_pct"] == 94
        assert data["oracle_model"] == "test-model"
        assert data["oracle_adapter"] == "ollama"
        assert data["has_expected"] is True
        assert data["expected_score"] == 4.0
        assert data["verdict"] is not None
        assert data["verdict_class"] is not None

    def test_collect_data_without_expected(self):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        data = generator._collect_data()

        assert data["has_expected"] is False
        assert data["expected_score"] is None
        assert data["verdict"] is None

    def test_creates_parent_directories(self, tmp_path):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "nested" / "dir" / "report.html"
        generator.generate_html(output_path)

        assert output_path.exists()

    def test_calibration_report_is_self_contained_no_cdn_urls(self, tmp_path):
        """Calibration reports must work offline: no external CDN references."""
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        generator.generate_html(output_path)

        html = output_path.read_text()
        assert (
            "cdn.tailwindcss.com" not in html
        ), "Tailwind CDN found; report is not self-contained"
        assert (
            "cdn.jsdelivr.net" not in html
        ), "jsdelivr CDN found; report is not self-contained"
        assert (
            "fonts.googleapis.com" not in html
        ), "Google Fonts CDN found; report is not self-contained"
