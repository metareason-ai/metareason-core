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


def make_analysis_result():
    return {
        "population_mean": 4.0,
        "population_median": 4.1,
        "population_std": 0.3,
        "hdi_lower": 3.5,
        "hdi_upper": 4.5,
        "hdi_prob": 0.94,
        "oracle_noise_mean": 0.3,
        "oracle_noise_hdi": (0.1, 0.5),
        "n_samples": 10,
    }


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
        assert "MetaReason Calibration Report" in html
        assert "cal-test" in html
        assert "Confidence Assessment" in html
        assert "posteriorChart" in html
        assert "histogramChart" in html
        assert "noiseChart" in html
        assert "test-model" in html

    def test_with_expected_score(self, tmp_path):
        config = make_calibrate_config(expected_score=4.0)
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        generator.generate_html(output_path)

        html = output_path.read_text()
        assert "Bias Analysis" in html
        assert "Expected Score" in html
        assert "Expected Within HDI" in html

    def test_without_expected_score(self, tmp_path):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        output_path = tmp_path / "report.html"
        generator.generate_html(output_path)

        html = output_path.read_text()
        assert "Bias Analysis" not in html

    def test_chart_data_structure(self):
        config = make_calibrate_config()
        scores = [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        chart_data = generator._generate_chart_data()

        # Posterior KDE data
        assert "posterior_x" in chart_data
        assert "posterior_y" in chart_data
        assert len(chart_data["posterior_x"]) == 80
        assert len(chart_data["posterior_y"]) == 80

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

        # Annotation values
        assert "hdi_lower" in chart_data
        assert "hdi_upper" in chart_data
        assert "population_mean" in chart_data
        assert "population_median" in chart_data
        assert "noise_mean" in chart_data
        assert "noise_hdi_lower" in chart_data
        assert "noise_hdi_upper" in chart_data

    def test_collect_data(self):
        config = make_calibrate_config(expected_score=4.0)
        scores = [4.0, 3.5, 4.5]
        analysis = make_analysis_result()

        generator = CalibrationReportGenerator(config, scores, analysis)
        data = generator._collect_data()

        assert data["spec_id"] == "cal-test"
        assert data["repeats"] == 10
        assert data["hdi_pct"] == 94
        assert data["oracle_model"] == "test-model"
        assert data["oracle_adapter"] == "ollama"
        assert data["has_expected"] is True
        assert data["expected_score"] == 4.0
        assert data["bias"] == 0.0  # mean 4.0 - expected 4.0
        assert data["within_hdi"] is True

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
