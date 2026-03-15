from metareason.config.models import (
    AdapterConfig,
    BayesianAnalysisConfig,
    CalibrateMultiConfig,
    OracleConfig,
)
from metareason.reporting.multi_judge_report import MultiJudgeReportGenerator


def make_multi_config(**overrides):
    defaults = dict(
        spec_id="multi-cal-test",
        type="calibrate_multi",
        prompt="What is 2+2?",
        response="The answer is 4.",
        repeats=10,
        oracles={
            "judge_a": OracleConfig(
                type="llm_judge",
                model="model-a",
                adapter=AdapterConfig(name="ollama"),
                rubric="Rate 1-5",
            ),
            "judge_b": OracleConfig(
                type="llm_judge",
                model="model-b",
                adapter=AdapterConfig(name="openai"),
                rubric="Rate 1-5",
            ),
        },
        analysis=BayesianAnalysisConfig(hdi_probability=0.94),
    )
    defaults.update(overrides)
    return CalibrateMultiConfig(**defaults)


def make_multi_judge_result(expected_score=None):
    result = {
        "hdi_prob": 0.94,
        "n_judges": 2,
        "n_total_evaluations": 20,
        "judges": {
            "judge_a": {
                "bias_mean": 0.15,
                "bias_hdi": (-0.05, 0.35),
                "noise_mean": 0.40,
                "noise_hdi": (0.20, 0.60),
                "consistency_weight": 0.55,
                "raw_score_mean": 4.20,
                "raw_score_std": 0.45,
                "n_evaluations": 10,
            },
            "judge_b": {
                "bias_mean": -0.10,
                "bias_hdi": (-0.30, 0.10),
                "noise_mean": 0.50,
                "noise_hdi": (0.25, 0.75),
                "consistency_weight": 0.45,
                "raw_score_mean": 3.90,
                "raw_score_std": 0.55,
                "n_evaluations": 10,
            },
        },
    }
    if expected_score is not None:
        result["expected_score"] = expected_score
    else:
        result["true_quality_mean"] = 4.05
        result["true_quality_median"] = 4.06
        result["true_quality_std"] = 0.25
        result["hdi_lower"] = 3.55
        result["hdi_upper"] = 4.55
        result["bias_corrected_weighted_score"] = 4.02
    return result


class TestMultiJudgeReportGeneratorInit:
    def test_init_stores_attributes(self):
        config = make_multi_config()
        scores = {"judge_a": [4.0, 3.5], "judge_b": [3.0, 4.5]}
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)

        assert gen.config is config
        assert gen.scores_by_oracle is scores
        assert gen.multi_judge is result


class TestGenerateHtml:
    def test_generates_html_file(self, tmp_path):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        returned = gen.generate_html(output_path)

        assert returned == output_path
        assert output_path.exists()

        html = output_path.read_text()
        assert "MetaReason Multi-Judge Report" in html
        assert "multi-cal-test" in html

    def test_html_contains_chart_canvases(self, tmp_path):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "histogramChart" in html
        assert "biasPosteriorChart" in html
        assert "noisePosteriorChart" in html

    def test_html_contains_judge_assessment(self, tmp_path):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "Judge Comparison" in html
        assert "Consistency Weight" in html

    def test_creates_parent_directories(self, tmp_path):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5],
            "judge_b": [3.0, 3.5, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "nested" / "dir" / "report.html"
        gen.generate_html(output_path)

        assert output_path.exists()


class TestGenerateChartData:
    def test_chart_data_has_expected_keys(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert "judge_names" in chart_data
        assert "judge_histograms" in chart_data
        assert "bias_posteriors" in chart_data
        assert "noise_posteriors" in chart_data
        assert "histogram_labels" in chart_data
        assert "has_expected" in chart_data

    def test_judge_names_match_oracles(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert set(chart_data["judge_names"]) == {"judge_a", "judge_b"}

    def test_judge_histograms_per_judge(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert "judge_a" in chart_data["judge_histograms"]
        assert "judge_b" in chart_data["judge_histograms"]
        # 5 bins for scores 1-5
        assert len(chart_data["judge_histograms"]["judge_a"]) == 5
        assert len(chart_data["judge_histograms"]["judge_b"]) == 5

    def test_histogram_labels(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert chart_data["histogram_labels"] == ["1", "2", "3", "4", "5"]

    def test_bias_posteriors_per_judge(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert "judge_a" in chart_data["bias_posteriors"]
        assert "judge_b" in chart_data["bias_posteriors"]
        for name in ("judge_a", "judge_b"):
            bp = chart_data["bias_posteriors"][name]
            assert "x" in bp
            assert "y" in bp
            assert "mean" in bp
            assert "hdi_lower" in bp
            assert "hdi_upper" in bp

    def test_noise_posteriors_per_judge(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert "judge_a" in chart_data["noise_posteriors"]
        assert "judge_b" in chart_data["noise_posteriors"]
        for name in ("judge_a", "judge_b"):
            np_data = chart_data["noise_posteriors"][name]
            assert "x" in np_data
            assert "y" in np_data
            assert "mean" in np_data


class TestSelfContainedHtml:
    def test_no_cdn_references(self, tmp_path):
        """Multi-judge reports must work offline: no external CDN references."""
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

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


class TestWithExpectedScore:
    def test_with_expected_score_shows_accuracy(self, tmp_path):
        config = make_multi_config(expected_score=4.0)
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result(expected_score=4.0)

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "Accuracy" in html

    def test_collect_data_with_expected_score(self):
        config = make_multi_config(expected_score=4.0)
        scores = {
            "judge_a": [4.0, 3.5],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result(expected_score=4.0)

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["has_expected"] is True
        assert data["expected_score"] == 4.0
        assert "judge_a" in data["judge_verdicts"]
        assert "judge_b" in data["judge_verdicts"]


class TestWithoutExpectedScore:
    def test_without_expected_score_shows_estimated_quality(self, tmp_path):
        config = make_multi_config()  # no expected_score
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "Estimated Quality" in html

    def test_collect_data_without_expected_score(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.5],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["has_expected"] is False
        assert data["expected_score"] is None


class TestCollectData:
    def test_collect_data_basic_fields(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["spec_id"] == "multi-cal-test"
        assert data["repeats"] == 10
        assert data["n_judges"] == 2
        assert data["hdi_pct"] == 94
        assert "timestamp" in data
        assert data["multi_judge"] is result
        assert data["prompt"] == "What is 2+2?"
        assert data["response"] == "The answer is 4."

    def test_collect_data_recommendations_for_noisy_judge(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()
        # Make judge_b noisy (noise > 1.0)
        result["judges"]["judge_b"]["noise_mean"] = 1.5

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert len(data["recommendations"]) == 1
        assert "judge_b" in data["recommendations"][0]
        assert "noise=1.50" in data["recommendations"][0]

    def test_collect_data_no_recommendations_for_quiet_judges(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["recommendations"] == []

    def test_collect_data_uses_custom_hdi_probability(self):
        config = make_multi_config(
            analysis=BayesianAnalysisConfig(hdi_probability=0.89)
        )
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["hdi_pct"] == 89

    def test_collect_data_default_hdi_without_analysis(self):
        config = make_multi_config(analysis=None)
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert data["hdi_pct"] == 94

    def test_collect_data_judge_verdicts(self):
        config = make_multi_config()
        scores = {
            "judge_a": [4.0, 3.0],
            "judge_b": [3.0, 4.0],
        }
        result = make_multi_judge_result()

        gen = MultiJudgeReportGenerator(config, scores, result)
        data = gen._collect_data()

        assert "judge_verdicts" in data
        assert "judge_a" in data["judge_verdicts"]
        assert "judge_b" in data["judge_verdicts"]
        for v in data["judge_verdicts"].values():
            assert "verdict" in v
            assert "verdict_class" in v


class TestThreeJudges:
    """Verify the generator works with more than two judges."""

    def _make_three_judge_fixtures(self):
        config = make_multi_config(
            oracles={
                "judge_a": OracleConfig(
                    type="llm_judge",
                    model="model-a",
                    adapter=AdapterConfig(name="ollama"),
                    rubric="Rate 1-5",
                ),
                "judge_b": OracleConfig(
                    type="llm_judge",
                    model="model-b",
                    adapter=AdapterConfig(name="openai"),
                    rubric="Rate 1-5",
                ),
                "judge_c": OracleConfig(
                    type="llm_judge",
                    model="model-c",
                    adapter=AdapterConfig(name="anthropic"),
                    rubric="Rate 1-5",
                ),
            },
        )
        scores = {
            "judge_a": [4.0, 3.5, 4.5, 4.0, 3.0, 5.0, 4.0, 4.5, 3.5, 4.0],
            "judge_b": [3.0, 3.5, 4.0, 3.5, 3.0, 4.5, 4.0, 3.5, 3.0, 4.0],
            "judge_c": [4.5, 4.0, 5.0, 4.5, 4.0, 5.0, 4.5, 5.0, 4.0, 4.5],
        }
        result = make_multi_judge_result()
        result["n_judges"] = 3
        result["judges"]["judge_c"] = {
            "bias_mean": 0.50,
            "bias_hdi": (0.25, 0.75),
            "noise_mean": 0.30,
            "noise_hdi": (0.15, 0.45),
            "consistency_weight": 0.35,
            "raw_score_mean": 4.50,
            "raw_score_std": 0.40,
            "n_evaluations": 10,
        }
        return config, scores, result

    def test_three_judge_generates_html(self, tmp_path):
        config, scores, result = self._make_three_judge_fixtures()

        gen = MultiJudgeReportGenerator(config, scores, result)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        assert output_path.exists()
        html = output_path.read_text()
        assert "judge_a" in html
        assert "judge_b" in html
        assert "judge_c" in html

    def test_three_judge_histograms(self):
        config, scores, result = self._make_three_judge_fixtures()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert len(chart_data["judge_names"]) == 3
        assert "judge_c" in chart_data["judge_histograms"]
        assert len(chart_data["judge_histograms"]["judge_c"]) == 5

    def test_three_judge_posteriors(self):
        config, scores, result = self._make_three_judge_fixtures()

        gen = MultiJudgeReportGenerator(config, scores, result)
        chart_data = gen._generate_chart_data()

        assert len(chart_data["bias_posteriors"]) == 3
        assert len(chart_data["noise_posteriors"]) == 3
        assert "judge_c" in chart_data["bias_posteriors"]
        assert "judge_c" in chart_data["noise_posteriors"]
