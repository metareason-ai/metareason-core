import matplotlib

matplotlib.use("Agg")

from metareason.config.models import (  # noqa: E402
    AdapterConfig,
    AxisConfig,
    OracleConfig,
    PipelineConfig,
    SamplingConfig,
    SpecConfig,
)
from metareason.oracles.oracle_base import EvaluationResult  # noqa: E402
from metareason.pipeline.runner import SampleResult  # noqa: E402
from metareason.reporting.report_generator import ReportGenerator  # noqa: E402


def _make_fixtures():
    results = [
        SampleResult(
            sample_params={"tone": "formal", "complexity": 5.0},
            original_prompt="test prompt",
            final_response="test response",
            evaluations={
                "test_oracle": EvaluationResult(score=4.0, explanation="good")
            },
        )
        for _ in range(3)
    ]

    spec = SpecConfig(
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
            AxisConfig(
                name="tone",
                type="categorical",
                values=["formal", "casual"],
            ),
            AxisConfig(
                name="complexity",
                type="continuous",
                distribution="uniform",
                params={"low": 1.0, "high": 10.0},
            ),
        ],
    )

    analysis_results = {
        "test_oracle": {
            "population_mean": 4.0,
            "population_median": 4.1,
            "hdi_lower": 3.5,
            "hdi_upper": 4.5,
            "hdi_prob": 0.94,
            "oracle_noise_mean": 0.3,
            "oracle_noise_hdi": (0.1, 0.5),
            "n_samples": 3,
        }
    }

    return results, spec, analysis_results


class TestReportGeneratorInit:
    def test_report_generator_init(self):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        assert gen.results is results
        assert gen.spec_config is spec
        assert gen.analysis_results is analysis_results


class TestCollectData:
    def test_collect_data(self):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        data = gen._collect_data()
        assert data["spec_id"] == "test_spec"
        assert data["n_variants"] == 3
        assert data["n_oracles"] == 1
        assert data["hdi_pct"] == 94
        assert "test_oracle" in data["oracle_analyses"]
        assert "timestamp" in data


class TestGenerateHtml:
    def test_generate_html_creates_file(self, tmp_path):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        output_path = tmp_path / "report.html"
        result_path = gen.generate_html(output_path)

        assert result_path.exists()
        html = result_path.read_text()
        assert "MetaReason" in html
        assert "test_spec" in html
        assert "test_oracle" in html

    def test_generate_html_contains_images(self, tmp_path):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "data:image/png;base64," in html

    def test_generate_html_contains_data_table(self, tmp_path):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        output_path = tmp_path / "report.html"
        gen.generate_html(output_path)

        html = output_path.read_text()
        assert "<table>" in html
        assert "tone=formal" in html
        assert "complexity=5.00" in html

    def test_generate_html_creates_parent_dirs(self, tmp_path):
        results, spec, analysis_results = _make_fixtures()
        gen = ReportGenerator(results, spec, analysis_results)
        output_path = tmp_path / "subdir" / "nested" / "report.html"
        result_path = gen.generate_html(output_path)
        assert result_path.exists()
