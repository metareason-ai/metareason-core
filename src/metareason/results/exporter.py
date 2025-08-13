"""Results exporter for various output formats."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..pipeline.models import PipelineResult

logger = logging.getLogger(__name__)


class ResultExporter:
    """Export pipeline results to various formats."""

    def __init__(self):
        """Initialize result exporter."""
        pass

    def export_json(self, result: PipelineResult, output_path: Path) -> None:
        """Export results to JSON format.

        Args:
            result: Pipeline result to export
            output_path: Path for output file
        """
        logger.info(f"Exporting results to JSON: {output_path}")

        try:
            # Convert result to JSON-serializable format
            data = self._result_to_dict(result)

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"JSON export completed: {output_path}")

        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            raise

    def export_csv(self, result: PipelineResult, output_path: Path) -> None:
        """Export results to CSV format.

        Args:
            result: Pipeline result to export
            output_path: Path for output file
        """
        logger.info(f"Exporting results to CSV: {output_path}")

        try:
            # Create DataFrame from results
            df = self._result_to_dataframe(result)

            # Write to CSV
            df.to_csv(output_path, index=False)

            logger.info(f"CSV export completed: {output_path} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise

    def export_parquet(self, result: PipelineResult, output_path: Path) -> None:
        """Export results to Parquet format.

        Args:
            result: Pipeline result to export
            output_path: Path for output file
        """
        logger.info(f"Exporting results to Parquet: {output_path}")

        try:
            # Create DataFrame from results
            df = self._result_to_dataframe(result)

            # Write to Parquet
            df.to_parquet(output_path, index=False)

            logger.info(f"Parquet export completed: {output_path} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Parquet export failed: {e}")
            raise

    def export_html_report(self, result: PipelineResult, output_path: Path) -> None:
        """Export results as HTML report.

        Args:
            result: Pipeline result to export
            output_path: Path for output file
        """
        logger.info(f"Exporting results to HTML report: {output_path}")

        try:
            # Generate HTML content
            html_content = self._generate_html_report(result)

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            logger.info(f"HTML report export completed: {output_path}")

        except Exception as e:
            logger.error(f"HTML report export failed: {e}")
            raise

    def export_summary(self, result: PipelineResult, output_path: Path) -> None:
        """Export summary statistics to JSON.

        Args:
            result: Pipeline result to export
            output_path: Path for output file
        """
        logger.info(f"Exporting summary to: {output_path}")

        try:
            summary = result.get_summary()

            # Add additional summary statistics
            if result.bayesian_results:
                summary["bayesian_summary"] = {
                    "all_converged": result.bayesian_results.all_converged,
                    "all_reliable": result.bayesian_results.all_reliable,
                    "oracle_count": len(result.bayesian_results.individual_results),
                    "joint_analysis": result.bayesian_results.joint_posterior_mean
                    is not None,
                    "combined_confidence": result.bayesian_results.combined_confidence,
                }

            # Write summary
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Summary export completed: {output_path}")

        except Exception as e:
            logger.error(f"Summary export failed: {e}")
            raise

    def _result_to_dict(self, result: PipelineResult) -> Dict[str, Any]:
        """Convert pipeline result to dictionary format.

        Args:
            result: Pipeline result

        Returns:
            Dictionary representation
        """
        data = {
            "execution_summary": result.get_summary(),
            "config": result.config.model_dump(),
            "samples": {
                "metadata": result.samples.metadata,
                "shape": (
                    list(result.samples.samples.shape)
                    if hasattr(result.samples.samples, "shape")
                    else None
                ),
            },
            "step_results": [],
            "oracle_results": {},
        }

        # Add step results
        for step in result.step_results:
            step_data = {
                "step_index": step.step_index,
                "step_name": step.step_name,
                "success_rate": step.success_rate,
                "timing": step.timing,
                "metadata": step.metadata,
                "prompt_count": len(step.prompts),
                "response_count": len(step.responses),
                "error_count": len(step.errors),
                "errors": [str(e) for e in step.errors],
            }
            data["step_results"].append(step_data)

        # Add oracle results
        for oracle_name, oracle_results in result.oracle_results.items():
            data["oracle_results"][oracle_name] = [
                {
                    "score": oracle_result.score,
                    "metadata": oracle_result.metadata,
                }
                for oracle_result in oracle_results
            ]

        # Add Bayesian results
        if result.bayesian_results:
            data["bayesian_results"] = {
                "individual_results": {},
                "joint_posterior_mean": result.bayesian_results.joint_posterior_mean,
                "joint_hdi_lower": result.bayesian_results.joint_hdi_lower,
                "joint_hdi_upper": result.bayesian_results.joint_hdi_upper,
                "combined_confidence": result.bayesian_results.combined_confidence,
                "all_converged": result.bayesian_results.all_converged,
                "all_reliable": result.bayesian_results.all_reliable,
            }

            # Add individual oracle Bayesian results
            for (
                oracle_name,
                bayesian_result,
            ) in result.bayesian_results.individual_results.items():
                data["bayesian_results"]["individual_results"][oracle_name] = {
                    "posterior_mean": bayesian_result.posterior_mean,
                    "posterior_std": bayesian_result.posterior_std,
                    "hdi_lower": bayesian_result.hdi_lower,
                    "hdi_upper": bayesian_result.hdi_upper,
                    "credible_interval": bayesian_result.credible_interval,
                    "n_successes": bayesian_result.n_successes,
                    "n_trials": bayesian_result.n_trials,
                    "success_rate": bayesian_result.success_rate,
                    "converged": bayesian_result.converged,
                    "reliable": bayesian_result.reliable,
                    "r_hat": bayesian_result.r_hat,
                    "effective_sample_size": bayesian_result.effective_sample_size,
                    "n_divergences": bayesian_result.n_divergences,
                    "computation_time": bayesian_result.computation_time,
                }

        return data

    def _result_to_dataframe(self, result: PipelineResult) -> pd.DataFrame:
        """Convert pipeline result to pandas DataFrame.

        Args:
            result: Pipeline result

        Returns:
            DataFrame with one row per sample/response
        """
        rows = []

        # Get final step results (assuming that's what we want to export)
        final_step = result.step_results[-1] if result.step_results else None

        if final_step:
            for i in range(len(final_step.prompts)):
                row = {
                    "execution_id": result.execution_id,
                    "spec_id": result.config.spec_id,
                    "sample_index": i,
                    "prompt": (
                        final_step.prompts[i] if i < len(final_step.prompts) else ""
                    ),
                    "response": (
                        final_step.responses[i].content
                        if i < len(final_step.responses)
                        else ""
                    ),
                    "has_response": i < len(final_step.responses),
                }

                # Add oracle scores
                for oracle_name, oracle_results in result.oracle_results.items():
                    if i < len(oracle_results):
                        row[f"oracle_{oracle_name}_score"] = oracle_results[i].score
                    else:
                        row[f"oracle_{oracle_name}_score"] = None

                # Add sample variables (if available)
                if hasattr(result.samples, "samples") and i < len(
                    result.samples.samples
                ):
                    axis_names = result.samples.metadata.get("axis_names", [])
                    sample_row = result.samples.samples[i]
                    for j, axis_name in enumerate(axis_names):
                        if j < len(sample_row):
                            row[f"var_{axis_name}"] = sample_row[j]

                rows.append(row)

        return pd.DataFrame(rows)

    def _generate_html_report(self, result: PipelineResult) -> str:
        """Generate HTML report content.

        Args:
            result: Pipeline result

        Returns:
            HTML content string
        """
        summary = result.get_summary()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MetaReason Evaluation Report - {result.config.spec_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MetaReason Evaluation Report</h1>
                <h2>{result.config.spec_id}</h2>
                <p>Execution ID: {result.execution_id}</p>
                <p>Generated: {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h3>Execution Summary</h3>
                <div class="metric">Status: <span class="{
                    'success' if summary['status'] == 'success' else 'error'
                }">{summary['status']}</span></div>
                <div class="metric">Overall Success Rate: {summary['overall_success_rate']:.1%}</div>
                <div class="metric">Total Samples: {summary['total_samples']:,}</div>
                <div class="metric">Total Responses: {summary['total_responses']:,}</div>
                <div class="metric">Execution Time: {summary['execution_time']:.2f} seconds</div>
            </div>

            <div class="section">
                <h3>Pipeline Steps</h3>
                <table>
                    <tr><th>Step</th><th>Adapter/Model</th><th>Success Rate</th><th>Responses</th></tr>
        """

        for i, step in enumerate(result.step_results):
            html += f"""
                    <tr>
                        <td>{i+1}</td>
                        <td>{step.step_name}</td>
                        <td>{step.success_rate:.1%}</td>
                        <td>{len(step.responses)}</td>
                    </tr>
            """

        html += """
                </table>
            </div>
        """

        # Add oracle results
        if result.oracle_results:
            html += """
            <div class="section">
                <h3>Oracle Evaluation Results</h3>
                <table>
                    <tr><th>Oracle</th><th>Evaluations</th><th>Avg Score</th><th>Std Dev</th></tr>
            """

            for oracle_name, oracle_results in result.oracle_results.items():
                if oracle_results:
                    scores = [r.score for r in oracle_results]
                    avg_score = sum(scores) / len(scores)
                    std_dev = (
                        sum((s - avg_score) ** 2 for s in scores) / len(scores)
                    ) ** 0.5

                    html += f"""
                    <tr>
                        <td>{oracle_name}</td>
                        <td>{len(oracle_results)}</td>
                        <td>{avg_score:.3f}</td>
                        <td>{std_dev:.3f}</td>
                    </tr>
                    """

            html += """
                </table>
            </div>
            """

        # Add Bayesian results
        if result.bayesian_results:
            html += """
            <div class="section">
                <h3>Bayesian Analysis Results</h3>
                <table>
                    <tr><th>Oracle</th><th>Posterior Mean</th><th>95% HDI</th><th>Converged</th><th>Reliable</th></tr>
            """

            for (
                oracle_name,
                bayesian_result,
            ) in result.bayesian_results.individual_results.items():
                html += f"""
                <tr>
                    <td>{oracle_name}</td>
                    <td>{bayesian_result.posterior_mean:.3f}</td>
                    <td>[{bayesian_result.hdi_lower:.3f}, {bayesian_result.hdi_upper:.3f}]</td>
                    <td class="{
                        'success' if bayesian_result.converged else 'error'
                    }">{'Yes' if bayesian_result.converged else 'No'}</td>
                    <td class="{
                        'success' if bayesian_result.reliable else 'warning'
                    }">{'Yes' if bayesian_result.reliable else 'No'}</td>
                </tr>
                """

            html += """
                </table>
            </div>
            """

        html += """
        </body>
        </html>
        """

        return html
