#!/usr/bin/env python3
"""
Example usage of the MetaReason Bayesian Analysis module.

This example demonstrates how to use the Bayesian statistical analysis
features to analyze oracle evaluation results and compute confidence intervals.
"""

from typing import List

# Import MetaReason components
from metareason.analysis import (
    BayesianAnalyzer,
    MultiOracleAnalyzer,
    analyze_oracle_comparison,
    quick_analysis,
)
from metareason.config.statistical import (
    InferenceConfig,
    OutputConfig,
    PriorConfig,
    StatisticalConfig,
)
from metareason.oracles.base import OracleResult


def create_sample_oracle_results() -> List[OracleResult]:
    """Create sample oracle results for demonstration."""
    # Simulating oracle evaluation results with varying scores
    # In practice, these would come from actual oracle evaluations
    return [
        OracleResult(score=0.85, metadata={"response_id": "1", "evaluation_time": 0.5}),
        OracleResult(score=0.92, metadata={"response_id": "2", "evaluation_time": 0.3}),
        OracleResult(score=0.78, metadata={"response_id": "3", "evaluation_time": 0.7}),
        OracleResult(score=0.95, metadata={"response_id": "4", "evaluation_time": 0.4}),
        OracleResult(
            score=0.45, metadata={"response_id": "5", "evaluation_time": 0.6}
        ),  # Lower score
        OracleResult(score=0.88, metadata={"response_id": "6", "evaluation_time": 0.5}),
        OracleResult(score=0.91, metadata={"response_id": "7", "evaluation_time": 0.3}),
        OracleResult(score=0.73, metadata={"response_id": "8", "evaluation_time": 0.8}),
        OracleResult(
            score=0.35, metadata={"response_id": "9", "evaluation_time": 0.9}
        ),  # Lower score
        OracleResult(
            score=0.89, metadata={"response_id": "10", "evaluation_time": 0.4}
        ),
    ]


def example_1_quick_analysis():
    """Example 1: Quick analysis with default settings."""
    print("=" * 60)
    print("EXAMPLE 1: Quick Analysis")
    print("=" * 60)

    # Create sample data
    oracle_results = create_sample_oracle_results()

    # Quick analysis with default settings (95% credible interval)
    result = quick_analysis(
        oracle_results=oracle_results,
        oracle_name="accuracy_oracle",
        threshold=0.7,  # Consider scores >= 0.7 as "success"
        credible_interval=0.95,
    )

    print(f"Oracle: {result.oracle_name}")
    print(
        f"Success Rate: {result.success_rate:.2%} ({result.n_successes}/{result.n_trials})"
    )
    print(f"Posterior Mean: {result.posterior_mean:.3f}")
    print(f"95% HDI: [{result.hdi_lower:.3f}, {result.hdi_upper:.3f}]")
    print(
        f"Convergence: {'✓' if result.converged else '✗'} (R-hat: {result.r_hat:.3f})"
    )
    print(
        f"Reliability: {'✓' if result.reliable else '✗'} (ESS: {result.effective_sample_size:.0f})"
    )
    print(f"Computation Time: {result.computation_time:.2f}s")
    print()


def example_2_custom_configuration():
    """Example 2: Custom Bayesian configuration."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 60)

    # Create custom statistical configuration
    config = StatisticalConfig(
        model="beta_binomial",
        prior=PriorConfig(
            alpha=2.0, beta=1.0  # Slightly informative prior favoring success
        ),
        inference=InferenceConfig(
            method="mcmc",
            samples=2000,  # More samples for higher precision
            chains=4,
            target_accept=0.9,  # Higher acceptance rate
        ),
        output=OutputConfig(
            credible_interval=0.89, hdi_method="shortest"  # 89% credible interval
        ),
    )

    # Create analyzer with custom config
    analyzer = BayesianAnalyzer(config)

    # Generate results with different threshold
    oracle_results = create_sample_oracle_results()

    result = analyzer.analyze_oracle_results(
        oracle_results=oracle_results,
        oracle_name="quality_oracle",
        threshold=0.8,  # Higher threshold for quality
    )

    print(f"Oracle: {result.oracle_name}")
    print(
        f"Configuration: {config.prior.alpha:.1f}-{config.prior.beta:.1f} "
        f"Beta prior, {config.inference.samples} samples"
    )
    print(
        f"Success Rate: {result.success_rate:.2%} ({result.n_successes}/{result.n_trials})"
    )
    print(f"Posterior Mean: {result.posterior_mean:.3f} ± {result.posterior_std:.3f}")
    print(f"89% HDI: [{result.hdi_lower:.3f}, {result.hdi_upper:.3f}]")
    print(f"Prior Effect: Alpha={config.prior.alpha}, Beta={config.prior.beta}")
    print()


def example_3_multi_oracle_analysis():
    """Example 3: Multi-oracle analysis with correlations."""
    print("=" * 60)
    print("EXAMPLE 3: Multi-Oracle Analysis")
    print("=" * 60)

    # Create different oracle result sets
    accuracy_results = create_sample_oracle_results()

    # Create clarity results (correlated but different)
    clarity_results = [
        OracleResult(score=0.82, metadata={}),
        OracleResult(score=0.89, metadata={}),
        OracleResult(score=0.75, metadata={}),
        OracleResult(score=0.91, metadata={}),
        OracleResult(score=0.48, metadata={}),  # Lower
        OracleResult(score=0.85, metadata={}),
        OracleResult(score=0.88, metadata={}),
        OracleResult(score=0.70, metadata={}),
        OracleResult(score=0.38, metadata={}),  # Lower
        OracleResult(score=0.86, metadata={}),
    ]

    # Create helpfulness results (different pattern)
    helpfulness_results = [
        OracleResult(score=0.90, metadata={}),
        OracleResult(score=0.85, metadata={}),
        OracleResult(score=0.92, metadata={}),
        OracleResult(score=0.88, metadata={}),
        OracleResult(score=0.75, metadata={}),
        OracleResult(score=0.93, metadata={}),
        OracleResult(score=0.87, metadata={}),
        OracleResult(score=0.89, metadata={}),
        OracleResult(score=0.84, metadata={}),
        OracleResult(score=0.91, metadata={}),
    ]

    # Configure multi-oracle analyzer
    config = StatisticalConfig(inference=InferenceConfig(samples=1500, chains=3))

    multi_analyzer = MultiOracleAnalyzer(config)

    # Analyze multiple oracles
    oracle_results_dict = {
        "accuracy": accuracy_results,
        "clarity": clarity_results,
        "helpfulness": helpfulness_results,
    }

    # Different thresholds per oracle
    thresholds = {"accuracy": 0.7, "clarity": 0.75, "helpfulness": 0.8}

    multi_result = multi_analyzer.analyze_multiple_oracles(
        oracle_results_dict=oracle_results_dict,
        thresholds=thresholds,
        compute_joint=True,
        compute_correlations=True,
    )

    print("Individual Oracle Results:")
    print("-" * 30)
    for oracle_name, result in multi_result.individual_results.items():
        threshold = thresholds[oracle_name]
        print(
            f"{oracle_name.capitalize()}: {result.success_rate:.2%} "
            f"(threshold={threshold}) HDI: [{result.hdi_lower:.3f}, {result.hdi_upper:.3f}]"
        )

    print()
    print("Joint Analysis:")
    print("-" * 15)
    if multi_result.joint_posterior_mean:
        print(f"Joint Posterior Mean: {multi_result.joint_posterior_mean:.3f}")
        print(
            f"Joint HDI: [{multi_result.joint_hdi_lower:.3f}, {multi_result.joint_hdi_upper:.3f}]"
        )

    print(f"Combined Confidence: {multi_result.combined_confidence:.3f}")
    print(f"All Converged: {'✓' if multi_result.all_converged else '✗'}")
    print(f"All Reliable: {'✓' if multi_result.all_reliable else '✗'}")

    if multi_result.oracle_correlations:
        print()
        print("Oracle Correlations:")
        print("-" * 20)
        for (oracle1, oracle2), correlation in multi_result.oracle_correlations.items():
            print(f"{oracle1} ↔ {oracle2}: {correlation:.3f}")

    print()


def example_4_oracle_comparison():
    """Example 4: Comparing two oracles directly."""
    print("=" * 60)
    print("EXAMPLE 4: Oracle Comparison")
    print("=" * 60)

    # Create two sets of oracle results for comparison
    oracle_a_results = create_sample_oracle_results()

    # Oracle B performs slightly worse
    oracle_b_results = [
        OracleResult(score=0.75, metadata={}),
        OracleResult(score=0.82, metadata={}),
        OracleResult(score=0.68, metadata={}),
        OracleResult(score=0.85, metadata={}),
        OracleResult(score=0.35, metadata={}),  # Lower
        OracleResult(score=0.78, metadata={}),
        OracleResult(score=0.81, metadata={}),
        OracleResult(score=0.63, metadata={}),
        OracleResult(score=0.25, metadata={}),  # Lower
        OracleResult(score=0.79, metadata={}),
    ]

    # Compare the oracles
    result_a, result_b, prob_a_better = analyze_oracle_comparison(
        oracle_a_results=oracle_a_results,
        oracle_b_results=oracle_b_results,
        oracle_a_name="GPT-4",
        oracle_b_name="GPT-3.5",
        threshold=0.7,
    )

    print("Oracle Comparison Results:")
    print("-" * 25)
    print(f"{result_a.oracle_name}:")
    print(f"  Success Rate: {result_a.success_rate:.2%}")
    print(f"  Posterior Mean: {result_a.posterior_mean:.3f}")
    print(f"  HDI: [{result_a.hdi_lower:.3f}, {result_a.hdi_upper:.3f}]")
    print()
    print(f"{result_b.oracle_name}:")
    print(f"  Success Rate: {result_b.success_rate:.2%}")
    print(f"  Posterior Mean: {result_b.posterior_mean:.3f}")
    print(f"  HDI: [{result_b.hdi_lower:.3f}, {result_b.hdi_upper:.3f}]")
    print()
    print(
        f"Probability {result_a.oracle_name} > {result_b.oracle_name}: {prob_a_better:.1%}"
    )

    # Practical interpretation
    if result_a.posterior_mean > result_b.posterior_mean:
        diff = result_a.posterior_mean - result_b.posterior_mean
        print(
            f"{result_a.oracle_name} appears to perform {diff:.3f} points better on average."
        )

    print()


def example_5_confidence_interpretation():
    """Example 5: Understanding confidence intervals and reliability."""
    print("=" * 60)
    print("EXAMPLE 5: Confidence Interpretation")
    print("=" * 60)

    # Create scenarios with different sample sizes and success rates
    scenarios = [
        (
            "High confidence",
            [OracleResult(score=0.9, metadata={}) for _ in range(50)]
            + [OracleResult(score=0.3, metadata={}) for _ in range(10)],
        ),
        (
            "Moderate confidence",
            [OracleResult(score=0.8, metadata={}) for _ in range(15)]
            + [OracleResult(score=0.4, metadata={}) for _ in range(10)],
        ),
        (
            "Low confidence",
            [OracleResult(score=0.7, metadata={}) for _ in range(6)]
            + [OracleResult(score=0.3, metadata={}) for _ in range(4)],
        ),
    ]

    for scenario_name, results in scenarios:
        result = quick_analysis(results, oracle_name=scenario_name.replace(" ", "_"))

        print(f"Scenario: {scenario_name}")
        print(f"  Sample Size: {result.n_trials}")
        print(f"  Success Rate: {result.success_rate:.2%}")
        print(f"  Posterior Mean: {result.posterior_mean:.3f}")
        print(f"  95% HDI Width: {result.hdi_upper - result.hdi_lower:.3f}")
        print(f"  Reliable: {'✓' if result.reliable else '✗'}")

        # Interpretation
        hdi_width = result.hdi_upper - result.hdi_lower
        if hdi_width < 0.1:
            confidence_level = "High"
        elif hdi_width < 0.2:
            confidence_level = "Moderate"
        else:
            confidence_level = "Low"

        print(f"  Confidence Level: {confidence_level} (HDI width: {hdi_width:.3f})")
        print()


def main():
    """Run all examples."""
    print("MetaReason Bayesian Analysis Examples")
    print("=" * 60)
    print()

    try:
        # Run examples sequentially
        example_1_quick_analysis()
        example_2_custom_configuration()
        example_3_multi_oracle_analysis()
        example_4_oracle_comparison()
        example_5_confidence_interpretation()

        print("=" * 60)
        print("All examples completed successfully!")
        print()
        print("Key Takeaways:")
        print("- Bayesian analysis provides uncertainty quantification")
        print("- HDI intervals show credible ranges for true performance")
        print("- Multi-oracle analysis can reveal correlations and joint confidence")
        print("- Larger sample sizes lead to more reliable estimates")
        print("- Convergence diagnostics ensure statistical validity")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: Examples require PyMC and ArviZ for actual MCMC sampling.")
        print("Mock the PyMC calls for demonstration without dependencies.")


if __name__ == "__main__":
    main()
