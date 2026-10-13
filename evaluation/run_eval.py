
"""
Evaluation runner for SQL generation with multiple model support.

This module runs evaluation tests on generated SQL queries
across one or more LLM models.
"""

from typing import Dict, Any, List, Callable
import time
from datetime import datetime
import sys
from pathlib import Path

# Add the project root to the Python import path.
sys.path.insert(0, str(Path(__file__).parent.parent))

from sql.executor import get_database_connection
from sql.validator import validate_query
from evaluation.metrics import (
    calculate_all_metrics,
    calculate_sql_correctness,
    calculate_metrics_summary,
    calculate_composite_score,
)


def run_evaluation(
    db_path: str,
    dataset: List[Dict[str, Any]],
    llm_inference_fn: Callable,
    model_name: str = "default",
    calculate_metrics: bool = True,
    logger=None,
) -> List[Dict[str, Any]]:
    """
    Evaluate generated SQL queries against a database.

    Args:
        db_path: Path to the SQLite database.
        dataset: List of test cases.
        llm_inference_fn: Function that generates SQL from a question.
        model_name: Name of the model being evaluated.
        calculate_metrics: Whether to calculate similarity metrics.
        logger: Optional logger instance.

    Returns:
        A list of evaluation results with metrics.
    """
    results = []
    connection, cursor = get_database_connection(db_path)

    if logger:
        logger.info(f"Starting evaluation for model: {model_name}")

    try:
        for test_case in dataset:
            question = test_case.get("question", "")
            expected_query = test_case.get("expected_query", "")
            test_id = test_case.get("id")

            if logger:
                logger.debug(
                    f"Evaluating test {test_id}: {question[:60]}..."
                )

            start_time = time.time()

            try:
                # Generate SQL.
                generated_query = llm_inference_fn(question)
                generation_time = time.time() - start_time

                if logger:
                    logger.debug(
                        f"Test {test_id}: Query generated in "
                        f"{generation_time:.3f}s"
                    )

                # Validate the generated SQL.
                is_valid, validation_msg = validate_query(
                    generated_query
                )

                if logger:
                    if is_valid:
                        logger.debug(
                            f"Test {test_id}: Query validation PASSED"
                        )
                    else:
                        logger.warning(
                            f"Test {test_id}: Query validation FAILED - "
                            f"{validation_msg}"
                        )

                # Initialise execution status.
                execution_success = False
                error_msg = None
                execution_time = 0.0
                result_rows = 0

                # Execute only queries that pass validation.
                if is_valid:
                    exec_start = time.time()

                    try:
                        # Execute directly through SQLite.
                        # This ensures actual database errors are caught,
                        # including nonexistent tables and columns.
                        query_cursor = cursor.execute(generated_query)

                        # Fetch results before marking execution successful.
                        rows = query_cursor.fetchall()

                        result_rows = len(rows)
                        execution_success = True

                    except Exception as e:
                        error_msg = str(e)

                        if logger:
                            logger.warning(
                                f"Test {test_id}: Query execution FAILED - "
                                f"{error_msg}"
                            )

                    finally:
                        execution_time = time.time() - exec_start

                    if execution_success and logger:
                        logger.debug(
                            f"Test {test_id}: Query execution SUCCESS "
                            f"({execution_time:.3f}s, "
                            f"{result_rows} rows)"
                        )

                elif not is_valid:
                    error_msg = validation_msg

                # Calculate SQL similarity metrics.
                similarity_metrics = {}
                sql_correctness_score = 0.0

                if calculate_metrics and expected_query:
                    similarity_metrics = calculate_all_metrics(
                        generated_query,
                        expected_query,
                    )

                    sql_correctness_score = calculate_sql_correctness(
                        execution_success,
                        is_valid,
                    )

                    if logger:
                        logger.debug(
                            f"Test {test_id}: Metrics - Score: "
                            f"{calculate_composite_score(similarity_metrics):.4f}"
                        )

                result_dict = {
                    "test_id": test_id,
                    "model": model_name,
                    "question": question,
                    "generated_query": generated_query,
                    "expected_query": expected_query,
                    "is_valid": is_valid,
                    "validation_msg": validation_msg,
                    "execution_success": execution_success,
                    "error": error_msg,
                    "generation_time_sec": round(generation_time, 3),
                    "execution_time_sec": round(execution_time, 3),
                    "result_rows": result_rows,
                    "similarity_metrics": similarity_metrics,
                    "sql_correctness_score": round(
                        sql_correctness_score, 4
                    ),
                    "composite_score": (
                        round(
                            calculate_composite_score(
                                similarity_metrics
                            ),
                            4,
                        )
                        if similarity_metrics
                        else 0.0
                    ),
                }

                results.append(result_dict)

                if execution_success:
                    if logger:
                        logger.info(
                            f"Test {test_id} ({model_name}): PASSED - "
                            f"Score: {result_dict['composite_score']:.4f}"
                        )
                else:
                    if logger:
                        logger.warning(
                            f"Test {test_id} ({model_name}): FAILED - "
                            f"{error_msg or validation_msg}"
                        )

            except Exception as e:
                generation_time = time.time() - start_time

                if logger:
                    logger.error(
                        f"Test {test_id}: Evaluation FAILED - {str(e)}",
                        exc_info=True,
                    )

                results.append(
                    {
                        "test_id": test_id,
                        "model": model_name,
                        "question": question,
                        "expected_query": expected_query,
                        "error": f"Evaluation failed: {str(e)}",
                        "generation_time_sec": round(
                            generation_time, 3
                        ),
                    }
                )

    finally:
        connection.close()

    if logger:
        logger.info(f"Evaluation complete for model: {model_name}")

    return results


def run_multi_model_evaluation(
    db_path: str,
    dataset: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    get_inference_fn: Callable,
    logger=None,
) -> Dict[str, Any]:
    """
    Evaluate multiple LLM models and compare their results.

    Args:
        db_path: Path to the database.
        dataset: List of evaluation test cases.
        models: Model configurations with 'name' and other settings.
        get_inference_fn: Function that returns an inference function
                          for a model configuration.
        logger: Optional logger instance.

    Returns:
        Results, model summaries, comparison metrics, and timestamp.
    """
    all_results = {}
    model_summaries = {}

    if logger:
        logger.info("\nStarting Multi-Model Evaluation")
        logger.info(f"Number of models: {len(models)}")
        logger.info(f"Number of test cases: {len(dataset)}")

    for model_config in models:
        model_name = model_config.get("name", "unknown")

        if logger:
            logger.info(f"\n{'-' * 80}")
            logger.info(f"Evaluating Model: {model_name}")
            logger.info(f"{'-' * 80}")

        try:
            # Get the inference function for this model.
            inference_fn = get_inference_fn(model_config)

            # Evaluate the model.
            results = run_evaluation(
                db_path=db_path,
                dataset=dataset,
                llm_inference_fn=inference_fn,
                model_name=model_name,
                calculate_metrics=True,
                logger=logger,
            )

            # Calculate summary metrics.
            summary = calculate_metrics_summary(results)
            summary["model"] = model_name
            summary["timestamp"] = datetime.now().isoformat()

            all_results[model_name] = results
            model_summaries[model_name] = summary

            if logger:
                from core.logging import log_model_summary

                log_model_summary(logger, model_name, summary)
            else:
                print_evaluation_summary(summary)

        except Exception as e:
            error_msg = f"Error evaluating {model_name}: {str(e)}"

            if logger:
                logger.error(error_msg, exc_info=True)
            else:
                print(f"ERROR: {error_msg}")

            model_summaries[model_name] = {
                "model": model_name,
                "error": str(e),
            }

    comparison = _calculate_model_comparison(model_summaries)

    return {
        "all_results": all_results,
        "model_summaries": model_summaries,
        "comparison": comparison,
        "evaluation_timestamp": datetime.now().isoformat(),
    }


def _calculate_model_comparison(
    model_summaries: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Calculate comparative metrics across models.

    Args:
        model_summaries: Dictionary containing each model's summary.

    Returns:
        Comparison metrics dictionary.
    """
    comparison = {
        "best_models": {},
        "metrics_by_model": {},
    }

    best_composite = {"model": None, "score": -1}
    best_execution = {"model": None, "score": -1}
    best_validity = {"model": None, "score": -1}

    for model_name, summary in model_summaries.items():
        if "error" in summary:
            continue

        comparison["metrics_by_model"][model_name] = {
            "composite_score": summary.get("composite_score", 0),
            "executed_pct": summary.get("executed_pct", 0),
            "valid_queries_pct": summary.get(
                "valid_queries_pct", 0
            ),
            "similarity_metrics": summary.get(
                "similarity_metrics", {}
            ),
        }

        composite = summary.get("composite_score", 0)
        if composite > best_composite["score"]:
            best_composite = {
                "model": model_name,
                "score": composite,
            }

        exec_pct = summary.get("executed_pct", 0)
        if exec_pct > best_execution["score"]:
            best_execution = {
                "model": model_name,
                "score": exec_pct,
            }

        valid_pct = summary.get("valid_queries_pct", 0)
        if valid_pct > best_validity["score"]:
            best_validity = {
                "model": model_name,
                "score": valid_pct,
            }

    comparison["best_models"] = {
        "composite_score": best_composite,
        "execution_rate": best_execution,
        "validity_rate": best_validity,
    }

    return comparison


def print_evaluation_summary(summary: Dict[str, Any]) -> None:
    """Print detailed evaluation summary."""
    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"\n{'-' * 80}")
    print(f"Model: {summary.get('model', 'N/A')}")
    print(f"{'-' * 80}")

    total_tests = summary.get("total_tests", 0)

    print(f"Total tests: {total_tests}")
    print(
        f"Valid queries: {summary.get('valid_queries', 0)}/"
        f"{total_tests} "
        f"({summary.get('valid_queries_pct', 0):.1f}%)"
    )
    print(
        f"Executed successfully: {summary.get('executed', 0)}/"
        f"{total_tests} "
        f"({summary.get('executed_pct', 0):.1f}%)"
    )
    print(
        f"Errors: {summary.get('errors', 0)} "
        f"({summary.get('error_pct', 0):.1f}%)"
    )

    metrics = summary.get("similarity_metrics", {})

    if metrics:
        print("\nSimilarity Metrics:")
        print(f"  - Exact Match: {metrics.get('exact_match', 0):.4f}")
        print(f"  - Token Match: {metrics.get('token_match', 0):.4f}")
        print(f"  - BLEU Score: {metrics.get('bleu_score', 0):.4f}")
        print(f"  - F1 Score: {metrics.get('f1_score', 0):.4f}")
        print(
            "  - Semantic Similarity: "
            f"{metrics.get('semantic_similarity', 0):.4f}"
        )
        print(
            "  - Composite Score: "
            f"{summary.get('composite_score', 0):.4f}"
        )

    print(f"{'-' * 80}\n")


def print_model_comparison(comparison: Dict[str, Any]) -> None:
    """Print the model comparison report."""
    print(f"\n{'=' * 80}")
    print("Model Comparison Report")
    print(f"{'=' * 80}\n")

    best = comparison.get("best_models")

    if best:
        print("Best Performing Models:")

        for metric_name, label in [
            ("composite_score", "Composite Score"),
            ("execution_rate", "Execution Rate"),
            ("validity_rate", "Validity Rate"),
        ]:
            item = best.get(metric_name, {})
            print(
                f"  - {label}: {item.get('model')} "
                f"({item.get('score', 0):.4f}"
                f"{'%' if metric_name != 'composite_score' else ''})"
            )

    metrics_by_model = comparison.get("metrics_by_model", {})

    if metrics_by_model:
        print(f"\n{'-' * 80}")
        print("Detailed Metrics by Model:")
        print(f"{'-' * 80}")

        for model_name, metrics in metrics_by_model.items():
            print(f"\n{model_name}:")
            print(
                f"  Composite Score: "
                f"{metrics['composite_score']:.4f}"
            )
            print(
                f"  Execution Rate: {metrics['executed_pct']:.1f}%"
            )
            print(
                f"  Validity Rate: "
                f"{metrics['valid_queries_pct']:.1f}%"
            )

            if metrics["similarity_metrics"]:
                print("  Similarity Metrics:")

                for metric, value in metrics[
                    "similarity_metrics"
                ].items():
                    print(f"    - {metric}: {value:.4f}")

    print(f"\n{'=' * 80}\n")
