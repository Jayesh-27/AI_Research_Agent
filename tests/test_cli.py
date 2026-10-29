"""Tests for CLI flags and execution options."""

from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner

from agent.cli import app
from agent.schemas import FinalAnswer, RunStats

runner = CliRunner()


def test_cli_save_failed_run_redirects_to_runs_failed(tmp_path):
    target_dir = tmp_path / "custom_samples"
    failed_answer = FinalAnswer(
        status="failed",
        user_question="Test question",
        short_answer="No evidence found",
        key_findings=[],
        sources=[],
        confidence={"level": "low", "score": 0.0, "rationale": ["Failed"]},  # type: ignore[arg-type]
        assumptions=[],
        limitations=["Failed run"],
        next_steps=[],
        stats=RunStats(run_id="test_run_123", latency_s=1.0),
    )

    with patch("agent.cli.run_research", new_callable=AsyncMock) as mock_research:
        mock_research.return_value = failed_answer
        res = runner.invoke(app, ["ask", "Test question", "--save", str(target_dir)])
        assert res.exit_code == 0

        # Custom directory should not have files
        assert not target_dir.exists() or len(list(target_dir.glob("*.json"))) == 0

        # Runs/failed directory should have the file
        from pathlib import Path

        failed_dir = Path("runs/failed")
        assert failed_dir.exists()
        saved_files = list(failed_dir.glob("*test_run_123*.json"))
        assert len(saved_files) == 1

        # Clean up created test file
        for f in saved_files:
            f.unlink()
        for f in failed_dir.glob("*test_run_123*.md"):
            f.unlink()


def test_cli_save_failed_run_with_force_save(tmp_path):
    target_dir = tmp_path / "forced_samples"
    failed_answer = FinalAnswer(
        status="failed",
        user_question="Forced test question",
        short_answer="No evidence found",
        key_findings=[],
        sources=[],
        confidence={"level": "low", "score": 0.0, "rationale": ["Failed"]},  # type: ignore[arg-type]
        assumptions=[],
        limitations=["Failed run"],
        next_steps=[],
        stats=RunStats(run_id="forced_run_456", latency_s=1.0),
    )

    with patch("agent.cli.run_research", new_callable=AsyncMock) as mock_research:
        mock_research.return_value = failed_answer
        res = runner.invoke(
            app, ["ask", "Forced test question", "--save", str(target_dir), "--force-save"]
        )
        assert res.exit_code == 0

        # Target directory should have the file because force-save was enabled
        assert target_dir.exists()
        saved_files = list(target_dir.glob("*forced_run_456*.json"))
        assert len(saved_files) == 1

