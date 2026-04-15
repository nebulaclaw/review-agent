"""pipeline.image_dual_check.report_locale persisted to config.yaml."""

from __future__ import annotations

import yaml

from reviewagent.config import apply_pipeline_report_locale_to_yaml_file


def test_apply_report_locale_merges_into_image_dual_check(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "pipeline:\n  image_dual_check:\n    report_locale: zh\n",
        encoding="utf-8",
    )
    apply_pipeline_report_locale_to_yaml_file(str(p), "en")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["pipeline"]["image_dual_check"]["report_locale"] == "en"


def test_apply_report_locale_preserves_other_pipeline_keys(tmp_path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "llm:\n  model: x\npipeline:\n  wordlist:\n    enabled: true\n"
        "  image_dual_check:\n    report_locale: zh\n    image_dual_consistency_enabled: false\n",
        encoding="utf-8",
    )
    apply_pipeline_report_locale_to_yaml_file(str(p), "zh")
    apply_pipeline_report_locale_to_yaml_file(str(p), "en")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert data["llm"]["model"] == "x"
    assert data["pipeline"]["wordlist"]["enabled"] is True
    assert data["pipeline"]["image_dual_check"]["image_dual_consistency_enabled"] is False
    assert data["pipeline"]["image_dual_check"]["report_locale"] == "en"
