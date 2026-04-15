from reviewagent.content_violation import (
    ContentViolationType,
    VIOLATION_TYPE_LABELS,
    format_violation_row_for_report,
    label_for_violation_position,
    label_for_violation_type,
    violation_category_labels,
)


def test_enum_value():
    assert ContentViolationType.PORN.value == "porn"
    assert VIOLATION_TYPE_LABELS["porn"] == "色情内容"


def test_label_unknown_returns_raw():
    assert label_for_violation_type("custom_xyz") == "custom_xyz"


def test_label_dual_branch_position_zh_en():
    assert label_for_violation_position("ocr_text") == "读字分支"
    assert label_for_violation_position("vision", locale="en") == "Vision branch"


def test_format_violation_row_type_and_position():
    row = format_violation_row_for_report(
        {
            "type": "porn",
            "content": "低俗",
            "severity": "high",
            "position": "ocr_text",
        },
        locale="zh",
    )
    assert "色情内容" in row and "读字分支" in row and "低俗" in row
    assert "[高]" in row


def test_detection_method_label_zh_en():
    assert label_for_violation_type("text_detector") == "文本检测工具命中"
    assert label_for_violation_type("text_detector", locale="en") == "Text detector hit"


def test_format_violation_row_video_position_localized():
    row_zh = format_violation_row_for_report(
        {
            "type": "illegal",
            "content": "恋童",
            "severity": "high",
            "position": "frames 0-5 (6 sampled-frame hits merged)",
        },
        locale="zh",
    )
    assert "违法信息" in row_zh and "次采样命中已合并" in row_zh and "恋童" in row_zh
    row_en = format_violation_row_for_report(
        {
            "type": "illegal",
            "content": "test",
            "severity": "high",
            "position": "frames 0-5 (6 sampled-frame hits merged)",
        },
        locale="en",
    )
    assert "Illegal content" in row_en and "Frames 0" in row_en and "sampled hits merged" in row_en
    assert "[High]" in row_en


def test_category_labels_wordlist_violations_carry_content_type():
    viol = [
        {"type": "illegal", "detection_method": "wordlist", "content": "恋童"},
        {"type": "illegal", "detection_method": "wordlist", "content": "枪支"},
    ]
    assert violation_category_labels(viol) == ["违法信息"]


def test_category_labels_order_and_skip_dual():
    viol = [
        {"type": "illegal", "content": "a"},
        {"type": "dual_branch_disagreement", "content": "b"},
        {"type": "illegal", "content": "c"},
    ]
    assert violation_category_labels(viol) == ["违法信息"]
