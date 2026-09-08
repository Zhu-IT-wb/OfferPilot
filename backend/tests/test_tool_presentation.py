import json

from app.tools.agent_tool import AgentToolDefinition, ToolEffect, ToolPresentation
from app.tools.runtime_tools import REQUIRED_RUNTIME_TOOLS
from app.tools.tool_presentation import (
    presentation_for,
    redact_tool_identifiers,
    registered_tool_names,
    render_approval_prompt,
)


def test_every_required_runtime_tool_has_a_user_facing_name() -> None:
    assert REQUIRED_RUNTIME_TOOLS <= registered_tool_names()
    for tool_name in REQUIRED_RUNTIME_TOOLS:
        presentation = presentation_for(tool_name)
        assert presentation.display_name != "待处理操作"
        assert tool_name not in presentation.display_name


def test_application_bitable_link_uses_a_user_facing_name() -> None:
    assert presentation_for("get_application_bitable_link").display_name == "获取投递表链接"


def test_application_approval_is_a_readable_operation_summary() -> None:
    prompt = render_approval_prompt(
        "create_application",
        {
            "company": "百度",
            "role": "AI 应用工程师",
            "base_location": "北京",
            "jd_keywords": ["Agent", "RAG"],
        },
        effect=ToolEffect.EXTERNAL_WRITE,
    )

    assert prompt == (
        "请确认以下操作：新增投递记录\n\n"
        "公司：百度\n"
        "岗位：AI 应用工程师\n"
        "工作地点：北京\n\n"
        "影响：投递信息将保存到 OfferPilot，并同步到已连接的飞书服务。\n\n"
        "是否确认？"
    )
    assert "create_application" not in prompt
    assert "jd_keywords" not in prompt


def test_all_catalog_approval_prompts_hide_internal_names() -> None:
    for tool_name in registered_tool_names():
        prompt = render_approval_prompt(
            tool_name,
            {},
            effect=ToolEffect.LOCAL_WRITE,
        )
        assert tool_name not in prompt
        assert "待处理操作" not in prompt


def test_unknown_extension_tool_uses_a_non_leaking_fallback() -> None:
    prompt = render_approval_prompt(
        "vendor_private_write_v2",
        {"private_record_id": "secret-id"},
        effect=ToolEffect.EXTERNAL_WRITE,
    )

    assert "待处理操作" in prompt
    assert "vendor_private_write_v2" not in prompt
    assert "private_record_id" not in prompt
    assert "secret-id" not in prompt


def test_explicit_definition_presentation_overrides_the_catalog() -> None:
    definition = AgentToolDefinition(
        name="custom_calendar_write",
        description="test",
        parameters={"type": "object"},
        presentation=ToolPresentation(
            "同步候选日程",
            ("company",),
            "候选日程将写入外部日历。",
        ),
    )

    prompt = render_approval_prompt(
        definition.name,
        {"company": "字节跳动"},
        definition=definition,
        effect=ToolEffect.EXTERNAL_WRITE,
    )

    assert "同步候选日程" in prompt
    assert "公司：字节跳动" in prompt
    assert definition.name not in prompt


def test_user_text_redaction_covers_prompts_errors_and_model_replies() -> None:
    raw = (
        "准备执行 `create_application`；如果失败，再调用 "
        "sync_study_plan_to_calendar。调用 call_private_123 对应 "
        "agent_op_0123456789abcdef，状态为 write_outcome_unknown。"
    )

    sanitized = redact_tool_identifiers(raw)

    assert "create_application" not in sanitized
    assert "sync_study_plan_to_calendar" not in sanitized
    assert "新增投递记录" in sanitized
    assert "同步复习计划到飞书日历" in sanitized
    assert "call_private_123" not in sanitized
    assert "agent_op_0123456789abcdef" not in sanitized
    assert "write_outcome_unknown" not in sanitized
    assert "_" not in json.dumps(sanitized, ensure_ascii=False)


def test_redaction_removes_exact_runtime_identifiers_even_without_known_prefixes() -> None:
    sanitized = redact_tool_identifiers(
        "内部响应：arbitrary-model-id / vendor_specific_failure",
        internal_identifiers=("arbitrary-model-id", "vendor_specific_failure"),
    )

    assert sanitized == "内部响应：内部标识 / 内部标识"
