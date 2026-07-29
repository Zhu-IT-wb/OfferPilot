import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

from app.models.interview_knowledge import (
    KnowledgeDifficulty,
    KnowledgeQuestion,
    KnowledgeRubricKind,
    KnowledgeRubricPoint,
)
from app.repositories.interview_knowledge_repository import InterviewKnowledgeRepository
from app.services.knowledge_markdown import markdown_table_rows


_ATOMIC_MODULES = {
    "agent": (10, "Agent"),
    "rag": (20, "RAG"),
    "tools": (30, "LLM 工具调用"),
    "llm": (40, "大模型工程"),
}
_ATOMIC_CHAPTERS = {
    "agent": (
        (3, "基础概念"),
        (6, "设计范式"),
        (9, "工程与记忆"),
        (16, "高级设计"),
    ),
    "rag": (
        (3, "基础与选型"),
        (9, "索引构建"),
        (16, "检索优化"),
        (20, "生产与评测"),
    ),
    "tools": (
        (3, "Function Calling"),
        (8, "MCP"),
        (11, "Agent Skill"),
        (16, "协议与网关"),
    ),
    "llm": (
        (5, "基础原理"),
        (11, "训练与微调"),
        (15, "推理与生成"),
        (18, "应用与 Prompt"),
        (20, "架构与部署"),
        (22, "评测与选型"),
    ),
}
_TRADITIONAL_MODULES = {
    "Java基础面试题": (100, "java_basics", "Java 基础"),
    "Java集合面试题": (110, "java_collections", "Java 集合"),
    "Java并发编程面试题": (120, "java_concurrency", "Java 并发"),
    "Java虚拟机面试题": (130, "jvm", "JVM"),
    "Spring面试题": (140, "spring", "Spring"),
    "MySQL面试题": (150, "mysql", "MySQL"),
    "Redis面试题": (160, "redis", "Redis"),
    "计算机网络面试题": (170, "network", "计算机网络"),
    "操作系统面试题": (180, "operating_system", "操作系统"),
    "数据结构与算法面试题": (190, "data_structures", "数据结构与算法"),
    "消息队列面试题": (200, "message_queue", "消息队列"),
    "分布式面试题": (210, "distributed", "分布式"),
    "系统设计面试题": (220, "system_design", "系统设计"),
    "Linux命令面试题": (230, "linux", "Linux"),
    "Git面试题": (240, "git", "Git"),
    "Docker面试题": (250, "docker", "Docker"),
}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SOURCE_RE = re.compile(r"^来源：\[[^]]*]\((https?://[^)]+)\)", re.MULTILINE)
_NUMBERED_FILE_RE = re.compile(r"^(\d+)\.\s*")
_PARSER_VERSION = "2026-07-29-v3"
_MINIMUM_RETENTION_RATIO = 1.0
_RUBRIC_TRANSITIONS = {
    "例如",
    "比如",
    "如下图",
    "输出",
    "解释",
    "区别",
    "首先",
    "然后",
    "最后",
}


class KnowledgeCorpusSyncError(RuntimeError):
    """Raised before persistence when a Markdown sync looks unsafe."""


@dataclass(frozen=True)
class KnowledgeSyncReport:
    discovered_files: int
    imported_questions: int
    skipped_files: int
    new_questions: int
    updated_questions: int
    unchanged_questions: int
    removed_questions: int


class KnowledgeCorpus:
    """Owns Markdown ingestion and presents structured questions to callers."""

    def __init__(
        self,
        repository: InterviewKnowledgeRepository,
        source_root: Path,
    ) -> None:
        self.repository = repository
        self.source_root = Path(source_root)

    def sync_markdown(self, force: bool = False) -> KnowledgeSyncReport:
        existing = {
            question.id: question for question in self.repository.list_questions()
        }
        files = sorted(self.source_root.rglob("*.md")) if self.source_root.exists() else []
        manifest = _load_corpus_manifest(self.source_root)
        if manifest and not force:
            minimum_files = manifest.get("minimum_markdown_files", 0)
            if len(files) < minimum_files:
                raise KnowledgeCorpusSyncError(
                    "Refusing to sync: corpus manifest requires at least "
                    f"{minimum_files} Markdown files, but only {len(files)} were found."
                )
        questions: List[KnowledgeQuestion] = []
        skipped = 0
        for path in files:
            parsed = _parse_file(self.source_root, path)
            if not parsed:
                skipped += 1
                continue
            questions.extend(parsed)
        if manifest and not force:
            minimum_questions = manifest.get("minimum_questions", 0)
            if len(questions) < minimum_questions:
                raise KnowledgeCorpusSyncError(
                    "Refusing to sync: corpus manifest requires at least "
                    f"{minimum_questions} questions, but only {len(questions)} were parsed."
                )
        managed_existing = {
            question_id: question
            for question_id, question in existing.items()
            if question.source_file
        }
        questions = _preserve_existing_question_ids(
            questions,
            list(managed_existing.values()),
        )
        imported = {question.id: question for question in questions}
        if len(imported) != len(questions):
            raise KnowledgeCorpusSyncError(
                "Refusing to sync a corpus containing duplicate question IDs."
            )
        if managed_existing and not force:
            retention_ratio = len(imported) / len(managed_existing)
            if retention_ratio < _MINIMUM_RETENTION_RATIO:
                raise KnowledgeCorpusSyncError(
                    "Refusing to replace the last-known-good corpus: "
                    f"question count would fall from {len(managed_existing)} "
                    f"to {len(imported)}. Re-run with force=True after verifying "
                    "the source files and parser output."
                )
            retained_ids = set(managed_existing).intersection(imported)
            id_retention_ratio = len(retained_ids) / len(managed_existing)
            if id_retention_ratio < _MINIMUM_RETENTION_RATIO:
                raise KnowledgeCorpusSyncError(
                    "Refusing to replace the last-known-good corpus because "
                    f"only {len(retained_ids)} of {len(managed_existing)} question IDs "
                    "would be retained. Re-run with force=True only after verifying "
                    "that the ID migration is intentional."
                )
        new_count = sum(question_id not in existing for question_id in imported)
        updated_count = sum(
            question_id in existing
            and existing[question_id].content_hash != question.content_hash
            for question_id, question in imported.items()
        )
        unchanged_count = sum(
            question_id in existing
            and existing[question_id].content_hash == question.content_hash
            for question_id, question in imported.items()
        )
        removed_count = sum(
            question_id not in imported for question_id in managed_existing
        )
        self.repository.upsert_questions(questions)
        return KnowledgeSyncReport(
            discovered_files=len(files),
            imported_questions=len(questions),
            skipped_files=skipped,
            new_questions=new_count,
            updated_questions=updated_count,
            unchanged_questions=unchanged_count,
            removed_questions=removed_count,
        )

    def get_question(self, question_id: str) -> Optional[KnowledgeQuestion]:
        return self.repository.get_question(question_id)

    def list_questions(
        self,
        module_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
    ) -> List[KnowledgeQuestion]:
        return [
            question
            for question in self.repository.list_questions()
            if (module_id is None or question.module_id == module_id)
            and (chapter_id is None or question.chapter_id == chapter_id)
        ]

    def search(
        self, query: str, module_id: Optional[str] = None, limit: int = 20
    ) -> List[KnowledgeQuestion]:
        terms = [item.casefold() for item in query.split() if item]
        if not terms:
            return []
        candidates = self.list_questions(module_id=module_id)
        scored = []
        for question in candidates:
            prompt = question.prompt.casefold()
            body = f"{question.short_reference_answer} {question.full_reference_answer}".casefold()
            score = sum(5 for term in terms if term in prompt) + sum(
                1 for term in terms if term in body
            )
            if score:
                scored.append((score, question))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [question for _, question in scored[: max(limit, 0)]]


def _parse_file(source_root: Path, path: Path) -> List[KnowledgeQuestion]:
    atomic = _parse_atomic_question(source_root, path)
    if atomic is not None:
        return [atomic]
    return _parse_monolithic_questions(source_root, path)


def _load_corpus_manifest(source_root: Path) -> dict[str, int]:
    manifest_path = source_root / ".offerpilot-corpus.json"
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("manifest root must be an object")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported schema_version")
        return {
            "minimum_markdown_files": max(
                int(payload.get("minimum_markdown_files", 0)), 0
            ),
            "minimum_questions": max(int(payload.get("minimum_questions", 0)), 0),
        }
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise KnowledgeCorpusSyncError(
            f"Refusing to sync because the corpus manifest is invalid: {exc}"
        ) from exc


def _preserve_existing_question_ids(
    questions: List[KnowledgeQuestion],
    existing: List[KnowledgeQuestion],
) -> List[KnowledgeQuestion]:
    available = {question.id: question for question in existing}
    reconciled = []
    for question in questions:
        candidates = [
            item
            for item in available.values()
            if item.source_file == question.source_file
        ]
        matched = available.get(question.id)
        if matched is None or matched.source_file != question.source_file:
            matched = next(
                (
                    item
                    for item in candidates
                    if item.source_heading == question.source_heading
                ),
                None,
            )
        if matched is None:
            normalized_answer = _normalized_identity_text(
                question.full_reference_answer
            )
            answer_matches = [
                item
                for item in candidates
                if _normalized_identity_text(item.full_reference_answer)
                == normalized_answer
            ]
            matched = answer_matches[0] if len(answer_matches) == 1 else None
        if matched is None:
            reconciled.append(question)
            continue
        available.pop(matched.id, None)
        reconciled.append(_replace_question_id(question, matched.id))
    return reconciled


def _replace_question_id(
    question: KnowledgeQuestion, stable_id: str
) -> KnowledgeQuestion:
    if question.id == stable_id:
        return question
    points = [
        replace(
            point,
            id=f"{stable_id}_{point.kind.value}_{index}",
        )
        for index, point in enumerate(question.rubric_points, start=1)
    ]
    return replace(question, id=stable_id, rubric_points=points)


def _normalized_identity_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _parse_atomic_question(
    source_root: Path, path: Path
) -> Optional[KnowledgeQuestion]:
    relative = path.relative_to(source_root)
    if len(relative.parts) != 2 or relative.parts[0] not in _ATOMIC_MODULES:
        return None
    number_match = _NUMBERED_FILE_RE.match(path.stem)
    if number_match is None:
        return None
    number = int(number_match.group(1))
    markdown = path.read_text(encoding="utf-8")
    sections = _markdown_sections(markdown)
    title = next((text for level, text, _ in sections if level == 1), path.stem)
    prompt = _strip_number(_clean_inline(title))
    short = _section_text(markdown, sections, "简要回答", stop_level=2)
    full = _section_text(
        markdown,
        sections,
        "详细解析",
        stop_level=2,
        stop_titles=("面试总结",),
    )
    if not prompt or not short or not full:
        return None
    module_id = relative.parts[0]
    module_order, module_title = _ATOMIC_MODULES[module_id]
    chapter_order, chapter_title = _atomic_chapter(module_id, number)
    question_id = f"knowledge_{module_id}_{number:03d}"
    source_match = _SOURCE_RE.search(markdown)
    source_url = source_match.group(1) if source_match else ""
    points = _draft_rubric_points(question_id, short, full)
    return KnowledgeQuestion(
        id=question_id,
        module_id=module_id,
        module_title=module_title,
        module_order=module_order,
        chapter_id=f"{module_id}_chapter_{chapter_order}",
        chapter_title=chapter_title,
        chapter_order=chapter_order,
        question_order=number,
        prompt=prompt,
        difficulty=KnowledgeDifficulty.INTERMEDIATE,
        frequency=3,
        short_reference_answer=short,
        full_reference_answer=full,
        rubric_points=points,
        hint="可以从“" + "、".join(point.label for point in points[:4]) + "”展开。",
        source_title=module_title,
        source_url=source_url,
        source_chapter=prompt,
        source_file=relative.as_posix(),
        source_heading=prompt,
        content_hash=_content_hash(markdown),
        keywords=[module_id, *[point.label for point in points]],
    )


def _parse_monolithic_questions(
    source_root: Path, path: Path
) -> List[KnowledgeQuestion]:
    relative = path.relative_to(source_root)
    if (
        len(relative.parts) != 2
        or relative.parts[0] != "传统开发面试题"
        or path.stem not in _TRADITIONAL_MODULES
    ):
        return []
    module_order, module_id, module_title = _TRADITIONAL_MODULES[path.stem]
    markdown = path.read_text(encoding="utf-8")
    sections = _markdown_sections(markdown)
    source_match = _SOURCE_RE.search(markdown)
    document_source_url = source_match.group(1) if source_match else ""
    current_chapter = "综合"
    chapter_order_by_title = {current_chapter: 1}
    result = []
    question_order = 0
    for index, (level, title, start) in enumerate(sections):
        if level == 1:
            continue
        title = _clean_heading(title)
        is_question = level >= 3 or (level == 2 and _looks_like_question(title))
        if level == 2 and not is_question:
            current_chapter = title
            chapter_order_by_title.setdefault(
                current_chapter, len(chapter_order_by_title) + 1
            )
            continue
        if not is_question:
            continue
        content_start = markdown.find("\n", start) + 1
        heading_line = markdown[start:content_start]
        heading_url_match = re.search(r"\]\((https?://[^)]+)\)", heading_line)
        question_source_url = (
            heading_url_match.group(1) if heading_url_match else document_source_url
        )
        content_end = len(markdown)
        for next_level, _, next_start in sections[index + 1 :]:
            if next_level <= level:
                content_end = next_start
                break
        full_answer = _clean_markdown(markdown[content_start:content_end])
        if not full_answer:
            continue
        question_order += 1
        short_answer = _short_answer(full_answer)
        stable_anchor = heading_url_match.group(1) if heading_url_match else ""
        identity = (
            f"{relative.as_posix()}\n{stable_anchor}"
            if stable_anchor
            else f"{relative.as_posix()}\n{title}"
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        question_id = f"knowledge_{module_id}_{digest}"
        points = _draft_rubric_points(question_id, short_answer, full_answer)
        chapter_id = (
            f"{module_id}_"
            + hashlib.sha1(current_chapter.encode("utf-8")).hexdigest()[:8]
        )
        result.append(
            KnowledgeQuestion(
                id=question_id,
                module_id=module_id,
                module_title=module_title,
                module_order=module_order,
                chapter_id=chapter_id,
                chapter_title=current_chapter,
                chapter_order=chapter_order_by_title[current_chapter],
                question_order=question_order,
                prompt=title,
                difficulty=KnowledgeDifficulty.INTERMEDIATE,
                frequency=2,
                short_reference_answer=short_answer,
                full_reference_answer=full_answer,
                rubric_points=points,
                hint="可以从“"
                + "、".join(point.label for point in points[:4])
                + "”展开。",
                source_title=module_title,
                source_url=question_source_url,
                source_chapter=current_chapter,
                source_file=relative.as_posix(),
                source_heading=title,
                content_hash=_content_hash(markdown[start:content_end]),
                keywords=[module_id, current_chapter, *[point.label for point in points]],
            )
        )
    return result


def _markdown_sections(markdown: str) -> List[tuple[int, str, int]]:
    sections = []
    for match in re.finditer(r"^#{1,6}\s+.+$", markdown, re.MULTILINE):
        heading_match = _HEADING_RE.match(match.group(0))
        if heading_match:
            sections.append(
                (len(heading_match.group(1)), _clean_inline(heading_match.group(2)), match.start())
            )
    return sections


def _section_text(
    markdown: str,
    sections: List[tuple[int, str, int]],
    title_fragment: str,
    stop_level: int,
    stop_titles: tuple[str, ...] = (),
) -> str:
    for index, (level, title, start) in enumerate(sections):
        if title_fragment not in title:
            continue
        content_start = markdown.find("\n", start) + 1
        content_end = len(markdown)
        for next_level, next_title, next_start in sections[index + 1 :]:
            if any(fragment in next_title for fragment in stop_titles):
                content_end = next_start
                break
            if next_level <= min(level, stop_level):
                content_end = next_start
                break
        return _clean_markdown(markdown[content_start:content_end])
    return ""


def _draft_rubric_points(
    question_id: str,
    short_answer: str,
    full_answer: str = "",
) -> List[KnowledgeRubricPoint]:
    candidates = _table_rubric_candidates(full_answer or short_answer)
    for line in short_answer.splitlines():
        if _is_rubric_noise(line):
            continue
        for item in re.split(r"[；;。]+", line):
            candidate = item.strip(" ：:，,。；;-")
            if (
                candidate
                and candidate not in candidates
                and not _is_rubric_noise(candidate)
            ):
                candidates.append(candidate)
        if len(candidates) >= 6:
            break
    candidates = candidates[:6]
    if not candidates:
        fallback = next(
            (
                line.strip()
                for line in short_answer.splitlines()
                if line.strip() and not _is_rubric_noise(line)
            ),
            "围绕题目给出准确、完整的核心回答",
        )
        candidates = [fallback]
    points = []
    label_counts: dict[str, int] = {}
    for index, text in enumerate(candidates, start=1):
        base_label = _point_label(text)
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
        occurrence = label_counts[base_label]
        if occurrence == 1:
            label = base_label
        else:
            suffix = f"（{occurrence}）"
            label = f"{base_label[: 32 - len(suffix)]}{suffix}"
        points.append(
            KnowledgeRubricPoint(
                id=f"{question_id}_required_{index}",
                kind=KnowledgeRubricKind.REQUIRED,
                label=label,
                description=text,
            )
        )
    return points


def _table_rubric_candidates(markdown: str) -> List[str]:
    candidates = []
    for cells in markdown_table_rows(markdown):
        nonempty_cells = [cell for cell in cells if cell]
        if len(nonempty_cells) < 2:
            continue
        key, *details = nonempty_cells
        candidate = f"{key}：" + "；".join(details)
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= 6:
            break
    return candidates


def _point_label(text: str) -> str:
    normalized = re.sub(r"^(RAG|答案|核心|首先|然后|最后)\s*", "", text).strip()
    return normalized[:32] or "核心要点"


def _is_rubric_noise(value: str) -> bool:
    stripped = value.strip()
    if not stripped or stripped.startswith(("```", "~~~", "|")):
        return True
    transition = stripped.strip(" ：:，,。；;-_").casefold()
    if transition in _RUBRIC_TRANSITIONS:
        return True
    if re.fullmatch(r"[:|+\-=\s]+", stripped):
        return True
    if re.fullmatch(r"[{}()[\];,.<>/=+*'\"`_:\-\w]+", stripped) and not re.search(
        r"[\u4e00-\u9fff]", stripped
    ):
        return True
    meaningful = re.findall(r"[A-Za-z\u4e00-\u9fff]", stripped)
    return len(meaningful) < 2


def _short_answer(full_answer: str, max_chars: int = 420) -> str:
    paragraphs = [item.strip() for item in full_answer.split("\n") if item.strip()]
    selected = []
    length = 0
    for paragraph in paragraphs:
        if selected and length + len(paragraph) > max_chars:
            break
        selected.append(paragraph)
        length += len(paragraph)
        if length >= 120:
            break
    return "\n".join(selected)[:max_chars]


def _clean_inline(value: str) -> str:
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = re.sub(r"__([^_]+)__", r"\1", value)
    value = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", value)
    value = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", value)
    return value.strip()


def _clean_heading(value: str) -> str:
    return re.sub(r"^#\s*", "", _clean_inline(value)).strip()


def _looks_like_question(value: str) -> bool:
    if value.endswith(("?", "？")):
        return True
    prefixes = (
        "什么",
        "为什么",
        "如何",
        "怎么",
        "哪些",
        "讲讲",
        "说说",
        "介绍",
        "请",
        "有没有",
        "是否",
        "对比",
        "区别",
        "如果",
        "让你",
    )
    return value.startswith(prefixes)


def _clean_markdown(value: str) -> str:
    value = re.sub(r"!\[[^]]*]\([^)]+\)", "", value)
    value = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value)
    lines = []
    in_code_fence = False
    for line in value.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            lines.append(line.rstrip())
            continue
        if not stripped or re.fullmatch(r"[-*_]{3,}", stripped):
            continue
        if not in_code_fence and re.fullmatch(r"<[^>]+>", stripped):
            continue
        stripped = re.sub(r"^>\s*", "", stripped)
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
        lines.append(_clean_inline(stripped))
    return "\n".join(line for line in lines if line)


def _strip_number(value: str) -> str:
    return re.sub(r"^\d+[\\]?[.、｜|]\s*", "", value).strip()


def _content_hash(value: str) -> str:
    return hashlib.sha256(
        f"{_PARSER_VERSION}\n{value}".encode("utf-8")
    ).hexdigest()


def _atomic_chapter(module_id: str, number: int) -> tuple[int, str]:
    for index, (last_number, title) in enumerate(
        _ATOMIC_CHAPTERS[module_id], start=1
    ):
        if number <= last_number:
            return index, title
    return len(_ATOMIC_CHAPTERS[module_id]) + 1, "其他"
