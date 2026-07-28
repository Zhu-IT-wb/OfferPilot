from pathlib import Path

from app.repositories.interview_knowledge_repository import (
    InMemoryInterviewKnowledgeRepository,
)
from app.repositories.sqlite_interview_knowledge_repository import (
    SQLiteInterviewKnowledgeRepository,
)
from app.core.config import Settings
from app.services.knowledge_dependencies import build_default_knowledge_repository
from app.services.knowledge_corpus import KnowledgeCorpus, KnowledgeCorpusSyncError


def test_atomic_markdown_becomes_one_structured_question(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    question_file = source_root / "rag" / "1. 什么是 RAG？.md"
    question_file.parent.mkdir(parents=True)
    question_file.write_text(
        """# 1. 什么是 RAG？

来源：[https://example.com/rag](https://example.com/rag)

## 💡 简要回答

RAG 会先从外部知识库检索相关内容；再把检索结果交给大模型生成答案。

## 📝 详细解析

离线阶段负责切分和建立索引，在线阶段负责检索、精排与生成。
工具名 read_file 和空字符 \\0 的写法必须保留。

```python
def __init__(self):
    self.tool_name = "read_file"
```

## 🎯 面试总结

这里是总结，不属于完整解析。

扫码关注推广内容。
""",
        encoding="utf-8",
    )
    corpus = KnowledgeCorpus(
        repository=InMemoryInterviewKnowledgeRepository(),
        source_root=source_root,
    )

    report = corpus.sync_markdown()
    questions = corpus.list_questions(module_id="rag")

    assert report.discovered_files == 1
    assert report.imported_questions == 1
    assert report.skipped_files == 0
    assert len(questions) == 1
    question = questions[0]
    assert question.id == "knowledge_rag_001"
    assert question.prompt == "什么是 RAG？"
    assert question.chapter_title == "基础与选型"
    assert question.short_reference_answer == (
        "RAG 会先从外部知识库检索相关内容；再把检索结果交给大模型生成答案。"
    )
    assert "离线阶段负责切分" in question.full_reference_answer
    assert "read_file" in question.full_reference_answer
    assert "\\0" in question.full_reference_answer
    assert "__init__" in question.full_reference_answer
    assert "扫码关注" not in question.full_reference_answer
    assert question.source_url == "https://example.com/rag"
    assert question.source_file == "rag/1. 什么是 RAG？.md"
    assert len(question.required_points) == 2


def test_monolithic_markdown_is_split_by_chapter_and_question_headings(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "knowledge"
    traditional = source_root / "传统开发面试题"
    traditional.mkdir(parents=True)
    (traditional / "MySQL面试题.md").write_text(
        """# MySQL 面试题

来源：[https://example.com/mysql](https://example.com/mysql)

## 索引

### [#](https://example.com/mysql#bplus) 为什么 InnoDB 使用 B+ 树？

B+ 树分支多、高度低；叶子节点有序相连，适合范围查询。

### 什么是最左匹配原则？

联合索引按照从左到右的顺序匹配查询条件。

## 事务

这里是章节介绍，不应该成为题目。
""",
        encoding="utf-8",
    )
    (traditional / "Docker面试题.md").write_text(
        """# Docker 面试题

来源：[https://example.com/docker](https://example.com/docker)

## Docker 和虚拟机有什么区别？

| 对比项 | 容器 | 虚拟机 |
| --- | --- | --- |
| 内核 | 共享 | 独立 |

```java
docker.run();
```

容器共享宿主机内核，虚拟机包含完整客户机操作系统。
""",
        encoding="utf-8",
    )
    corpus = KnowledgeCorpus(
        repository=InMemoryInterviewKnowledgeRepository(),
        source_root=source_root,
    )

    report = corpus.sync_markdown()

    assert report.imported_questions == 3
    mysql_questions = corpus.list_questions(module_id="mysql")
    assert [item.prompt for item in mysql_questions] == [
        "为什么 InnoDB 使用 B+ 树？",
        "什么是最左匹配原则？",
    ]
    assert all(item.chapter_title == "索引" for item in mysql_questions)
    assert mysql_questions[0].source_url == "https://example.com/mysql#bplus"
    docker_question = corpus.list_questions(module_id="docker")[0]
    assert docker_question.chapter_title == "综合"
    assert "docker.run();" in docker_question.full_reference_answer
    assert all(
        "|" not in point.description
        and "---" not in point.description
        and "docker.run" not in point.description
        for point in docker_question.required_points
    )
    assert corpus.search("B+ 树", module_id="mysql")[0].prompt == (
        "为什么 InnoDB 使用 B+ 树？"
    )
    original_id = mysql_questions[0].id
    second_original_id = mysql_questions[1].id
    mysql_path = traditional / "MySQL面试题.md"
    renamed_markdown = mysql_path.read_text(encoding="utf-8").replace(
        "为什么 InnoDB 使用 B+ 树？", "InnoDB 为什么选择 B+ 树？"
    ).replace("mysql#bplus", "mysql#innodb-bplus")
    mysql_path.write_text(renamed_markdown, encoding="utf-8")
    corpus.sync_markdown()
    renamed = next(
        item
        for item in corpus.list_questions(module_id="mysql")
        if item.prompt == "InnoDB 为什么选择 B+ 树？"
    )
    assert renamed.id == original_id

    mysql_path.write_text(
        mysql_path.read_text(encoding="utf-8").replace(
            "### 什么是最左匹配原则？",
            "### 什么是覆盖索引？\n\n"
            "查询字段都在索引中时，可以减少回表。\n\n"
            "### 什么是最左匹配原则？",
        ),
        encoding="utf-8",
    )
    corpus.sync_markdown()
    after_insertion = {
        item.prompt: item.id for item in corpus.list_questions(module_id="mysql")
    }
    assert after_insertion["InnoDB 为什么选择 B+ 树？"] == original_id
    assert after_insertion["什么是最左匹配原则？"] == second_original_id


def test_sync_is_incremental_and_removes_deleted_source_questions(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    source_file = source_root / "agent" / "1. 什么是 Agent？.md"
    source_file.parent.mkdir(parents=True)
    template = """# 1. 什么是 Agent？

来源：[https://example.com/agent](https://example.com/agent)

## 简要回答

Agent 会围绕目标自主规划；并通过工具执行动作。

## 详细解析

{detail}

## 面试总结
不导入。
"""
    source_file.write_text(template.format(detail="第一版解析。"), encoding="utf-8")
    corpus = KnowledgeCorpus(
        repository=InMemoryInterviewKnowledgeRepository(),
        source_root=source_root,
    )

    first = corpus.sync_markdown()
    question_id = corpus.list_questions()[0].id
    second = corpus.sync_markdown()
    source_file.write_text(template.format(detail="第二版解析。"), encoding="utf-8")
    third = corpus.sync_markdown()
    source_file.unlink()
    fourth = corpus.sync_markdown(force=True)

    assert first.new_questions == 1
    assert second.unchanged_questions == 1
    assert third.updated_questions == 1
    assert corpus.get_question(question_id) is None
    assert fourth.removed_questions == 1


def test_sync_rejects_any_unforced_corpus_drop(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    agent = source_root / "agent"
    agent.mkdir(parents=True)
    template = """# {number}. {prompt}

## 简要回答
这是足够用于生成评分点的简要回答；它包含第二个关键要点。

## 详细解析
这里是完整解析。

## 面试总结
不导入。
"""
    for number in range(1, 6):
        (agent / f"{number}. 第 {number} 题？.md").write_text(
            template.format(number=number, prompt=f"第 {number} 题？"),
            encoding="utf-8",
        )
    repository = InMemoryInterviewKnowledgeRepository()
    corpus = KnowledgeCorpus(repository, source_root)
    corpus.sync_markdown()
    (agent / "5. 第 5 题？.md").unlink()

    try:
        corpus.sync_markdown()
    except KnowledgeCorpusSyncError as exc:
        assert "refusing" in str(exc).lower()
    else:
        raise AssertionError("Any unforced corpus drop must be rejected.")

    assert len(corpus.list_questions()) == 5


def test_sync_rejects_same_size_catalog_with_most_ids_replaced(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    agent = source_root / "agent"
    agent.mkdir(parents=True)
    template = """# {number}. 第 {number} 题？

## 简要回答
这是一个核心要点；这是另一个核心要点。

## 详细解析
这是完整解析。

## 面试总结
不导入。
"""
    for number in range(1, 6):
        (agent / f"{number}. 第 {number} 题？.md").write_text(
            template.format(number=number), encoding="utf-8"
        )
    corpus = KnowledgeCorpus(InMemoryInterviewKnowledgeRepository(), source_root)
    corpus.sync_markdown()
    for number in range(1, 6):
        (agent / f"{number}. 第 {number} 题？.md").unlink()
        replacement = number + 5
        (agent / f"{replacement}. 第 {replacement} 题？.md").write_text(
            template.format(number=replacement), encoding="utf-8"
        )

    try:
        corpus.sync_markdown()
    except KnowledgeCorpusSyncError as exc:
        assert "question ids" in str(exc).lower()
    else:
        raise AssertionError("Large-scale question ID churn must be rejected.")

    assert corpus.get_question("knowledge_agent_001") is not None


def test_manifest_protects_first_sync_from_partial_source_mount(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    agent = source_root / "agent"
    agent.mkdir(parents=True)
    (source_root / ".offerpilot-corpus.json").write_text(
        '{"schema_version": 1, "minimum_markdown_files": 2, "minimum_questions": 2}',
        encoding="utf-8",
    )
    (agent / "1. 第 1 题？.md").write_text(
        """# 1. 第 1 题？

## 简要回答
核心要点一；核心要点二。

## 详细解析
这是完整解析。

## 面试总结
不导入。
""",
        encoding="utf-8",
    )
    corpus = KnowledgeCorpus(InMemoryInterviewKnowledgeRepository(), source_root)

    try:
        corpus.sync_markdown()
    except KnowledgeCorpusSyncError as exc:
        assert "manifest" in str(exc).lower()
    else:
        raise AssertionError("A partial first sync must fail its corpus manifest.")

    assert corpus.list_questions() == []


def test_sqlite_corpus_round_trips_source_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    source_file = source_root / "llm" / "1. 什么是大语言模型？.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        """# 1. 什么是大语言模型？

来源：[https://example.com/llm](https://example.com/llm)

## 简要回答
大语言模型通过海量文本训练学习语言规律；能够完成多种生成任务。

## 详细解析
模型通常基于 Transformer，并通过预训练和后训练获得能力。

## 面试总结
不导入。
""",
        encoding="utf-8",
    )
    repository = SQLiteInterviewKnowledgeRepository(
        str(tmp_path / "offerpilot.db"), questions=[]
    )
    corpus = KnowledgeCorpus(repository=repository, source_root=source_root)

    corpus.sync_markdown()
    restored = corpus.get_question("knowledge_llm_001")

    assert restored is not None
    assert restored.source_file == "llm/1. 什么是大语言模型？.md"
    assert restored.source_heading == "什么是大语言模型？"
    assert len(restored.content_hash) == 64
    assert "llm" in restored.keywords


def test_sqlite_incremental_sync_survives_repository_restart(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    source_file = source_root / "rag" / "1. 什么是 RAG？.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        """# 1. 什么是 RAG？

## 简要回答
RAG 先检索外部资料；再让模型生成答案。

## 详细解析
知识保存在模型参数之外，因此可以独立更新。

## 面试总结
不导入。
""",
        encoding="utf-8",
    )
    database = str(tmp_path / "offerpilot.db")
    first_repository = SQLiteInterviewKnowledgeRepository(
        database, initialize_questions=False
    )
    first = KnowledgeCorpus(first_repository, source_root).sync_markdown()
    reopened_repository = SQLiteInterviewKnowledgeRepository(
        database, initialize_questions=False
    )

    second = KnowledgeCorpus(reopened_repository, source_root).sync_markdown()

    assert first.new_questions == 1
    assert second.unchanged_questions == 1
    assert second.updated_questions == 0


def test_default_repository_syncs_configured_markdown_corpus(tmp_path: Path) -> None:
    source_root = tmp_path / "knowledge"
    source_file = source_root / "tools" / "1. 什么是 Function Calling？.md"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        """# 1. 什么是 Function Calling？

来源：[https://example.com/tools](https://example.com/tools)

## 简要回答
Function Calling 让模型输出结构化工具调用意图；真正的工具由应用执行。

## 详细解析
模型负责选择工具和参数，宿主应用负责校验、执行并返回结果。

## 面试总结
不导入。
""",
        encoding="utf-8",
    )
    settings = Settings(
        storage_backend="memory",
        knowledge_source_path=str(source_root),
        knowledge_markdown_sync_enabled=True,
    )

    repository = build_default_knowledge_repository(settings)

    assert [item.id for item in repository.list_questions()] == [
        "knowledge_tools_001"
    ]


def test_default_repository_can_defer_markdown_sync_until_startup(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "knowledge"
    source_root.mkdir()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Markdown sync must not run while dependencies import.")

    monkeypatch.setattr(KnowledgeCorpus, "sync_markdown", fail_if_called)
    repository = build_default_knowledge_repository(
        Settings(
            storage_backend="memory",
            knowledge_source_path=str(source_root),
            knowledge_markdown_sync_enabled=True,
        ),
        sync_markdown=False,
    )

    assert len(repository.list_questions()) >= 12
