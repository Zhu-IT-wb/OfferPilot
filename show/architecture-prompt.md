# OfferPilot Architecture Image Prompt

Generated with the built-in ImageGen tool. Target asset: `architecture.png`.

```text
Use case: infographic-diagram.
Asset type: a polished, friendly, technically accurate architecture image for a Chinese open-source project's GitHub README.
Primary request: Create a single clean 4:3 landscape architecture diagram titled "OfferPilot 系统架构". The audience is software engineers and technical interviewers. Make the structure immediately readable at README width, with fewer crossing wires than an auto-laid-out Mermaid diagram. White background, restrained blue / teal / soft amber accents to distinguish responsibilities, charcoal text, clear Chinese sans-serif typography, strong hierarchy, generous whitespace, crisp simple functional icons. Flat technical editorial design, not a marketing poster. Large, fully legible text, no tiny footnotes. Produce a high-resolution bitmap.

Content and logical relationships:
1. Top entry row, two labeled entry points: "飞书机器人" and "学习与训练 Web". Both point down into one shared "FastAPI" routing strip, with the subtitle "身份校验 · 会话隔离 · 请求路由".
2. Below the API, split into three clearly distinguished execution columns. This is the main reading structure:
   - Wide left column: "主 Agent" / "LangGraph ReAct". Clearly show the cycle "模型决策" → "工具执行" → "结果观察" → back to "模型决策". Under the cycle write "参数校验 · 用户确认 · 完成检查". Tool execution has a small label "统一工具接口". This column is reached by the API branch labeled "自然语言对话".
   - Middle column: "学习与训练服务", with three concise lines "八股复习", "每日刷题", "项目训练". It is reached by the API branch labeled "页面操作".
   - Right column: "源码分析 Agent", with subtitle "独立 Tool Calling Runtime" and short line "浏览 · 搜索 · 精读 · 受限命令". It is reached by the API branch labeled "仓库导入". Beneath it draw a direct arrow to "只读 GitHub Workspace" with subtitle "文件与行号证据".
3. Below the main Agent, one tidy, aligned capability row: "业务服务" with subtitle "投递 · 面试 · 复习计划"; "MCP 只读检索" with subtitle "Qdrant · 稠密 + BM25 · RRF"; "飞书开放平台" with subtitle "多维表格 · 日历". Connect these to the main Agent's tool execution via a short organized branching connector. Tool results feed back into the Agent cycle. Do not draw a tool edge to the separate source-code analysis Agent. Web learning/training services may reuse business services with one clearly labeled dashed connector "复用领域服务".
4. Make persistence a subtle, separate bottom foundation, with two clearly distinct stores, not one combined box:
   - "Checkpoint SQLite" / "图状态 · 暂停与恢复", connected only to the main Agent runtime. Near that connection label "interrupt / resume".
   - "业务 SQLite" / "业务数据 · 会话事件 · 操作回执", connected to business / learning services.
5. Show shared model access as one unobtrusive line or footer band labeled "共享模型服务 · OpenAI-compatible API", visually associated with all three execution columns without a web of arrows. Do not imply that only the main Agent uses an LLM.

Technical constraints:
- The Web branch must bypass the main Agent for direct learning and repository-import workflows.
- Source analysis belongs on the Web/import branch, not the main Agent's registered tool branch.
- MCP retrieves evidence only; no second answer-generation LLM inside MCP or Qdrant.
- interrupt is Runtime behavior; checkpoint is persistence. Do not label the database as performing approvals.
- Use Chinese labels exactly as supplied, with correctly spelled English identifiers.
- Keep connectors orthogonal or cleanly curved, with obvious arrowheads, no line through text, no ambiguous crossing. Use thoughtful alignment and whitespace. It is fine to simplify connector routing as long as these actual dependencies remain true.
- No source-code blocks, no Mermaid text, no fake logos, no people, no mascot, no 3D machinery, no decorative blobs, no watermark, no repository validation metrics, no benchmark table, no current-status claim.
- Return only the finished image.
```

## Connector Correction

The first image was reviewed, then edited with the built-in ImageGen tool using:

```text
Use case: precise-object-edit. Edit target: the attached generated OfferPilot architecture diagram.
Preserve the overall composition, colors, icons, all Chinese and English labels, typography and all correctly drawn upper API branches and Agent loop. Change ONLY the incorrect dependency connectors in the lower half. This is a correctness fix, not a redesign.

Required connector corrections:
1. DELETE the blue arrow from the bottom of the small "业务服务" tile to "Checkpoint SQLite". Business data does not go into checkpoint.
2. Draw one short blue arrow from the OUTER BOUNDARY of the large "主 Agent / LangGraph ReAct" container to "Checkpoint SQLite", with the label "interrupt / resume" placed next to it. It must clearly originate at the Agent runtime container, NOT at any of the three capability tiles. This represents graph-state persistence.
3. Connect the "业务服务" tile to "业务 SQLite" with a neatly routed connector through available whitespace, not through other tiles or labels. Preserve the existing teal connection from "学习与训练服务" to "业务 SQLite". Business SQLite is shared by business and learning services.
4. DELETE the orange dashed connector labeled "复用领域服务" that currently points from "学习与训练服务" to "飞书开放平台", and delete that connector's label too. Do not replace it with another arrow. This optional dependency is better left out of this high-level diagram than connected to the wrong node.
5. DELETE ALL the three upward arrowheads / connector stubs rising from the bottom "共享模型服务 · OpenAI-compatible API" band. The band should be a clean standalone shared-service caption, NOT connected to either SQLite database or Workspace. Preserve the text "为所有执行模块提供统一的模型访问能力".

Strict constraints: Do not alter the top entry nodes, API, three execution columns, source Agent → read-only Workspace connection, inner ReAct cycle or capability labels. Do not add new text or modules. Keep arrows unambiguous, never crossing text, and keep the final image as readable and clean as the original. The final diagram must not imply that databases call language models or that business services use the checkpoint as their business database.
```

## Final Connector Cleanup

```text
Use case: precise-object-edit. Edit the supplied architecture image with ONE very small correction only.
Erase the BLUE vertical downward connector that runs from the "业务服务" tile to the "Checkpoint SQLite" box. In this 1448 by 1086 image, the erroneous blue vertical line is at approximately x=142, y=800 through y=868, and ends in a blue arrowhead above the left database. Also erase the standalone text "interrupt / resume" immediately beside that arrow.
IMPORTANT: Keep the GREEN connector from "业务服务" curving right to "业务 SQLite" intact, including its starting point. Do not erase that green line.
DO NOT draw ANY replacement connector to Checkpoint SQLite. Leave that bottom-left database box as a standalone runtime-persistence foundation; its label "图状态 · 暂停与恢复" is sufficient. No blue line and no blue arrowhead may remain between the business tile and checkpoint box.
Preserve every other pixel as closely as possible: all layout, text, boxes, colors, icons and every other connector stay unchanged. Do not redesign or add anything.
```
