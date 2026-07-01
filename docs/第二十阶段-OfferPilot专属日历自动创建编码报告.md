# 第二十阶段：OfferPilot 专属日历自动创建编码报告

## 阶段目标

本阶段目标是把飞书日历联动从“依赖用户手动配置某个 calendar_id”推进到“OfferPilot 自动管理自己的秋招日历”。

用户在飞书里说出类似“后天下午三点腾讯 AI 应用开发岗位一面”后，系统应完成：

1. 识别为投递进度/面试安排。
2. 保存投递记录和面试安排。
3. 首次需要同步日历时，自动创建 `OfferPilot 秋招日历`。
4. 将自动创建得到的 `calendar_id` 持久化。
5. 后续面试日程复用该日历，不重复创建。
6. 在该日历内创建面试日程，并设置默认提前 30 分钟提醒。

## 关键实现

### 1. 增加运行时配置存储

新增 Repository 层通用运行配置接口：

- `get_runtime_setting(key: str) -> Optional[str]`
- `set_runtime_setting(key: str, value: str) -> None`

内存版 Repository 使用 `runtime_settings` 字典保存。

SQLite 版 Repository 新增表：

```sql
CREATE TABLE IF NOT EXISTS runtime_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
)
```

当前用于保存：

```text
feishu.offerpilot_calendar_id
```

### 2. FeishuCalendarService 支持创建共享日历

新增：

- `FeishuCalendarResult`
- `FeishuCalendarService.create_shared_calendar()`
- `FeishuCalendarService.should_manage_offerpilot_calendar()`

调用飞书官方“创建共享日历”接口：

```text
POST /calendar/v4/calendars
```

默认创建：

```text
OfferPilot 秋招日历
```

默认权限：

```text
private
```

### 3. 面试日程同步前自动确保日历存在

在 `offerpilot_tools._sync_interview_schedule_to_calendar()` 中新增：

```text
_ensure_calendar_for_sync()
```

逻辑：

1. 如果用户显式配置了真实 `FEISHU_CALENDAR_ID`，直接使用该日历。
2. 如果仍是默认 `primary`，且自动创建开关开启，则进入 OfferPilot 托管日历模式。
3. 先从 Repository 读取已保存的 `calendar_id`。
4. 如果没有，则调用飞书 API 创建 `OfferPilot 秋招日历`。
5. 创建成功后保存 `calendar_id`。
6. 后续日程都写入该日历。

### 4. 新增配置项

新增默认配置：

```env
FEISHU_CALENDAR_AUTO_CREATE_ENABLED=true
FEISHU_OFFERPILOT_CALENDAR_SUMMARY=OfferPilot 秋招日历
FEISHU_OFFERPILOT_CALENDAR_DESCRIPTION=OfferPilot 自动创建，用于记录秋招投递、笔试和面试提醒。
FEISHU_OFFERPILOT_CALENDAR_PERMISSIONS=private
```

说明：当前无需强制手动配置 `FEISHU_CALENDAR_ID`。如果保持默认 `primary`，系统会优先使用自动创建的 OfferPilot 专属日历。

## 影响文件

- `backend/app/core/config.py`
- `backend/app/services/feishu_service.py`
- `backend/app/tools/offerpilot_tools.py`
- `backend/app/repositories/offerpilot_repository.py`
- `backend/app/repositories/sqlite_offerpilot_repository.py`
- `backend/tests/test_feishu_service.py`
- `backend/tests/test_offerpilot_repository.py`
- `backend/tests/test_sqlite_offerpilot_repository.py`
- `backend/tests/test_offerpilot_tools.py`

## 测试结果

在 `backend` 目录运行：

```bash
.venv/bin/python -m pytest
```

结果：

```text
85 passed
```

## 当前能力边界

本阶段已经实现：

- 自动创建 OfferPilot 专属飞书共享日历。
- 自动保存并复用 `calendar_id`。
- 面试安排自动创建飞书日程。
- 日程默认提前 30 分钟提醒。

仍需后续增强：

- 如果要让日程百分百出现在用户个人主日历中，需要补用户身份授权 OAuth，或把用户加入日程参与人/共享日历成员。
- 后续可以增加“查看 OfferPilot 秋招日历链接/日历 ID”的调试接口，方便联调时定位日历。

