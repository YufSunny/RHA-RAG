# 可视化汽车行业RAG系统示例

**面向汽车行业（尤其是新能源汽车 / NEV）的检索增强生成系统。底层基于 [RHA-RAG](https://github.com/...) 的证明链引擎，将图表可视化作为工具调用能力。**

[English](README.md) | [中文](README_zh.md)

当用户提问 *"某汽车生产商A 2024 年纯电（BEV）与插混（PHEV）销量分别是多少？"* 时，本系统 会**检索**汽车语料中的相关行，**标注**来源文件与字段标签，输出严格的 JSON `ChartSpec`，并以 SSE 流式逐节点推送的方式，在浏览器中渲染一张全宽 **ECharts** 柱状图，支持一键 **⬇ PNG** 下载。切换到 **Full** 模式后，同一问题会经过完整的显式证明链（clarify → grade → reason → verify → answer → visualize），每一步都可审计。

系统自带 **某汽车生产商A** 种子语料（年度 / 季度 / 车型销量、全球市场份额、City-EV 车型发布），**首启即可回答真实的汽车行业问题，零上传开箱即用**。将中汽协（CAAM）/ 乘联会（CPCA）/ 主机厂 TSB / 维修手册等自有文档放入 `data/local/`，同一套流水线即可处理。

技术栈：[LangGraph](https://langchain-ai.github.io/langgraph/)（编排）、[Milvus Lite](https://milvus.io/docs)（向量库）、[PostgreSQL](https://www.postgresql.org/)（对话记忆）、[ECharts](https://echarts.apache.org/)（图表渲染）、[Docker](https://www.docker.com/)（部署）。

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Domain](https://img.shields.io/badge/domain-Automotive%20%2F%20NEV-0066cc.svg)](#-本系统能力)
[![Charts](https://img.shields.io/badge/charts-ECharts%205-aa2233.svg)](https://echarts.apache.org/)
[![Orchestration](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Vector store](https://img.shields.io/badge/vectorstore-Milvus%20Lite-blueviolet.svg)](https://milvus.io/)
[![Memory](https://img.shields.io/badge/memory-PostgreSQL-336791.svg)](https://www.postgresql.org/)

![screenshot](imgs/demo.png)

---

## ✨ 本系统能力

本系统是一套**面向汽车行业领域微调的 RAG + 可视化工具调用**系统，**不是通用聊天机器人**。两项核心能力：

### 1. 引用回答，而非自由联想

最终答案中的每一条声明，都同时引用来源中的**字段标签**（例如 `Table 3 — P0420 故障按年款分布`）**和**来源**文件名**（例如 `toyota-p0420.md`）。检索到的文档一律视为**纯数据**——**绝不视为指令**

### 2. 图表即工具调用

当回答中包含数值数据（销量、故障率、市场份额、月度趋势）时，第二次 LLM 调用会输出一份 **Pydantic 校验过的 `ChartSpec`**：

```text
{type, title, x_label, y_label, data[], citation{filename,label}, confidence, note}
```

前端将其渲染为全宽 ECharts 卡片（坐标轴、tooltip、时序 dataZoom、**⬇ PNG** 导出按钮）。图表与文字答案共享同一份 `citation`。**没有来源引用的图表会在 Pydantic 层被拒收**，杜绝幻觉出图。

图表在**两种模式**下都可用：

- **Fast 模式（默认）** — `generate_query → retrieve → generate_answer → visualize`。低延迟，适合"问答 + 图表"工作流。
- **Full 模式** — `clarify → generate_query → retrieve → grade → reason → verify → generate_answer → visualize`。新增完整证明链，适合新颖或高风险问题。

---

## 🚀 快速开始

### 前置依赖

- **Docker**（推荐）— 已包含 PostgreSQL。
- **Python 3.12+**（备选）— 本地安装使用。

### Docker（推荐）

```bash
cp .env.docker.example .env          # 填入 API keys
docker compose --env-file .env up --build
# → http://localhost:7500  （PORT 环境变量，默认 7500）
```

PostgreSQL 作为伴生容器随应用启动，**无需任何外部服务**。

### 本地安装

#### Linux / macOS

```bash
pip install -r requirements.txt
python server.py
```

#### Windows（PowerShell）

```powershell
pip install -r requirements.txt
python server.py
```

### 配置项

所有可调参数集中在 **[config.py](config.py)**。Docker 模式下将环境变量写入 `.env` 即可。

| 环境变量 | 服务 | 用途 |
| --- | --- | --- |
| `ZAI_API_KEY` | [Z.ai](https://www.z.ai/) | GLM-OCR，用于 PDF / 图片处理 |
| `QWEN_API_KEY` | [DashScope](https://dashscope.aliyun.com/) | 千问 `text-embedding-v4` 向量模型 |
| `OPENAI_API_KEY` | [DeepSeek](https://api.deepseek.com) | DeepSeek V4 Pro 大模型 |
| `PORT` | — | 主机端口:容器端口（默认 `7500`） |
| `DEFAULT_FAST_MODE` | — | `true` / `false` — `/api/chat` 的默认模式（默认 `true`） |

`DATABASE_URL` 设置 PostgreSQL 连接串（传 `""` 即内存模式）。`MAX_HISTORY_TURNS` 控制多轮上下文轮数（默认 6，设为 0 关闭）。`LLM_THINKING` 启用 DeepSeek 思考模式（默认 `true`）。

即使**没有 API key 服务也能正常启动**——先上传文档、配齐 key、再点击 **Re-index** 重建索引即可。

---

## 🧠 工作原理

### Fast 模式（默认）

```text
generate_query → retrieve → generate_answer → visualize → END
```

### Full 模式（证明链）

```text
User Question
    │
    ▼
clarify          将自然语言改写为可验证的目标陈述
    │
    ▼
generate_query   LLM 决策：是检索知识库，还是直接回答？
    │
    ├──(无工具调用)── END
    │
    ▼
retrieve         对本地 Milvus 向量库做语义检索
    │
    ▼
grade            用结构化 LLM 输出（yes/no）评估检索到的文档相关性
    │
    ▼
reason           构建逻辑证明链（@cite / @common / @MP / @TA）
    │
    ▼
verify           按推理规则验证每一推导步骤
    │
    ▼
generate_answer  生成最终回答，每条声明都明确标注来源
    │
    ▼
visualize        工具调用：为数值回答输出一份 ChartSpec（Pydantic）
    │
    ▼
END
```

每个节点通过 SSE（Server-Sent Events）实时推送到 Web UI。点击顶栏 **Full** 切换模式；点击 **Fast** 切回。

**汽车语境下的节点职责：**

1. **clarify（澄清）** — 将问题改写为可验证陈述。例如：*"某汽车生产商A 2024 年 BEV 销量为 1,764,992 辆，这一断言是否成立？"*
2. **generate_query（生成检索查询）** — 决策是否调用 `retrieve`，还是直接回答。
3. **retrieve（检索）** — 在 Milvus 上做语义检索（top-5 切片）——CSV 行、Markdown 章节、TSB 文本，统一以纯文本形式索引。
4. **grade（打分）** — 结构化 LLM 输出对每个召回切片打 `yes`/`no`。**文档一律视为纯数据**（防提示注入加固 —— 假装是"指令"的 PDF 会被拒收）。
5. **reason（推理）** — "你是一名逻辑学家"：构建证明链，每一步必须是 `@cite`、`@common` 或由前序步骤推导得出。
6. **verify（验证）** — 审计证明链。在最终回答写出前，标记矛盾、缺失引用、不可靠的推导。
7. **generate_answer（生成回答）** — 写出最终答案，每个声明都引用来源中的标签（表号、章节标题、行号范围）以及来源文件名。
8. **visualize（可视化）** — **工具调用。** 第二次 LLM 调用审视回答 + 召回上下文，输出严格 JSON 的 `ChartSpec`。服务端 Pydantic 校验，校验失败的 spec 永远到不了前端。客户端 ECharts 渲染。

推理链使用形式化证明记号：

| 标记 | 含义 |
| --- | --- |
| `@cite` | 引自某来源文档的陈述 |
| `@common` | 通识（教科书级共识） |
| `@MP` | Modus Ponens（假言推理）—— 由前序步骤推导 |
| `@TA` | 重言式 / 量词公理 |

> 完整搭建故事——逐文件、含设计决策与踩坑记录——见 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 💬 对话记忆

本聊天支持多轮：同一浏览器会话内，每条新问题都会带入历史 Q&A，因此追问可以正确解析指代（*"它紧凑吗？"* → *"拓扑空间是紧致的吗？"*；*"那些里面哪个跳得最大？"* → 引用上一轮模型的回答），检索也具备上下文感知能力，答案可在前序轮次之上构建。

- **按浏览器会话隔离。** 会话 ID 存于 `localStorage`（按标签页隔离，刷新不丢）。侧边栏列出所有历史会话，点击切换。历史持久化到 PostgreSQL。Postgres 不可达时，服务回退到内存存储（重启清空）。
- **哪些环节使用历史：** `clarify`（解析追问指代）、检索决策（`generate_query`）以及最终 `generate_answer`。`reason` / `verify` 证明链**只锚定在召回来源上**，不带历史以保证可审计。
- **可调深度。** [config.py](config.py) 中的 `MAX_HISTORY_TURNS`（默认 6；设为 0 关闭）。
- **清空聊天。** "Clear chat" 按钮（或 `POST /api/clear`）会从 PostgreSQL 删除该会话的全部历史。

---

## 🏗️ 系统架构

| 组件 | 技术方案 |
| --- | --- |
| 领域 | **汽车行业** —— 主机厂、NEV 市场数据、TSB 公告、维修手册、DTC 故障码 |
| 可视化 | **ECharts 5** via CDN；`ChartSpec` Pydantic 模型；通过 `getDataURL` 导出 PNG |
| 编排 | LangGraph `StateGraph`（8 节点、1 条条件边、`RhaState`） |
| 证明链 | 形式化 `@cite` / `@common` / `@MP` / `@TA` 步骤记号，含 verifier 节点 |
| 对话记忆 | 按会话持久化到 PostgreSQL（回退：内存） |
| LLM | DeepSeek V4 Pro via `ChatDeepSeekFixed`（含 thinking 模式补丁） |
| Embeddings | 千问 `text-embedding-v4`（batch ≤ 10） |
| 向量库 | Milvus Lite（本地文件 `milvus.db`，COSINE / AUTOINDEX） |
| OCR | GLM-OCR via Z.ai（`ZaiClient`，data-URI 格式） |
| PDF 渲染 | PyMuPDF（页面 → PNG → OCR） |
| Web 服务器 | FastAPI + 节点级实时 SSE 流 |
| 前端 | 原生 JS 聊天 App —— 聊天气泡、Markdown / LaTeX、会话侧边栏、ECharts 卡片、PNG 导出、Fast/Full 切换 |
| 部署 | Docker Compose（应用 + PostgreSQL），默认端口 7500 |

---

## 📄 支持的文档

将以下文件放入 `data/local/` 或通过 Web UI 上传：

| 类型 | 扩展名 | 处理方式 |
| --- | --- | --- |
| 纯文本 | `.txt` `.md` | 直接读取 |
| **CSV**（销量表、DTC 列表） | `.csv` | 直接读取（表头 + 行拼接为文本） |
| HTML | `.html` `.htm` | BeautifulSoup 文本抽取 |
| Word | `.docx` | PyMuPDF 文本抽取 |
| PDF（TSB 公告、维修手册） | `.pdf` | GLM-OCR（逐页渲染为 PNG 再 OCR） |
| 图片（仪表盘照片） | `.jpg` `.jpeg` `.png` | GLM-OCR |

OCR 结果缓存为 `<file>.ocr.md`（按 mtime 检查），因此重建索引很快，**仅对变化的文件重做 OCR**。

### 自带种子语料

`data/auto-seed/` 自带 5 份公开来源的 **某汽车生产商A** 文档，**首启即可演示，零上传**：

- `automaker-annual.csv` — 年度产量、销量、BEV/PHEV 分项，中国 vs 海外，2019–2025
- `automaker-quarterly.csv` — 季度销量，2023-Q2 → 2025-Q4
- `automaker-models-2025.csv` — 2025 年某汽车生产商A 前 13 个车型家族销量，含同比变化
- `automaker-market-share.md` — 2025 年全球插混 / 纯电市场份额、2024 年中国 NEV 市场份额（CPCA 数据）
- `automaker-city-ev.md` — City-EV 车型资料（某汽车生产商A Global 2025 年 6 月 30 日新闻稿）

所有数据均来自公开新闻稿、CPCA / CAAM 月度报告、Statista 汇编、CnEVPost 报道。**不含任何专有或受限资料**。来源与许可证见 [data/auto-seed/README.md](data/auto-seed/README.md)。

将你自己的中汽协月度销量、乘联会细分市场份额、主机厂 TSB 公告或维修手册放入 `data/local/`，即可将本系统扩展至你的车队 / 品牌场景。

---

## 📁 项目结构

```text
.
├── server.py              FastAPI Web 服务器 + REST API + SSE 流
├── run.py                 CLI 流水线（输出 → run.log）
├── config.py              可调管理配置
├── database.py            PostgreSQL 对话持久化
├── Dockerfile             Docker 镜像
├── docker-compose.yml     Docker Compose（应用 + PostgreSQL），默认 PORT=7500
├── .env                   本地环境变量（API keys + PORT）
├── rha_rag/                核心包
│   ├── llm.py             ChatDeepSeekFixed + 模型配置
│   ├── pipeline.py        加载器、OCR（含 .ocr.md 缓存）、embeddings、Milvus 存储
│   ├── graph.py           LangGraph 节点 + 装配
│   └── viz.py             ChartSpec Pydantic 模型 + visualize（工具调用）节点
├── prompts/               LLM 提示词模板（运行时加载）
│   ├── clarify.txt  generate.txt  grade.txt  reason.txt  verify.txt  visualize.txt
├── templates/
│   └── index.html         Web UI（深色主题、ECharts、流式）
├── data/
│   ├── local/             放入汽车文档的目录
│   ├── uploads/           通过 Web UI 上传的目录
│   └── auto-seed/         某汽车生产商A 种子语料（启动时自动加载）
├── tests/                 pytest 单元测试（43 个用例，无需 LLM）
├── test/ground_truth.json 评测问题集（5 道数学 + 9 道 某汽车生产商A）
├── eval.py                LLM-judge 评测脚本
├── requirements.txt
├── .env.example
└── ARCHITECTURE.md        详细的逐文件搭建报告
```

---

## 🔌 API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/api/status` | 系统状态（`ready`、`documents`、`chunks`、`missing_keys`、`errors`） |
| `GET` | `/api/files` | 列出 `uploads/` 与 `data/local/` 下所有文件 |
| `POST` | `/api/upload` | 上传文档（multipart form） |
| `DELETE` | `/api/files/{name}` | 从 `uploads/` 或 `data/local/` 删除文件 |
| `POST` | `/api/reindex` | 强制重建文档索引 |
| `POST` | `/api/chat` | 提问 → 返回节点级 SSE 流（多轮；需传 `session_id`） |
| `POST` | `/api/clear` | 清空某会话的历史对话 |

`/api/chat` 接收 `{"question": "...", "session_id": "...", "fast": true}`，以 `data: {"node": "...", "content": "...", "done": false}` 形式流式推送事件，以 `{"node": "done", "done": true}` 收尾。`visualize` 事件的 `content` 为 JSON 编码的 `ChartSpec`（服务端校验）。`session_id` 是对话记忆的键（参见[对话记忆](#-对话记忆)）。流水线未就绪时返回 `503` 并附 `details`。

---

## 💻 命令行用法

```bash
python run.py "某汽车生产商A 2024 年 BEV 销量是多少？"
# 或交互模式：
python run.py
# 或通过管道：
echo "2024 年中国 NEV 市场份额某汽车生产商A占多少？" | python run.py
```

输出同时写入 stdout 与 `run.log`。流水线与 Web 服务器一致——不开 server 就能调试改动。

---

## 📝 设计说明

### ChartSpec 传输

图表走现有的 SSE 通道——**协议未变**。`visualize` 节点产出一条 `HumanMessage`，其 `.content` 即 JSON spec。前端解析后构建 ECharts option（bar / line / pie / scatter / table），渲染为全宽图表卡片。PNG 导出使用 `chart.getDataURL({ type: 'png', pixelRatio: 2, backgroundColor: '#0d1117' })`——**完全离线，无需服务端往返**。

### DeepSeek V4 补丁

`ChatDeepSeekFixed` 修补了 DeepSeek V4 思考模式的 3 处不兼容：

1. **`reasoning_content` 保留** —— 工具调用往返时必须保留，LangChain 默认会清掉。
2. **List content 序列化** —— 带 list content 的 tool / assistant 消息必须被字符串化。
3. **`tool_choice` 降级** —— 思考模式拒收 `{"type":"function",...}`；强制为 `"auto"`。

两个 LLM 客户端均设 `max_retries=5`，因此上游瞬时断连（"Server disconnected without sending a response"）会自动重试，不会直接 fail。详见 [langchain-ai/langchain#37178](https://github.com/langchain-ai/langchain/issues/37178)。

### 为什么用 Milvus Lite 而非 `langchain-milvus`？

项目在 `rha_rag/pipeline.py` 中以小型 `MilvusClient` 封装（`MilvusLiteStore`）直接读写本地 `milvus.db` 文件——**无需独立 server**。`langchain-milvus` 0.3.3 与 `pymilvus` 2.6.x 不兼容（其 ORM `Collection` 路径无法解析 `MilvusClient` 注册的连接别名）。该封装直连 `MilvusClient`，绕开此问题。完整论证见 [ARCHITECTURE.md §7](ARCHITECTURE.md)。

---

## 📜 许可证

MIT —— 见 [LICENSE](LICENSE)。
