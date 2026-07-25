# AI Agent 技术栈参考

## 一、大模型调用层（地基）

| 技术 | 说明 | 本项目对应 |
|------|------|-----------|
| LLM API 调用 | 发 prompt、收回答（OpenAI 兼容格式） | DeepSeek v4-flash |
| 流式输出（Streaming） | 逐 token 返回，消除等待感 | `stream=True` |
| 结构化输出（JSON Mode） | 强制 LLM 返回 JSON，用于解析 | 记忆压缩时用 |
| Function Calling / Tool Use | LLM 自主决定调用哪个工具、传什么参数 | 未用 |
| Thinking Mode（思考模式） | LLM 先内部推理再输出答案 | DeepSeek 支持 |

### Function Calling 核心概念

```
用户提问 → LLM 判断需要查资料 → 返回工具调用指令：
{
  "tool": "search_novel",
  "arguments": {"query": "楚子航的父亲", "chapter": 0}
}
→ 你的代码执行工具 → 把结果喂回 LLM → LLM 基于结果生成回答
```

**与普通聊天的区别：** Agent 自己决定"什么时候用什么工具"，而不是固定管线。

---

## 二、RAG（检索增强生成）

### 完整管线

```
原始文档 → 切块 → 向量化 → 存入向量库
用户提问 → 向量化 → 检索 Top-K → (Rerank) → 拼 prompt → LLM 生成
```

### 各环节技术选型

| 环节 | 常见选型 | 本项目 |
|------|---------|--------|
| 切块策略 | 固定窗口 / 段落聚合 / 语义切块 | 段落聚合（500字+100重叠） |
| Embedding 模型 | BGE / text2vec / OpenAI ada | BGE-base-zh-v1.5（本地） |
| 向量存储 | ChromaDB / FAISS / Milvus / numpy | numpy 暴力余弦 |
| 关键词检索 | BM25 / Elasticsearch | rank_bm25 + jieba 自定义词典 |
| 融合策略 | RRF / 加权平均 | RRF（k=60） |
| Rerank | bge-reranker / Cohere / LLM 打分 | bge-reranker-v2-m3（本地） |
| 指代消解 | 规则 / LLM | 规则 + 记忆锚点 |

### 切块策略对比

| 策略 | 做法 | 优势 | 劣势 |
|------|------|------|------|
| 固定窗口 | 每 N 字硬切 | 简单、可预测 | 可能切断句子 |
| 段落聚合 | 按段落累积到目标字数 | 语义完整 | 块大小不均匀 |
| 语义切块 | 用 embedding 相似度检测语义边界 | 最精准 | 计算成本高 |
| 递归切块 | 按层级分隔符递归（章→段→句） | LangChain 默认 | 通用但不专精 |

---

## 三、记忆系统（Memory）

### 分层架构

| 层 | 类比 | 内容 | 生命周期 |
|---|---|---|---|
| 工作记忆 | CPU 寄存器 | 当前 context window | 单次推理 |
| 短期记忆 | RAM | 最近 N 轮对话原文 | 会话内 |
| 中期记忆 | 缓存 | 摘要 + 锚点 | 会话内（压缩后） |
| 长期记忆 | 硬盘 | 持久化文件 / 向量库 | 跨会话 |

### 主流框架对比

| 框架 | 核心理念 | 记忆管理方式 | 适合场景 |
|------|---------|-------------|---------|
| Letta/MemGPT | OS 虚拟内存 | Agent 自主读写（function call） | 长期个人助手 |
| Mem0 | 通用中间件 | 系统自动 Write→Manage→Read | 多用户 SaaS |
| Zep/Graphiti | 时序知识图谱 | 自动提取实体关系+时间标注 | 需要时序推理 |
| LangMem | LangGraph 原生 | 与图工作流集成 | LangGraph 项目 |

### 记忆操作（Mem0 范式）

- **ADD**：新事实写入
- **UPDATE**：旧事实被修正
- **DELETE**：被否定的事实删除
- **MERGE**：多条相似记忆合并

### 本项目记忆设计

```
锚点区（结构化关键信息）
  ├── discussed_characters: 讨论过的角色
  ├── user_focus: 用户关注点
  ├── consensus: 已达成的共识
  └── open_questions: 未解决的疑问

里程碑（每20轮，不可压缩）
摘要区（早期对话的压缩）
首轮钉住（第一轮对话原文永保留）
原始区（最近40轮完整对话）
```

---

## 四、编排/框架层（Orchestration）

| 框架 | 定位 | 核心概念 | 适合 |
|------|------|---------|------|
| LangGraph | 有状态工作流 | 节点+边+条件分支+状态 | 复杂多步 Agent |
| LangChain | 链式调用 | Chain（prompt→LLM→parse） | 简单管线 |
| CrewAI | 多 Agent 协作 | 角色分工+任务委派 | 多角色场景 |
| AutoGen | 对话式协作 | Agent 间对话 | 研究/实验 |
| 手写 | 完全控制 | 自定义循环/状态机 | 小规模、学习 |

### LangGraph 核心概念

```python
from langgraph.graph import StateGraph

graph = StateGraph(State)
graph.add_node("retrieve", retrieve_node)    # 检索
graph.add_node("generate", generate_node)    # 生成
graph.add_node("reflect", reflect_node)      # 反思

graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", should_continue, {
    "reflect": "reflect",
    "end": END
})
graph.add_edge("reflect", "retrieve")  # 不满意则重新检索
```

---

## 五、Agent 推理模式

| 模式 | 流程 | 适用场景 |
|------|------|---------|
| ReAct | 思考→行动→观察→循环 | 需要多步工具调用 |
| CoT（思维链） | 分步推理→最终答案 | 复杂逻辑/数学 |
| Plan-and-Solve | 先列计划→逐步执行 | 多步任务规划 |
| Reflection | 生成→自检→修正 | 防幻觉、提质 |
| Self-RAG | 检索→判断够不够→不够再检索 | 自适应检索 |

### ReAct 示例

```
用户：楚子航的爸爸为什么要把箱子交给奥丁？

Thought: 这个问题需要找到雨夜那段的原文
Action: search_novel("箱子 奥丁 交给")
Observation: [检索到序幕第15块：男人把手提箱扔向奥丁...]

Thought: 找到了，但还需要理解动机
Action: search_novel("楚子航父亲 保护 儿子")
Observation: [检索到序幕第18块：以前你很多次都不听话...]

Thought: 综合两段，父亲是用箱子做交易换取儿子安全
Final Answer: 他是在用箱子里的东西和奥丁做交易，换取楚子航的安全...
```

---

## 六、工程支撑

| 技术 | 说明 | 工具 |
|------|------|------|
| Prompt Engineering | 系统提示词设计、角色定义 | 手写 |
| Guardrails | 输入输出过滤、防幻觉 | NeMo Guardrails / 自定义 |
| Evaluation | 检索质量评估 | RAGAS / 自建评估集 |
| Observability | 调用链追踪、调试 | LangSmith / 日志 |
| Structured Output | 解析 LLM 输出为结构化数据 | Pydantic / JSON Mode |
| Token 管理 | 上下文窗口预算分配 | tiktoken / 手动计算 |

---

## 七、本项目技术栈总结

```
已实现：
  ✓ LLM API（DeepSeek v4-flash，流式）
  ✓ RAG 全链路（切块→BGE→BM25→RRF→Rerank→生成）
  ✓ 记忆系统（锚点+摘要+里程碑+原始区+持久化）
  ✓ 指代消解（规则+记忆锚点）
  ✓ 手写编排（固定管线：检索→生成）
  ✓ 工程化（日志、错误处理、原子写入、Go 启动器）

未实现（后续方向）：
  ✗ Function Calling（让 AI 自主决定何时检索）
  ✗ LangGraph 编排（条件分支、多步推理）
  ✗ ReAct 循环（思考→行动→观察）
  ✗ 评估体系（Recall@K、MRR）
  ✗ 实体感知检索（Wiki 层产出辅助）
  ✗ Reflection（答后自检）
```
