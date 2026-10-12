# Swarm 代码原理详解

> 本文档从源码角度拆解 Swarm 的实现原理。Swarm 是一个**轻量、无状态、可测试**的多智能体（multi-agent）编排框架，通过 `Agent` 和 **handoff（移交）** 两个原语表达智能体之间的协作。

---

## 目录

- [1. 设计哲学](#1-设计哲学)
- [2. 整体架构](#2-整体架构)
- [3. 核心数据模型（types.py）](#3-核心数据模型typespy)
- [4. 核心执行引擎（core.py）](#4-核心执行引擎corepy)
  - [4.1 run：主循环](#41-run主循环)
  - [4.2 get_chat_completion：构造请求](#42-get_chat_completion构造请求)
  - [4.3 handle_tool_calls：执行工具](#43-handle_tool_calls执行工具)
  - [4.4 handle_function_result：归一化返回值](#44-handle_function_result归一化返回值)
  - [4.5 run_and_stream：流式执行](#45-run_and_stream流式执行)
- [5. 工具函数（util.py）](#5-工具函数utilpy)
- [6. 交互式 REPL（repl/repl.py）](#6-交互式-replreplreplpy)
- [7. 关键机制专题](#7-关键机制专题)
- [8. 执行时序图](#8-执行时序图)
- [9. 示例解读](#9-示例解读)
- [10. 边界与已知局限](#10-边界与已知局限)

---

## 1. 设计哲学

Swarm 的目标是让智能体的**协调（coordination）**与**执行（execution）**保持轻量、高度可控、易于测试。它凭两个原语实现：

| 原语 | 含义 |
| --- | --- |
| `Agent` | 封装一组 `instructions`（指令）与 `functions`（工具），可随时把对话移交给另一个 `Agent` |
| **handoff（移交）** | 一个工具函数**直接返回另一个 `Agent`**，框架据此切换当前活跃智能体 |

三条核心原则：

1. **无状态（stateless）**：完全基于 Chat Completions API，客户端运行，调用之间不保存任何状态。会话历史由调用方自行传入传出。
2. **客户端编排**：`run()` 在本地循环，处理工具调用、移交与上下文变量，而不是依赖服务端线程。
3. **约定优于配置**：函数签名、类型注解、docstring 会被自动转换为模型可用的 JSON Schema；`context_variables` 参数会被自动注入并从 schema 中隐藏。

> 说明：Swarm 与 Assistants API 中的 "Assistants" **没有关系**，命名相似仅为方便。该项目定位为教学/实验性质，官方已推荐生产场景迁移到 OpenAI Agents SDK。

---

## 2. 整体架构

```
swarm/
├── swarm/                    # 核心包（全部逻辑，约 400 行）
│   ├── core.py               # Swarm 类：run 循环、工具调用、handoff（心脏）
│   ├── types.py              # Pydantic 数据模型：Agent / Response / Result
│   ├── util.py               # 工具函数：schema 生成、流式合并、调试打印
│   ├── __init__.py           # 对外导出 Swarm / Agent / Response
│   └── repl/repl.py          # 命令行交互式 demo loop
├── examples/                 # 官方示例（basic / triage_agent / airline / ...）
├── tests/                    # pytest + MockOpenAIClient（无需真实 API Key）
├── assets/                   # logo 与架构图
├── setup.cfg                 # 包元数据与依赖
└── pyproject.toml            # build-system 声明
```

**分层关系**：

```
用户代码
   │  client.run(agent, messages, ...)
   ▼
Swarm (core.py) ──调用──▶ OpenAI Client (chat.completions.create)
   │  依赖
   ├── types.py    （Agent / Response / Result 数据结构）
   └── util.py     （function_to_json / merge_chunk / debug_print）
```

---

## 3. 核心数据模型（types.py）

三个 Pydantic `BaseModel`，是贯穿全流程的数据契约。

### 3.1 Agent

```python
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
```

| 字段 | 作用 |
| --- | --- |
| `name` | 智能体名称，用于日志与「消息来自谁」标记 |
| `model` | 该智能体使用的模型，可被 `model_override` 覆盖 |
| `instructions` | **字符串**或**函数**。字符串直接作为 system prompt；函数可接收 `context_variables` 动态生成指令 |
| `functions` | 智能体可调用的 Python 函数列表 |
| `tool_choice` | 传给模型的工具选择策略 |
| `parallel_tool_calls` | 是否允许模型并行调用多个工具 |

### 3.2 Response

`run()` 的返回值，代表一次（可能多轮）执行的最终状态：

```python
class Response(BaseModel):
    messages: List = []          # 本次新增的消息（含 tool 消息）
    agent: Optional[Agent] = None # 最后处理消息的 Agent
    context_variables: dict = {}  # 更新后的上下文变量
```

继续对话时，把 `messages` 与新用户消息拼接后再次传入 `run()` 即可——这正是「无状态」的体现。

### 3.3 Result

工具函数的统一返回载体，允许同时携带值、目标智能体和上下文变量更新：

```python
class Result(BaseModel):
    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
```

一个函数可以返回三者中的任意子集（例如只更新上下文变量、或只做 handoff）。

---

## 4. 核心执行引擎（core.py）

`Swarm` 类是整个框架的心脏。其构造函数允许注入自定义 client，这正是**可测试性**的来源：

```python
class Swarm:
    def __init__(self, client=None):
        if not client:
            client = OpenAI()
        self.client = client
```

### 4.1 run：主循环

`run()` 是整个框架最核心的方法，其本质是一个**「请求 → 执行工具 → 可能移交 → 再来一轮」**的 while 循环：

```
active_agent = agent
context_variables = copy.deepcopy(context_variables)   # 深拷贝，避免污染调用方
history = copy.deepcopy(messages)
init_len = len(messages)                               # 记录起始长度

while len(history) - init_len < max_turns and active_agent:
    completion = get_chat_completion(...)              # 1. 请求模型
    message = completion.choices[0].message
    message.sender = active_agent.name
    history.append(json.loads(message.model_dump_json()))  # 转普通 dict 入库

    if not message.tool_calls or not execute_tools:    # 2. 无工具调用 → 结束
        break

    partial_response = handle_tool_calls(...)          # 3. 执行工具
    history.extend(partial_response.messages)          #    追加 tool 结果消息
    context_variables.update(partial_response.context_variables)  # 4. 更新上下文
    if partial_response.agent:                         # 5. 若发生 handoff 则切换
        active_agent = partial_response.agent

return Response(messages=history[init_len:], agent=active_agent,
                context_variables=context_variables)
```

**要点解读**：

- **深拷贝**：`context_variables` 与 `messages` 都做 `copy.deepcopy`，保证函数内部修改不会反向影响调用方，也保证「无状态」语义干净。
- **`init_len` 切片**：只返回本次新增的消息（`history[init_len:]`），历史部分由调用方负责保存与回传。
- **`max_turns` 语义**：以「本轮新增消息数」计，防止无限循环。
- **`execute_tools=False`**：即使模型请求了工具，也立即中断并返回——常用于人工审批（human-in-the-loop）或分步执行。
- **消息转 dict**：`json.loads(message.model_dump_json())` 把 OpenAI SDK 类型转成普通 dict，避免后续序列化问题。

### 4.2 get_chat_completion：构造请求

负责把 `Agent` 翻译成一次 Chat Completions 请求，并处理三个关键细节：

```python
context_variables = defaultdict(str, context_variables)   # 缺失的变量默认空串
instructions = (
    agent.instructions(context_variables)                  # 函数式指令
    if callable(agent.instructions) else agent.instructions
)
messages = [{"role": "system", "content": instructions}] + history

tools = [function_to_json(f) for f in agent.functions]
# 关键：把 context_variables 从模型可见的 schema 中隐藏
for tool in tools:
    params = tool["function"]["parameters"]
    params["properties"].pop("context_variables", None)
    if "context_variables" in params["required"]:
        params["required"].remove("context_variables")

create_params = {
    "model": model_override or agent.model,
    "messages": messages,
    "tools": tools or None,
    "tool_choice": agent.tool_choice,
    "stream": stream,
}
if tools:
    create_params["parallel_tool_calls"] = agent.parallel_tool_calls

return self.client.chat.completions.create(**create_params)
```

| 细节 | 说明 |
| --- | --- |
| `defaultdict(str, ...)` | 让 `context_variables["不存在的键"]` 返回 `""` 而非 KeyError，指令函数更健壮 |
| 动态 system prompt | `instructions` 为函数时以 `context_variables` 为入参实时生成 |
| 隐藏 `context_variables` | 该参数由**框架注入**，属于实现细节，不应暴露给模型 |
| `tools or None` | 无工具时传 `None`，而不是空列表 |

### 4.3 handle_tool_calls：执行工具

逐个执行模型请求的工具调用：

```python
function_map = {f.__name__: f for f in functions}

for tool_call in tool_calls:
    name = tool_call.function.name
    if name not in function_map:                 # 1. 工具不存在 → 记录错误，继续
        partial_response.messages.append({
            "role": "tool", "tool_call_id": tool_call.id,
            "tool_name": name, "content": f"Error: Tool {name} not found.",
        })
        continue
    args = json.loads(tool_call.function.arguments)

    func = function_map[name]
    if "context_variables" in func.__code__.co_varnames:   # 2. 按签名注入上下文
        args["context_variables"] = context_variables
    raw_result = function_map[name](**args)

    result = self.handle_function_result(raw_result, debug) # 3. 归一化返回值
    partial_response.messages.append({
        "role": "tool", "tool_call_id": tool_call.id,
        "tool_name": name, "content": result.value,
    })
    partial_response.context_variables.update(result.context_variables)  # 4. 累积上下文
    if result.agent:                             # 5. 记录 handoff 目标
        partial_response.agent = result.agent
```

**要点解读**：

- **工具名 → 函数**：用 `__name__` 建映射，因此函数名即工具名。
- **上下文的「约定式」注入**：用 `func.__code__.co_varnames` 检查函数是否声明了 `context_variables` 形参，有则注入——无需显式装饰器或标记。
- **未知工具优雅降级**：不抛异常，而是把错误信息作为 tool 消息回灌给模型，让模型有机会自我纠正。
- **多次 handoff 只取最后一个**：循环内 `partial_response.agent` 会被后一次覆盖（官方文档明确此行为）。
- **错误隔离**：单个工具出错不影响其他工具的执行顺序。

### 4.4 handle_function_result：归一化返回值

用 Python 3.10 的结构模式匹配（`match/case`）把工具的任意返回值统一成 `Result`：

```python
def handle_function_result(self, result, debug) -> Result:
    match result:
        case Result() as result:          # 已是 Result → 原样返回
            return result
        case Agent() as agent:            # 返回 Agent → 视为 handoff
            return Result(
                value=json.dumps({"assistant": agent.name}),
                agent=agent,
            )
        case _:                           # 其他 → 转字符串
            try:
                return Result(value=str(result))
            except Exception as e:
                raise TypeError(...)
```

**这就是 handoff 的实现原理**：工具函数返回一个 `Agent` 实例，框架自动识别并把结果值序列化为 `{"assistant": "<名称>"}`，同时把该 `Agent` 设为新的活跃智能体。

### 4.5 run_and_stream：流式执行

当 `run(stream=True)` 时，`run()` 会委托给 `run_and_stream()`，它是一个**生成器**，边接收模型流式输出边增量拼装：

```python
message = {
    "content": "", "sender": agent.name, "role": "assistant",
    "function_call": None,
    "tool_calls": defaultdict(lambda: {"function": {"arguments": "", "name": ""},
                                       "id": "", "type": ""}),
}
completion = self.get_chat_completion(..., stream=True, ...)

yield {"delim": "start"}
for chunk in completion:
    delta = json.loads(chunk.choices[0].delta.json())
    if delta["role"] == "assistant":
        delta["sender"] = active_agent.name
    yield delta
    delta.pop("role", None); delta.pop("sender", None)
    merge_chunk(message, delta)          # 增量合并到完整消息
yield {"delim": "end"}

message["tool_calls"] = list(message.get("tool_calls", {}).values())
...
```

流式模式在标准 Chat Completions 流事件之外**新增两类自定义事件**：

| 事件 | 含义 |
| --- | --- |
| `{"delim": "start"}` / `{"delim": "end"}` | 标记一个 Agent 处理单条消息的开始/结束，便于识别 Agent 切换 |
| `{"response": Response}` | 在流的末尾返回聚合完成的完整 `Response`，方便下游统一处理 |

其余流程（工具执行、handoff、上下文更新）与非流式一致。

---

## 5. 工具函数（util.py）

### 5.1 function_to_json：函数 → JSON Schema

把 Python 函数自动转成模型可调用的工具描述：

```python
def function_to_json(func) -> dict:
    type_map = {str: "string", int: "integer", float: "number",
                bool: "boolean", list: "array", dict: "object", type(None): "null"}
    signature = inspect.signature(func)

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")  # 未知类型默认 string
        parameters[param.name] = {"type": param_type}

    required = [p.name for p in signature.parameters.values()
                if p.default == inspect._empty]                # 无默认值即必填

    return {"type": "function", "function": {
        "name": func.__name__,
        "description": func.__doc__ or "",
        "parameters": {"type": "object", "properties": parameters,
                       "required": required}}}
```

映射规则：

- **docstring → `description`**
- **类型注解 → JSON 类型**，未标注则默认 `string`
- **无默认值的参数 → `required`**
- 暂不支持逐参数描述（可写进 docstring，未来可能解析）

### 5.2 merge_chunk / merge_fields：流式增量合并

```python
def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value                    # 字符串拼接（内容分片）
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)        # 递归合并嵌套字典

def merge_chunk(final_response, delta):
    delta.pop("role", None)
    merge_fields(final_response, delta)
    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")          # 按 index 归位到对应工具调用
        merge_fields(final_response["tool_calls"][index], tool_calls[0])
```

流式 chunk 是**增量分片**：文本按字符流式返回，工具调用的 arguments 也是分片拼接，`index` 字段用于区分并发的多个工具调用。此函数负责把它们重新拼成完整消息。

### 5.3 debug_print

按需输出带时间戳的灰色调试日志，仅在 `debug=True` 时生效。

---

## 6. 交互式 REPL（repl/repl.py）

`run_demo_loop(starting_agent, ...)` 提供一个命令行循环，演示「无状态框架如何维持连续对话」：

```python
messages = []
agent = starting_agent
while True:
    user_input = input("User: ")
    messages.append({"role": "user", "content": user_input})
    response = client.run(agent=agent, messages=messages, ...)
    if stream:
        response = process_and_print_streaming_response(response)
    else:
        pretty_print_messages(response.messages)
    messages.extend(response.messages)   # 关键：把新增消息拼回本地历史
    agent = response.agent               # 关键：跟踪最后活跃的 Agent
```

**这正是「无状态」的运行范式**：框架自己不保存任何东西，`messages` 与 `agent` 都由调用方在循环外维护并逐轮回传。

---

## 7. 关键机制专题

### 7.1 Handoff（移交）的完整链路

```
工具函数 return agent_b
        │
        ▼
handle_function_result 识别 Agent 实例
        │  → Result(value='{"assistant": "Agent B"}', agent=agent_b)
        ▼
handle_tool_calls 把 tool 消息入队，并设置 partial_response.agent = agent_b
        │
        ▼
run 主循环：active_agent = partial_response.agent
        │
        ▼
下一轮 get_chat_completion 用 agent_b.instructions 作为新的 system prompt
        │
        ▼
Response.agent == agent_b
```

**注意**：切换的只是 `system` prompt，**聊天历史保持不变**。一次模型回复中若请求了多个 handoff，只有**最后一个**生效。

### 7.2 context_variables 的注入与隐藏

| 环节 | 行为 |
| --- | --- |
| 请求模型时 | 从工具 schema 的 `properties` / `required` 中**移除** `context_variables`（模型不可见） |
| 执行工具时 | 若函数签名含 `context_variables` 形参，**自动注入**当前上下文 |
| 指令生成时 | 若 `instructions` 是可调用对象，以 `context_variables` 为入参调用 |
| 结果回传时 | 函数返回的 `Result.context_variables` 会 **update** 进全局上下文 |

**循环内的调用顺序也需要留意**：`get_chat_completion` 里用的是**进入循环那一刻**的 `context_variables`；工具执行后更新的上下文会在**下一轮**请求时生效。

### 7.3 无状态设计

框架层面**不持有任何会话状态**。连续性完全依赖调用方每次把完整的 `messages`（含历史）与最新 `agent` 传回来。好处：

- 易于部署在无服务器/多实例环境；
- 状态归属清晰，便于持久化到任意存储；
- 与 Chat Completions API 心智模型一致。

### 7.4 三种「中断/控制」手段

| 手段 | 作用 |
| --- | --- |
| `max_turns` | 限制循环轮数，防死循环 |
| `execute_tools=False` | 一遇到工具调用就停下并返回，便于人工审批/分步执行 |
| `model_override` | 临时覆盖所有 Agent 的模型（便于测试或降本） |

---

## 8. 执行时序图

以「用户提问 → 模型请求工具 → 执行工具 → 返回结果」为例：

```mermaid
sequenceDiagram
    participant U as 调用方
    participant S as Swarm.run
    participant M as OpenAI Client
    participant F as 工具函数

    U->>S: run(agent, messages, context_variables)
    S->>S: 深拷贝 messages / context_variables
    loop while 未超过 max_turns
        S->>M: chat.completions.create(system+messages, tools)
        M-->>S: assistant 消息（含 tool_calls）
        S->>S: message.sender = agent.name; 追加进 history
        alt 有 tool_calls 且 execute_tools=True
            loop 每个 tool_call
                S->>F: 注入 context_variables 后调用
                F-->>S: str / Result / Agent
                S->>S: handle_function_result 归一化
                S->>S: 追加 tool 消息；更新上下文；记录 handoff
            end
            opt 返回了 Agent
                S->>S: active_agent = 新 Agent
            end
        else 无工具调用
            S->>S: break
        end
    end
    S-->>U: Response(messages, agent, context_variables)
```

---

## 9. 示例解读

| 示例 | 文件 | 演示的原理 |
| --- | --- | --- |
| 最小可用 | `examples/basic/bare_minimum.py` | 单 Agent 一次 `run` |
| 函数调用 | `examples/basic/function_calling.py` | 工具 schema 自动生成与执行 |
| Handoff | `examples/basic/agent_handoff.py` | 工具返回 `Agent` 触发移交 |
| 上下文变量 | `examples/basic/context_variables.py` | 函数式 `instructions` + 自动注入 |
| 手工循环 | `examples/basic/simple_loop_no_helpers.py` | 不用 REPL，手写无状态对话循环 |

**Handoff 示例核心**（`agent_handoff.py`）：

```python
spanish_agent = Agent(name="Spanish Agent", instructions="You only speak Spanish.")

def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent          # ← 返回 Agent 即触发 handoff

english_agent.functions.append(transfer_to_spanish_agent)
response = client.run(agent=english_agent, messages=[{"role": "user", "content": "Hola..."}])
```

**上下文变量示例核心**（`context_variables.py`）：

```python
def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."

def print_account_details(context_variables: dict):
    print(f"Account Details: {context_variables.get('name')} {context_variables.get('user_id')}")
    return "Success"

response = client.run(agent=agent, messages=messages,
                      context_variables={"name": "James", "user_id": 123})
```

其他较复杂的多智能体场景见 `examples/airline`、`support_bot`、`personal_shopper`、`triage_agent`。

---

## 10. 边界与已知局限

**测试性**：`Swarm(client=...)` 支持注入 `MockOpenAIClient`，配合 `set_sequential_responses` 可完全离线测试（见 `tests/test_core.py`、`tests/mock_client.py`）。已覆盖：简单消息、工具调用、`execute_tools=False`、handoff。

**局限**：

| 局限 | 说明 |
| --- | --- |
| 官方已停止演进 | 已被 OpenAI Agents SDK 取代，建议生产迁移 |
| 无内建持久化 | 无长期记忆，历史需自行管理 |
| 无内建追踪/日志 | 仅有 `debug_print`，缺乏生产级 tracing |
| 无护栏（guardrails） | 无输入/输出安全校验机制 |
| 仅同步调用 | 同步阻塞式执行，未提供异步接口 |
| 参数级描述缺失 | 工具参数的逐项描述暂不支持自动解析 |
| 单机客户端编排 | 循环在客户端，长会话需自行控制上下文长度 |

**工程信息**：

- Python 3.10+（依赖 `match/case` 结构模式匹配）。
- 主要依赖：`openai>=1.33.0`；元数据定义在 `setup.cfg`，`pyproject.toml` 仅声明 build-backend。

---

## 附：一句话总结

> Swarm 用一个「请求模型 → 执行工具 → 可能移交 → 循环」的无状态 while 循环（`core.py: run`），配合三个 Pydantic 模型（`Agent` / `Response` / `Result`）和一套「函数签名 ↔ JSON Schema」的自动转换，就用几百行代码实现了多智能体协作与移交的全部精髓。
