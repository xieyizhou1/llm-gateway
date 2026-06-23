# LLM Gateway Anthropic 协议透传与 OpenAI↔Anthropic 双向转换需求文档

> 状态：待实现  
> 创建时间：2026-06-23  
> 背景：Kimi for Coding 仅开放 `/v1/messages`（Anthropic Messages API）访问，OpenAI `/v1/chat/completions` 端点已返回 `access_terminated_error`。

---

## 1. 现状分析

### 1.1 测试结论

使用有效 Kimi API Key 对两个域名、三种协议进行全面测试，结果如下：

| 域名 | 端点 | 协议 | 模型 | HTTP 状态 | 结果 |
|------|------|------|------|-----------|------|
| `api.kimi.com/coding/v1` | `/v1/messages` | Anthropic Messages API | `kimi-for-coding` | 200 | ✅ 可用 |
| `api.kimi.com/coding/v1` | `/v1/messages` | Anthropic Messages API | `kimi-k2.7-code` | 200 | ✅ 可用 |
| `api.kimi.com/coding/v1` | `/v1/chat/completions` | OpenAI Chat Completions API | `kimi-for-coding` | 403 | ❌ `access_terminated_error` |
| `api.kimi.com/coding/v1` | `/v1/chat/completions` | OpenAI Chat Completions API | `kimi-k2.7-code` | 403 | ❌ `access_terminated_error` |
| `api.kimi.com/coding/v1` | `/v1/responses` | OpenAI Responses API | `kimi-for-coding` | 404 | ❌ 不支持 |
| `api.kimi.com/coding/v1` | `/v1/models` | - | - | 200 | ✅ 可用 |
| `api.moonshot.cn` | `/v1/chat/completions` | OpenAI Chat Completions API | `kimi-k2.7-code` | 401 | ❌ Key 不匹配 |
| `api.moonshot.cn` | `/anthropic/v1/messages` | Anthropic Messages API | `kimi-k2.7-code` | 401 | ❌ Key 不匹配 |
| `api.moonshot.cn` | `/v1/responses` | OpenAI Responses API | `kimi-k2.7-code` | 404 | ❌ 不支持 |

### 1.2 关键结论

1. **当前 Key 仅适用于 `api.kimi.com/coding/v1` 域名**，不适用于 `api.moonshot.cn`。
2. **Kimi for Coding 仅支持 Anthropic Messages API 协议**（`/v1/messages`）。
3. **OpenAI Chat Completions API 协议已被 Kimi 终止**（`/v1/chat/completions` 返回 `access_terminated_error`）。
4. **OpenAI Responses API 协议 Kimi 完全不支持**（返回 404）。
5. `kimi-for-coding` 与 `kimi-k2.7-code` 在 Anthropic 端点下均可调用。

---

## 2. 业务目标

在 **不破坏现有 OpenAI 兼容入口** 的前提下，让 `llm-gateway` 能够继续通过 Kimi for Coding 提供服务：

- **目标 1**：Gateway 新增 `/v1/messages` 入口，支持 Anthropic Messages API 协议。
- **目标 2**：当客户端走 `/v1/chat/completions`（OpenAI 协议）且目标 Provider 为 Kimi 时，Gateway 自动将 OpenAI 请求转换为 Anthropic 请求，调用 Kimi `/v1/messages`，再将 Anthropic 响应转换回 OpenAI 响应返回。
- **目标 3**：保留现有 DeepSeek 等 OpenAI 兼容 Provider 的路由能力。
- **目标 4**：支持流式（SSE）与非流式两种模式。
- **目标 5**：支持 tools/tool_calls 与 reasoning/thinking 字段映射。

---

## 3. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端                                   │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │ Claude Code / Anthropic │      │ 其他服务 / OpenAI 兼容     │   │
│  │  POST /v1/messages      │      │  POST /v1/chat/completions │   │
│  └──────────┬──────────┘      └──────────────┬──────────────┘   │
└─────────────┼────────────────────────────────┼──────────────────┘
              │                                │
              ▼                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      llm-gateway                                │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │ Anthropic Handler   │      │ OpenAI Handler              │   │
│  │ - 鉴权/限流/路由     │      │ - 鉴权/限流/路由             │   │
│  │ - 直接透传 upstream │      │ - OpenAI→Anthropic 转换      │   │
│  │                     │      │ - Anthropic→OpenAI 转换      │   │
│  └──────────┬──────────┘      └──────────────┬──────────────┘   │
└─────────────┼────────────────────────────────┼──────────────────┘
              │                                │
              │    Anthropic /v1/messages      │
              └────────────────┬───────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  https://api.kimi.com   │
                  │     /coding/v1/messages │
                  │     (Kimi for Coding)   │
                  └─────────────────────────┘
```

---

## 4. 功能需求

### 4.1 Provider 协议配置

在 `config.yaml` 中为 Provider 增加 `protocol` 字段：

```yaml
providers:
  kimi:
    base_url: https://api.kimi.com/coding/v1
    protocol: anthropic   # 新增：可选 anthropic | openai，默认 openai
    headers:
      User-Agent: claude-code/1.0
      anthropic-version: 2023-06-01
    keys:
      - id: kimi-1
        key: ${KIMI_KEY_1}
    model_map:
      kimi-for-coding: kimi-for-coding
      claude-sonnet-4-6: kimi-for-coding
      gpt-5.5: kimi-for-coding
```

- 当 `protocol: anthropic` 时，forwarder 使用 `/messages` 路径而非 `/chat/completions`。
- 请求头需携带 `Authorization: Bearer <key>`、`Content-Type: application/json`、`anthropic-version: 2023-06-01` 以及用户自定义 headers（如 `User-Agent`）。

### 4.2 Anthropic 入口（透传）

新增 HTTP 路由：

```
POST /v1/messages
```

行为：
1. 使用 Anthropic Messages API 请求体格式解析请求。
2. 走现有鉴权、限流、router 逻辑。
3. 若目标 Provider 是 `anthropic` 协议，直接透传请求体到上游 `/v1/messages`。
4. 若目标 Provider 是 `openai` 协议，返回 400 或不支持提示。
5. 流式响应透传 SSE。

### 4.3 OpenAI 入口（转换）

保留现有 HTTP 路由：

```
POST /v1/chat/completions
```

行为：
1. 使用 OpenAI Chat Completions API 请求体格式解析请求。
2. 走现有鉴权、限流、router 逻辑。
3. 若目标 Provider 是 `anthropic` 协议：
   - 将 OpenAI 请求转换为 Anthropic 请求（复用/扩展 `OpenAIToAnthropicRequest`）。
   - 调用上游 `/v1/messages`。
   - 将 Anthropic 响应转换回 OpenAI 响应（新增 `AnthropicToOpenAIResponse`）。
4. 若目标 Provider 是 `openai` 协议，保持现有逻辑不变。

### 4.4 字段映射

#### 4.4.1 OpenAI 请求 → Anthropic 请求

| OpenAI 字段 | Anthropic 字段 | 说明 |
|-------------|----------------|------|
| `model` | `model` | 通过 `model_map` 映射 |
| `messages` | `messages` | system 消息提取到顶层 `system` |
| `system` 角色消息 | `system` 顶层字段 | 合并多个 system 消息 |
| `max_tokens` | `max_tokens` | 必填，缺失时填默认值 |
| `temperature` | `temperature` | 可选 |
| `stream` | `stream` | 布尔 |
| `tools` | `tools` | 格式转换 |
| `tool_choice` | `tool_choice` | 格式转换 |

#### 4.4.2 Anthropic 响应 → OpenAI 响应

| Anthropic 字段 | OpenAI 字段 | 说明 |
|----------------|-------------|------|
| `content[].type=text` | `choices[0].message.content` | 文本内容 |
| `content[].type=thinking` | `choices[0].message.reasoning_content` | 推理内容 |
| `content[].type=tool_use` | `choices[0].message.tool_calls` | 工具调用 |
| `stop_reason=end_turn` | `choices[0].finish_reason=stop` | 正常结束 |
| `stop_reason=tool_use` | `choices[0].finish_reason=tool_calls` | 工具调用结束 |
| `stop_reason=max_tokens` | `choices[0].finish_reason=length` | 长度限制 |
| `usage.input_tokens` | `usage.prompt_tokens` | 输入 token |
| `usage.output_tokens` | `usage.completion_tokens` | 输出 token |
| `id` | `id` | 请求 ID |
| `model` | `model` | 模型名 |

#### 4.4.3 流式 SSE 映射

Anthropic SSE 事件类型 → OpenAI SSE 事件类型：

| Anthropic | OpenAI |
|-----------|--------|
| `message_start` | 生成 `choices[0].delta={}` 初始化 |
| `content_block_start` (text) | `choices[0].delta.content=文本开始` |
| `content_block_delta` (text_delta) | `choices[0].delta.content=文本片段` |
| `content_block_delta` (thinking_delta) | `choices[0].delta.reasoning_content=推理片段` |
| `content_block_delta` (input_json_delta) | `choices[0].delta.tool_calls[].function.arguments` |
| `message_stop` | `choices[0].finish_reason=end_turn` 或 `tool_calls` |

---

## 5. 非功能需求

### 5.1 向后兼容

- `protocol` 字段默认值为 `openai`，不配置时不影响现有 DeepSeek 等 Provider。
- 现有 `/v1/chat/completions` 路由行为不变，仅当路由到 `anthropic` 协议 Provider 时才触发转换。

### 5.2 错误透传

- 上游 Anthropic 错误（如 401/429/500）应原样或带 `provider` 信息返回给客户端。
- 转换失败时返回 400 并附带具体错误信息。

### 5.3 日志与监控

- 记录请求使用的入口协议、目标 Provider 协议、转换耗时。
- Dashboard 中可区分 `/v1/messages` 与 `/v1/chat/completions` 请求量。

### 5.4 配置校验

- 启动时校验：当 Provider `protocol=anthropic` 时，`model_map` 中映射的 upstream_model 必须能在该 Provider 下调用。
- 当 `protocol=anthropic` 时，推荐配置 `anthropic-version` header，未配置时给出 warning。

---

## 6. 待修改模块

| 模块 | 改动点 |
|------|--------|
| `internal/config` | ProviderConfig 增加 `Protocol string` 字段 |
| `internal/provider` | 新增 `AnthropicCompatibleAdapter`，包含 `/messages` URL 构建与请求构造 |
| `internal/adapter` | 新增 `AnthropicToOpenAIResponse` 响应转换；完善 `OpenAIToAnthropicRequest` |
| `internal/forwarder` | 根据 Provider protocol 选择 adapter 与转换方向；支持 Anthropic 响应处理 |
| `internal/server` | 新增 `/v1/messages` 路由与 handler |
| `internal/router` | 无重大改动，沿用现有 provider/model 路由逻辑 |
| `internal/models` | 如有缺失，补充 Anthropic 响应/流式 chunk 结构体字段 |
| `config.yaml` | 更新 Kimi provider 配置为 `protocol: anthropic` |

---

## 7. 实现阶段

### 阶段一：MVP（最小可用）

1. 新增 `AnthropicCompatibleAdapter`。
2. 新增 `/v1/messages` 透传入口。
3. 实现非流式 `AnthropicToOpenAIResponse`。
4. OpenAI `/v1/chat/completions` → Kimi Anthropic 转换链路跑通。
5. 更新 `config.yaml` 并部署测试。

### 阶段二：完整能力

1. 流式 SSE 双向转换。
2. tool_calls / tool_use 映射。
3. reasoning / thinking 内容映射。
4. 错误处理与重试策略适配。
5. Dashboard 指标补充。

---

## 8. 测试计划

### 8.1 单元测试

- `AnthropicToOpenAIResponse` 各种 content 类型转换。
- `OpenAIToAnthropicRequest` system 提取、tools 转换。
- SSE chunk 转换。

### 8.2 集成测试

- 直接 curl gateway `/v1/messages` 验证透传。
- 直接 curl gateway `/v1/chat/completions` 验证转换。
- 使用 Claude Code 连接 gateway 验证端到端可用。
- 验证 DeepSeek `/v1/chat/completions` 链路未受影响。

### 8.3 线上灰度

- 先替换一个 Kimi key 灰度观察。
- 监控错误率、延迟、token 消耗。

---

## 9. 风险与注意事项

1. **Anthropic 与 OpenAI 字段语义差异**：如 `tool_calls` ID 格式、`finish_reason` 枚举不完全对齐，需充分测试。
2. **流式事件顺序**：Anthropic 的 `message_start`、`content_block_start`、`content_block_delta`、`message_stop` 顺序需正确映射为 OpenAI 的 `choices[0].delta` 序列。
3. **Kimi 策略可能再变**：需保持 adapter 可配置，以便未来快速切换回 OpenAI 协议或新增其他 Provider。
4. **Key 风控**：频繁测试可能导致 key 被回收，生产部署后注意观察 401/403 比例。

---

## 10. 配置示例（目标态）

```yaml
server:
  host: 0.0.0.0
  port: 18080

auth:
  disabled: false
  virtual_keys:
    - key: ${LG_LOCAL_VIRTUAL_KEY}
      user: default
      allowed_models:
        - kimi-for-coding
        - claude-sonnet-4-6
        - gpt-5.5
      rpm_limit: 100
      tpm_limit: 200000
      concurrency_limit: 10

providers:
  kimi:
    base_url: https://api.kimi.com/coding/v1
    protocol: anthropic
    supports_images: true
    headers:
      User-Agent: claude-code/1.0
      anthropic-version: 2023-06-01
    keys:
      - id: kimi-1
        key: ${KIMI_KEY_1}
        weight: 1
        rpm_limit: 60
        concurrency_limit: 5
    model_map:
      kimi-for-coding: kimi-for-coding
      claude-sonnet-4-6: kimi-for-coding
      gpt-5.5: kimi-for-coding

  deepseek:
    base_url: https://api.deepseek.com/v1
    protocol: openai
    supports_images: false
    keys:
      - id: deepseek-1
        key: ${DEEPSEEK_KEY_1}
        weight: 1
        rpm_limit: 60
        concurrency_limit: 10
    model_map:
      kimi-for-coding: deepseek-v4-pro
      claude-sonnet-4-6: deepseek-v4-pro
      gpt-5.5: deepseek-v4-pro

router:
  strategy: round_robin
  provider_order:
    - deepseek
    - kimi
  retry_count: 8
  timeout_seconds: 120
  cooldown_seconds: 60
  failure_threshold: 3
```

---

*文档结束*
