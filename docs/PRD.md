# LLM Gateway — 产品需求文档

> 版本：v0.2 | 日期：2026-05-28
> 上游 Provider：Kimi (Moonshot)、DeepSeek
> 下游协议：OpenAI Chat Completions + Anthropic Messages

---

## 1. 项目目标

在客户端（业务方）与 LLM 服务商之间构建中间网关层，实现：

1. **统一入口**：客户端可用 OpenAI SDK 或 Anthropic SDK 格式调用
2. **多 Key 管理**：维护 Kimi/DeepSeek API Key 池，支持动态添加/禁用
3. **负载均衡**：轮询/加权策略分发请求，避免单 Key 触发 Rate Limit
4. **限流保护**：滑动窗口限流，保障下游稳定性
5. **高可用**：健康检查 + 自动故障转移 + 指数退避重试
6. **可观测性**：结构化日志（trace_id）、Prometheus 指标

**核心洞察**：Kimi/DeepSeek API 本身就是 OpenAI-compatible 的，所以内部统一用 OpenAI 格式与上游通信。

## 2. 转换矩阵

| 客户端入口 | 请求格式 | 转发格式 | 响应转换 |
|-----------|---------|---------|---------|
| `POST /v1/chat/completions` | OpenAI | OpenAI → Kimi/DeepSeek | 几乎透传 |
| `POST /v1/messages` | Anthropic | Anthropic → OpenAI → Kimi/DeepSeek | OpenAI → Anthropic |

## 3. 架构

```
Client (OpenAI SDK)          Client (Anthropic SDK)
        │                              │
        ▼                              ▼
┌───────────────┐              ┌───────────────┐
│ /v1/chat/     │              │ /v1/messages  │
│ completions   │              │               │
└───────┬───────┘              └───────┬───────┘
        │                              │
        └──────────────┬───────────────┘
                       ▼
              ┌─────────────────┐
              │     Auth        │
              │  虚拟Key校验     │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │     Router      │
              │ 选Key+限流+健康  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │     Adapter     │
              │  Anthropic↔OpenAI│
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │    Forwarder    │
              │  httpx→Kimi/DS  │
              └────────┬────────┘
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
      ┌──────────┐          ┌──────────┐
      │   Kimi   │          │ DeepSeek │
      │   API    │          │   API    │
      └──────────┘          └──────────┘
```

## 4. 核心模块

### 4.1 Auth
- 虚拟 Key 格式：`Bearer sk-virtual-<id>`
- 虚拟 Key → 允许模型列表 + RPM 配额
- 真实 Key 绝不暴露

### 4.2 Router
- **负载策略**：轮询（默认）/ 加权轮询
- **限流**：每个 Key 独立滑动窗口（Redis）
- **健康检查**：
  - 429 → 冷却 60s
  - 401 → 标记失效
  - 5xx → 重试其他 Key

### 4.3 Adapter
- OpenAI `chat.completions` ↔ Anthropic `messages`
- 关键映射：
  - `model`：配置映射表
  - `messages`：`system` role → Anthropic 顶层 `system`
  - `max_tokens` / `temperature` / `stream`：透传
  - `tools`：Phase 1 不做
- SSE 流式响应逐帧转换

### 4.4 Forwarder
- `httpx.AsyncClient`（不用 OpenAI SDK）
- 超时：连接 10s，读取 120s（流式）
- 重试：最多 2 次，指数退避 1s → 3s

## 5. 配置

```yaml
server:
  host: "0.0.0.0"
  port: 8080

auth:
  virtual_keys:
    - key: "sk-virtual-team-a"
      allowed_models: ["claude-sonnet", "gpt-4o"]
      rpm_limit: 100

providers:
  kimi:
    base_url: "https://api.moonshot.cn/v1"
    keys:
      - id: "kimi-1"
        key: "${KIMI_KEY_1}"
        weight: 1
        rpm_limit: 60
    model_map:
      "gpt-4": "moonshot-v1-8k"
      "claude-sonnet": "moonshot-v1-32k"

  deepseek:
    base_url: "https://api.deepseek.com/v1"
    keys:
      - id: "deepseek-1"
        key: "${DEEPSEEK_KEY_1}"
        weight: 1
        rpm_limit: 60
    model_map:
      "gpt-4o": "deepseek-chat"
      "claude-haiku": "deepseek-chat"

redis:
  host: "localhost"
  port: 6379
  db: 0

router:
  strategy: "round_robin"
  retry_count: 2
  timeout_seconds: 60

logging:
  level: "INFO"
  format: "json"
```

## 6. 技术栈

- Go 1.22+
- Fiber（高性能 HTTP 框架）
- `net/http` + `httputil.ReverseProxy`（转发层）
- go-redis（Redis 客户端）
- Prometheus client_golang
- viper（配置管理）

## 7. 风险与待决策

| # | 问题 | 当前假设 |
|---|------|---------|
| 1 | Tool Calling | Phase 1 不做 |
| 2 | 多 Provider fallback | Phase 1 先做 Kimi 多 Key，DeepSeek Phase 2 |
| 3 | 虚拟 Key 持久化 | 先放配置文件 |
| 4 | 流式 token 计数 | Phase 1 不做精确计数 |

## 8. 验收标准

- [ ] `curl` 调用 OpenAI 入口返回正确响应
- [ ] `curl` 调用 Anthropic 入口返回正确响应
- [ ] SSE 流式逐字输出正常
- [ ] 禁用 1 个 Key 后自动路由到剩余 Key
- [ ] 单 Key 超 RPM 时限流（429）
- [ ] Prometheus `/metrics` 可见
- [ ] 所有请求日志带 trace_id
