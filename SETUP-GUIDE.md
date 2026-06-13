# LLM Gateway 配置完成报告

## Gateway 服务端状态 (101.35.235.78:8080)

| 端点 | 状态 | 说明 |
|------|------|------|
| `POST /v1/chat/completions` | 可用 | OpenAI Chat Completions API |
| `POST /v1/messages` | 可用 | Anthropic Messages API |
| `POST /v1/responses` | 可用 | OpenAI Responses API (Codex 格式) |
| 流式 SSE | 可用 | 全端点支持 |
| Tool Calling | 可用 | 已验证 function_call 转换 |

服务端需要设置这些环境变量，`config.yaml` 会从 `${...}` 占位符读取：

```bash
export LG_VIRTUAL_KEY_TEAM_A=sk-virtual-team-a
export DEEPSEEK_KEY_1=<your-deepseek-api-key>
```

---

## 1. Codex CLI 配置结果

Codex CLI 0.134.0 可以通过自定义 `model_providers` 使用 Gateway。不要使用顶层
`base_url`，该字段不是有效配置。

推荐创建独立 profile，避免覆盖默认 OpenAI/ChatGPT 配置：

```toml
# ~/.codex/gateway.config.toml
model_provider = "gateway"
model = "gpt-5.5"
model_reasoning_effort = "medium"

[model_providers.gateway]
name = "LLM Gateway"
base_url = "http://101.35.235.78:8080/v1"
env_key = "LLM_GATEWAY_API_KEY"
wire_api = "responses"
```

Windows PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("LLM_GATEWAY_API_KEY", "sk-virtual-team-a", "User")
codex -p gateway
```

或非交互测试：

```powershell
codex exec -p gateway --skip-git-repo-check "Reply with OK only."
```

---

## 2. Claude Code 配置 (推荐)

Claude Code 支持 `ANTHROPIC_BASE_URL` 环境变量，strace 确认它会连接 Gateway。

### Windows PowerShell 配置

```powershell
# 1. 安装 Claude Code
npm install -g @anthropic-ai/claude-code

# 2. 永久设置环境变量（系统级，重启后仍有效）
[Environment]::SetEnvironmentVariable("ANTHROPIC_BASE_URL", "http://101.35.235.78:8080", "User")
[Environment]::SetEnvironmentVariable("ANTHROPIC_AUTH_TOKEN", "sk-virtual-team-a", "User")

# 3. 重启 PowerShell，然后运行
claude
```

### 为什么不用 PowerShell Profile？

Profile 方式只在 PowerShell 内有效，换到其他终端（CMD、VS Code Terminal、Git Bash）就失效。系统环境变量方式是全局的，推荐用这种方式。

### 模型映射

| Claude Code 请求 | Gateway 映射到 |
|------------------|----------------|
| claude-sonnet | deepseek-v4-pro |
| claude-sonnet-4-6 | deepseek-v4-pro |
| claude-opus-4-7 | deepseek-v4-pro |
| claude-3-5-sonnet | deepseek-v4-pro |
| claude-3-5-sonnet-20241022 | deepseek-v4-pro |
| claude-3-7-sonnet-20250219 | deepseek-v4-pro |

---

## 3. aider 配置（替代 Codex CLI）

aider 原生支持自定义 base URL，可作为 Codex CLI 的替代：

```bash
# 安装
pip install aider-install && aider-install

# 使用 Gateway
aider --model openai/gpt-4o \
      --api-key sk-virtual-team-a \
      --api-base-url http://101.35.235.78:8080
```

---

## 4. Tool Calling 验证结果

已通过 curl 完整验证：

```bash
curl -X POST http://101.35.235.78:8080/v1/responses \
  -H "Authorization: Bearer sk-virtual-team-a" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "input": "上海今天天气",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

返回：
```json
{
  "output": [
    {"type": "message", "content": [{"type":"output_text","text":"让我查询..."}]},
    {"type": "function_call", "name": "get_weather", "arguments": "{\"city\": \"上海\"}"}
  ]
}
```

Gateway 正确将 Responses API tools 转发到 DeepSeek，并将 DeepSeek 的 function_call 转回 Responses API 格式。

---

## 5. MCP 说明

MCP (Model Context Protocol) 是 Claude Code / Codex CLI 客户端层面的功能：
- Gateway 只负责 HTTP API 转发，不处理 MCP 协议
- Claude Code 的 MCP 工具调用**会在客户端组装**，然后通过 `/v1/messages` 发到 Gateway
- 只要 Claude Code 能连上 Gateway，MCP 工具就能通过 Gateway 工作

---

## 已知限制

1. Claude Code / Codex CLI 的真实请求包含大量客户端特有字段，Gateway 只转发上游支持的核心字段
2. Gateway 目前只有 HTTP，如需 HTTPS 需额外配置反向代理
