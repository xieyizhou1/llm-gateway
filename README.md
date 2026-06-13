# LLM Gateway

Go 1.22+ 实现的 LLM 网关，支持 OpenAI/Anthropic 协议转换。

## 技术栈

- Go 1.22+
- Fiber (HTTP 框架)
- go-redis (Redis 客户端)
- viper (配置管理)
- Prometheus client_golang

## 目录结构

```
.
├── cmd/gateway/          # 主入口
├── internal/
│   ├── adapter/          # 协议转换 (Anthropic ↔ OpenAI)
│   ├── auth/             # 虚拟 Key 鉴权
│   ├── config/           # 配置加载
│   ├── forwarder/        # HTTP 转发
│   ├── middleware/       # 日志、Metrics、TraceID
│   ├── models/           # 请求/响应结构体
│   └── router/           # 负载均衡 + Key Pool + 限流
├── tests/
│   ├── unit/             # 单元测试
│   └── api/              # 接口测试
├── Dockerfile
├── docker-compose.yaml
└── config.yaml
```
