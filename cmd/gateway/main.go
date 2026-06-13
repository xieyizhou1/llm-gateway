// LLM Gateway 主入口。
// 提供 OpenAI Chat Completions / Anthropic Messages / OpenAI Responses 三种 API 格式，
// 内部统一转发到 Kimi / DeepSeek 上游 Provider。
package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"

	"llm-gateway/internal/adapter"
	"llm-gateway/internal/auth"
	"llm-gateway/internal/config"
	"llm-gateway/internal/forwarder"
	mw "llm-gateway/internal/middleware"
	"llm-gateway/internal/models"
	"llm-gateway/internal/router"
	"llm-gateway/internal/usage"
)

// GatewayDeps 保存网关初始化所需的依赖。
type GatewayDeps struct {
	logger      *mw.Logger
	cfg         *config.Config
	redisClient *redis.Client
	authService *auth.Auth
	router      *router.Router
	forwarder   *forwarder.Forwarder
	usageStore  *usage.Store
}

// setupGateway 加载配置并初始化所有依赖，返回 GatewayDeps 和可能的错误。
func setupGateway(cfgPath string, logLevel string) (*GatewayDeps, error) {
	logger := mw.NewLogger(logLevel)
	if logger == nil {
		logger = mw.NewLogger("INFO")
	}

	cfg, err := config.Load(cfgPath)
	if err != nil {
		logger.Error("system", "gateway", "load_config_failed", err, nil)
		return nil, err
	}

	logger = mw.NewLogger(cfg.Logging.Level)

	redisClient := redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%d", cfg.Redis.Host, cfg.Redis.Port),
		DB:   cfg.Redis.DB,
	})
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		logger.Warn("system", "redis", "ping_failed", map[string]interface{}{
			"error": err.Error(),
			"note":  "continuing without rate limiting",
		})
		_ = redisClient.Close()
		redisClient = nil
	}

	var usageStore *usage.Store
	if cfg.Usage.Enabled {
		usageStore, err = usage.Open(cfg.Usage.SQLitePath)
		if err != nil {
			logger.Error("system", "usage", "open_store_failed", err, map[string]interface{}{"path": cfg.Usage.SQLitePath})
			return nil, err
		}
		if err := usageStore.SeedConfigVirtualKeys(context.Background(), cfg.Auth.VirtualKeys); err != nil {
			_ = usageStore.Close()
			return nil, err
		}
	}

	authService := auth.NewAuth(&cfg.Auth)
	if usageStore != nil {
		keys, err := usageStore.ActiveVirtualKeys(context.Background())
		if err != nil {
			_ = usageStore.Close()
			return nil, err
		}
		for _, key := range keys {
			authService.AddOrUpdateByHash(key)
		}
	}
	r := router.NewRouter(cfg, redisClient)
	fwd := forwarder.NewForwarder(r, time.Duration(cfg.Router.TimeoutSeconds)*time.Second, cfg.Router.RetryCount)

	return &GatewayDeps{
		logger:      logger,
		cfg:         cfg,
		redisClient: redisClient,
		authService: authService,
		router:      r,
		forwarder:   fwd,
		usageStore:  usageStore,
	}, nil
}

func main() {
	cfgPath := os.Getenv("CONFIG_PATH")
	if cfgPath == "" {
		cfgPath = "config.yaml"
	}

	deps, err := setupGateway(cfgPath, os.Getenv("LOG_LEVEL"))
	if err != nil {
		os.Exit(1)
	}
	defer deps.forwarder.Close()
	if deps.usageStore != nil {
		defer deps.usageStore.Close()
	}
	if deps.redisClient != nil {
		defer deps.redisClient.Close()
	}

	logger := deps.logger
	cfg := deps.cfg
	authService := deps.authService
	r := deps.router
	fwd := deps.forwarder
	usageStore := deps.usageStore

	// Fiber
	app := fiber.New(fiber.Config{
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  60 * time.Second,
	})

	app.Use(recover.New())
	app.Use(mw.TraceIDMiddleware())
	app.Use(mw.RequestLogMiddleware(logger))
	app.Use(mw.MetricsMiddleware())
	if usageStore != nil {
		app.Use(func(c *fiber.Ctx) error {
			if err := c.Next(); err != nil {
				return err
			}
			if keyHash, ok := c.Locals("virtual_key_hash").(string); ok && keyHash != "" {
				usageStore.TouchVirtualKey(context.Background(), keyHash)
			}
			return nil
		})
	}

	app.Get("/", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok"})
	})

	app.Get("/v1", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok"})
	})

	// Health
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok", "providers": r.HealthStatus()})
	})

	// Metrics
	app.Get("/metrics", func(c *fiber.Ctx) error {
		c.Set("Content-Type", "text/plain; charset=utf-8")
		req, _ := http.NewRequest(http.MethodGet, "/metrics", nil)
		promhttp.Handler().ServeHTTP(
			&fiberResponseWriter{ctx: c},
			req,
		)
		return nil
	})

	if cfg.Dashboard.Enabled && usageStore != nil {
		registerDashboard(app, cfg, usageStore, authService, r)
	}

	// OpenAI Chat Completions
	app.Post("/v1/chat/completions", authMiddleware(authService, usageStore), handleOpenAIChatCompletions(fwd, logger, r, usageStore))

	app.Get("/v1/models", authMiddleware(authService, usageStore), func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"object": "list",
			"data": []fiber.Map{
				{"id": "gpt-5.5", "object": "model", "owned_by": "llm-gateway"},
				{"id": "gpt-5.4", "object": "model", "owned_by": "llm-gateway"},
				{"id": "gpt-4o", "object": "model", "owned_by": "llm-gateway"},
				{"id": "deepseek-v4-pro", "object": "model", "owned_by": "llm-gateway"},
			},
		})
	})

	// Anthropic Messages
	app.Post("/v1/messages", authMiddleware(authService, usageStore), handleAnthropicMessages(fwd, logger, r, usageStore))

	// OpenAI Responses (Codex)
	app.Post("/v1/responses", authMiddleware(authService, usageStore), handleResponses(fwd, logger, r, usageStore))

	// Graceful shutdown
	go func() {
		addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
		logger.Info("system", "gateway", "server_start", map[string]interface{}{
			"addr": addr,
		})
		if err := app.Listen(addr); err != nil {
			logger.Error("system", "gateway", "server_error", err, nil)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("system", "gateway", "server_shutdown", nil)
	if err := app.Shutdown(); err != nil {
		logger.Error("system", "gateway", "shutdown_error", err, nil)
	}
}

func handleOpenAIChatCompletions(fwd *forwarder.Forwarder, logger *mw.Logger, r *router.Router, usageStore *usage.Store) fiber.Handler {
	return func(c *fiber.Ctx) error {
		traceID := mw.ExtractTraceID(c)
		ctx := c.Context()
		start := time.Now()
		record := newUsageRecord(c, "openai_chat")
		defer func() {
			if usageStore != nil && record != nil && !record.Stream {
				finishUsageRecord(c, usageStore, start, record)
			}
		}()

		var req models.OpenAIChatCompletionRequest
		if err := c.BodyParser(&req); err != nil {
			logger.Error(traceID, "handler", "parse_openai_request", err, nil)
			record.ErrorCode = "invalid_request"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
		}
		adapter.EnsureAssistantToolCallReasoning(req.Messages)
		record.Model = req.Model
		record.Stream = req.Stream
		record.HasImage = req.HasImageContent()
		record.InputHash = hashMessages(req.Messages)

		// 模型权限检查
		vk, _ := auth.ExtractVirtualKey(c)
		if vk != nil && len(vk.AllowedModels) > 0 {
			allowed := false
			for _, m := range vk.AllowedModels {
				if m == req.Model {
					allowed = true
					break
				}
			}
			if !allowed {
				logger.Warn(traceID, "handler", "model_not_allowed", map[string]interface{}{
					"model":    req.Model,
					"key_hash": vk.KeyHash,
				})
				record.ErrorCode = "model_not_allowed"
				record.ErrorMessage = "model not allowed"
				return c.Status(fiber.StatusForbidden).JSON(fiber.Map{"error": "model not allowed"})
			}
		}

		logger.Info(traceID, "handler", "openai_request", map[string]interface{}{
			"model":  req.Model,
			"stream": req.Stream,
		})

		if req.Stream {
			result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, &req)
			if err != nil {
				logger.Error(traceID, "forwarder", "forward_openai_stream", err, nil)
				record.ErrorCode = "gateway_error"
				record.ErrorMessage = err.Error()
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
			}
			resp := result.Response
			applyForwardInfo(record, result.Info)
			defer resp.Body.Close()

			// 检查上游 200 响应是否实际是 JSON 错误
			reader, errBody, peekErr := peekUpstreamStreamError(resp.Body)
			if peekErr != nil && errBody != nil {
				logger.Error(traceID, "handler", "upstream_stream_error", peekErr, map[string]interface{}{
					"provider": result.Info.Provider,
					"key_id":   result.Info.ProviderKeyID,
				})
				r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
				record.ErrorCode = "upstream_error"
				record.ErrorMessage = string(errBody)
				record.ResponseBody = truncateBody(string(errBody))
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
			}

			c.Set("Content-Type", "text/event-stream")
			c.Set("Cache-Control", "no-cache")
			c.Set("Connection", "keep-alive")

			capture := &limitedBodyCapture{limit: 16000}
			writer := io.MultiWriter(c.Context().Response.BodyWriter(), capture)
			err = adapter.RewriteOpenAIStreamBodyWithReasoning(reader, &ttftMarkWriter{
				w: writer,
				mark: func() {
					markTTFT(record, start)
				},
			}, func(finishReason string) {
				record.FinishReason = finishReason
			}, func(usage *models.OpenAIUsage) {
				record.PromptTokens = usage.PromptTokens
				record.OutputTokens = usage.CompletionTokens
				record.TotalTokens = usage.TotalTokens
			})
			record.ResponseBody = capture.String()
			if err != nil {
				logger.Error(traceID, "handler", "stream_copy", err, nil)
			}
			finishUsageRecord(c, usageStore, start, record)
			return nil
		}

		result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, &req)
		if err != nil {
			logger.Error(traceID, "forwarder", "forward_openai", err, nil)
			record.ErrorCode = "gateway_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
		}
		resp := result.Response
		applyForwardInfo(record, result.Info)
		defer resp.Body.Close()
		if resp.StatusCode >= 400 {
			return proxyUpstreamError(c, resp, record)
		}

		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			logger.Error(traceID, "handler", "read_openai_response", err, nil)
			record.ErrorCode = "read_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "read response failed"})
		}
		record.ResponseBody = truncateBody(string(rawBody))

		// 检查上游 200 响应是否实际是 JSON 错误（某些 Provider 会这样做）
		trimmed := strings.TrimSpace(string(rawBody))
		if strings.HasPrefix(trimmed, "{\"error\"") || strings.HasPrefix(trimmed, "{\"type\"") {
			logger.Error(traceID, "handler", "upstream_nonstream_error", fmt.Errorf("upstream returned 200 with error body"), map[string]interface{}{
				"provider": result.Info.Provider,
				"key_id":   result.Info.ProviderKeyID,
			})
			r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
			record.ErrorCode = "upstream_error"
			record.ErrorMessage = string(rawBody)
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
		}

		openAIResp, cacheUsage, err := decodeOpenAIResponseAndCache(rawBody)
		if err != nil {
			logger.Error(traceID, "handler", "decode_openai_response", err, nil)
			record.ErrorCode = "decode_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "decode response failed"})
		}
		adapter.EnsureAssistantToolCallReasoningInResponse(&openAIResp)

		outputHash := hashOpenAIResponseContent(&openAIResp)
		record.PromptTokens = openAIResp.Usage.PromptTokens
		record.OutputTokens = openAIResp.Usage.CompletionTokens
		record.TotalTokens = openAIResp.Usage.TotalTokens
		record.CacheReadTokens = cacheUsage.CacheReadTokens
		record.CacheWriteTokens = cacheUsage.CacheWriteTokens
		record.CacheHitRate = cacheUsage.CacheHitRate
		record.OutputHash = outputHash
		record.FinishReason = firstOpenAIFinishReason(openAIResp.Choices)
		logger.Audit(traceID, "llm_call", map[string]interface{}{
			"model":              req.Model,
			"model_version":      req.Model,
			"prompt_version":     "",
			"prompt_tokens":      openAIResp.Usage.PromptTokens,
			"completion_tokens":  openAIResp.Usage.CompletionTokens,
			"cache_read_tokens":  cacheUsage.CacheReadTokens,
			"cache_write_tokens": cacheUsage.CacheWriteTokens,
			"cache_hit_rate":     cacheUsage.CacheHitRate,
			"input_hash":         hashMessages(req.Messages),
			"output_hash":        outputHash,
		})

		return c.Status(resp.StatusCode).JSON(openAIResp)
	}
}

func proxyUpstreamError(c *fiber.Ctx, resp *http.Response, record *usage.RequestLog) error {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		record.ErrorCode = "read_error"
		record.ErrorMessage = err.Error()
		return c.Status(fiber.StatusBadGateway).JSON(fiber.Map{"error": "read upstream error failed"})
	}
	record.StatusCode = resp.StatusCode
	record.ErrorCode = "upstream_error"
	record.ErrorMessage = string(body)
	record.ResponseBody = truncateBody(string(body))
	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		c.Set("Content-Type", contentType)
	}
	return c.Status(resp.StatusCode).Send(body)
}

// peekUpstreamStreamError 检查上游 200 响应是否是伪装成 SSE 的 JSON 错误。
// 某些 Provider（如 Kimi）在流式请求时返回 HTTP 200，但 body 是 {"error":...}。
// 返回非 nil 时应当作上游错误处理。
func peekUpstreamStreamError(r io.Reader) (io.Reader, []byte, error) {
	// 读取前 4KB 来判断是否是 JSON 错误
	buf := make([]byte, 4096)
	n, err := r.Read(buf)
	if n > 0 {
		// 检查是否以 JSON 错误开头（可能前面有空白字符）
		trimmed := strings.TrimSpace(string(buf[:n]))
		if strings.HasPrefix(trimmed, "{\"error\"") || strings.HasPrefix(trimmed, "{\"type\"") {
			// 把整个响应读出来构造错误
			var rest bytes.Buffer
			rest.Write(buf[:n])
			if err == nil {
				io.Copy(&rest, r)
			}
			return nil, rest.Bytes(), fmt.Errorf("upstream returned 200 with error body")
		}
		// 不是错误，把已读的数据和剩余 reader 拼接返回
		return io.MultiReader(bytes.NewReader(buf[:n]), r), nil, nil
	}
	if err != nil && err != io.EOF {
		return nil, nil, err
	}
	return r, nil, nil
}

func handleAnthropicMessages(fwd *forwarder.Forwarder, logger *mw.Logger, r *router.Router, usageStore *usage.Store) fiber.Handler {
	return func(c *fiber.Ctx) error {
		traceID := mw.ExtractTraceID(c)
		ctx := c.Context()
		start := time.Now()
		record := newUsageRecord(c, "anthropic_messages")
		defer func() {
			if usageStore != nil && record != nil && !record.Stream {
				finishUsageRecord(c, usageStore, start, record)
			}
		}()

		var anthropicReq models.AnthropicMessageRequest
		if err := c.BodyParser(&anthropicReq); err != nil {
			logger.Error(traceID, "handler", "parse_anthropic_request", err, nil)
			record.ErrorCode = "invalid_request"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
		}
		record.Model = anthropicReq.Model
		record.Stream = anthropicReq.Stream
		record.InputHash = hashAnthropicMessages(anthropicReq.Messages, anthropicReq.System)

		// 模型权限检查
		vk, _ := auth.ExtractVirtualKey(c)
		if vk != nil && len(vk.AllowedModels) > 0 {
			allowed := false
			for _, m := range vk.AllowedModels {
				if m == anthropicReq.Model {
					allowed = true
					break
				}
			}
			if !allowed {
				logger.Warn(traceID, "handler", "model_not_allowed", map[string]interface{}{
					"model":    anthropicReq.Model,
					"key_hash": vk.KeyHash,
				})
				record.ErrorCode = "model_not_allowed"
				record.ErrorMessage = "model not allowed"
				return c.Status(fiber.StatusForbidden).JSON(fiber.Map{"error": "model not allowed"})
			}
		}

		logger.Info(traceID, "handler", "anthropic_request", map[string]interface{}{
			"model":  anthropicReq.Model,
			"stream": anthropicReq.Stream,
		})

		// 转换为 OpenAI 请求
		openAIReq := adapter.AnthropicToOpenAI(&anthropicReq)
		record.HasImage = openAIReq.HasImageContent()

		if anthropicReq.Stream {
			if len(anthropicReq.Tools) > 0 {
				openAIReq.Stream = false
				result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, openAIReq)
				if err != nil {
					logger.Error(traceID, "forwarder", "forward_anthropic_tool_stream", err, nil)
					record.ErrorCode = "gateway_error"
					record.ErrorMessage = err.Error()
					finishUsageRecord(c, usageStore, start, record)
					return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
				}
				resp := result.Response
				applyForwardInfo(record, result.Info)
				defer resp.Body.Close()

				rawBody, err := io.ReadAll(resp.Body)
				if err != nil {
					logger.Error(traceID, "handler", "read_anthropic_tool_stream_response", err, nil)
					record.ErrorCode = "read_error"
					record.ErrorMessage = err.Error()
					finishUsageRecord(c, usageStore, start, record)
					return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "read response failed"})
				}
				record.ResponseBody = truncateBody(string(rawBody))

				// 检查上游 200 响应是否实际是 JSON 错误（某些 Provider 会这样做）
				trimmed := strings.TrimSpace(string(rawBody))
				if strings.HasPrefix(trimmed, "{\"error\"") || strings.HasPrefix(trimmed, "{\"type\"") {
					logger.Error(traceID, "handler", "upstream_tool_stream_error", fmt.Errorf("upstream returned 200 with error body"), map[string]interface{}{
						"provider": result.Info.Provider,
						"key_id":   result.Info.ProviderKeyID,
					})
					r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
					record.ErrorCode = "upstream_error"
					record.ErrorMessage = string(rawBody)
					finishUsageRecord(c, usageStore, start, record)
					return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
				}

				openAIResp, cacheUsage, err := decodeOpenAIResponseAndCache(rawBody)
				if err != nil {
					logger.Error(traceID, "handler", "decode_anthropic_tool_stream_response", err, nil)
					record.ErrorCode = "decode_error"
					record.ErrorMessage = err.Error()
					finishUsageRecord(c, usageStore, start, record)
					return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "decode response failed"})
				}

				anthropicResp := adapter.OpenAIToAnthropic(&openAIResp)
				outputHash := hashAnthropicResponseContent(anthropicResp)
				record.PromptTokens = anthropicResp.Usage.InputTokens
				record.OutputTokens = anthropicResp.Usage.OutputTokens
				record.TotalTokens = anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens
				record.CacheReadTokens = cacheUsage.CacheReadTokens
				record.CacheWriteTokens = cacheUsage.CacheWriteTokens
				record.CacheHitRate = cacheUsage.CacheHitRate
				record.OutputHash = outputHash
				logger.Audit(traceID, "llm_call", map[string]interface{}{
					"model":              anthropicReq.Model,
					"model_version":      anthropicReq.Model,
					"prompt_version":     "",
					"input_tokens":       anthropicResp.Usage.InputTokens,
					"output_tokens":      anthropicResp.Usage.OutputTokens,
					"cache_read_tokens":  cacheUsage.CacheReadTokens,
					"cache_write_tokens": cacheUsage.CacheWriteTokens,
					"cache_hit_rate":     cacheUsage.CacheHitRate,
					"input_hash":         hashAnthropicMessages(anthropicReq.Messages, anthropicReq.System),
					"output_hash":        outputHash,
				})

				c.Set("Content-Type", "text/event-stream")
				c.Set("Cache-Control", "no-cache")
				c.Set("Connection", "keep-alive")
				writeAnthropicResponseSSE(c.Context().Response.BodyWriter(), anthropicResp)
				finishUsageRecord(c, usageStore, start, record)
				return nil
			}

			result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, openAIReq)
			if err != nil {
				logger.Error(traceID, "forwarder", "forward_anthropic_stream", err, nil)
				record.ErrorCode = "gateway_error"
				record.ErrorMessage = err.Error()
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
			}
			resp := result.Response
			applyForwardInfo(record, result.Info)
			defer resp.Body.Close()

			// 检查上游 200 响应是否实际是 JSON 错误（某些 Provider 会这样做）
			reader, errBody, peekErr := peekUpstreamStreamError(resp.Body)
			if peekErr != nil && errBody != nil {
				logger.Error(traceID, "handler", "upstream_stream_error", peekErr, map[string]interface{}{
					"provider": result.Info.Provider,
					"key_id":   result.Info.ProviderKeyID,
				})
				// 标记 key 为失败，让后续重试能选择其他 key
				r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
				record.ErrorCode = "upstream_error"
				record.ErrorMessage = string(errBody)
				record.ResponseBody = truncateBody(string(errBody))
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
			}

			c.Set("Content-Type", "text/event-stream")
			c.Set("Cache-Control", "no-cache")
			c.Set("Connection", "keep-alive")

			bodyWriter := &ttftMarkWriter{
				w: c.Context().Response.BodyWriter(),
				mark: func() {
					markTTFT(record, start)
				},
			}

			// 逐行读取 SSE，转换后输出
			buf := make([]byte, 4096)
			var pending []byte
			streamState := newAnthropicStreamState()

			for {
				n, err := reader.Read(buf)
				if n > 0 {
					pending = append(pending, buf[:n]...)
					for {
						idx := findDoubleNewline(pending)
						if idx < 0 {
							break
						}
						line := string(pending[:idx])
						pending = pending[idx+2:]

						if strings.HasPrefix(line, "data:") {
							data := strings.TrimPrefix(line, "data:")
							data = strings.TrimSpace(data)
							if data == "[DONE]" {
								streamState.finish(bodyWriter, "stop", record)
								continue
							}

							var chunk models.OpenAIStreamChunk
							if jsonErr := json.Unmarshal([]byte(data), &chunk); jsonErr == nil {
								if chunk.Usage != nil && record != nil {
									record.PromptTokens = chunk.Usage.PromptTokens
									record.OutputTokens = chunk.Usage.CompletionTokens
									record.TotalTokens = chunk.Usage.TotalTokens
								}
								streamState.start(bodyWriter, &chunk, anthropicReq)
								for _, choice := range chunk.Choices {
									streamState.emitReasoning(bodyWriter, choice.Delta.ReasoningContent)
									streamState.emitText(bodyWriter, choice.Delta.Content)
									for _, tc := range choice.Delta.ToolCalls {
										streamState.emitToolCall(bodyWriter, tc)
									}
									if choice.FinishReason != nil && *choice.FinishReason != "" {
										streamState.finish(bodyWriter, *choice.FinishReason, record)
									}
								}
							} else {
								// 无法解析时透传
								bodyWriter.Write([]byte(line + "\n\n"))
							}
						}
					}
				}
				if err == io.EOF {
					if !streamState.messageStopped {
						streamState.finish(bodyWriter, "stop", record)
					}
					break
				}
				if err != nil {
					logger.Error(traceID, "handler", "stream_read", err, nil)
					break
				}
			}
			finishUsageRecord(c, usageStore, start, record)
			return nil
		}

		result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, openAIReq)
		if err != nil {
			logger.Error(traceID, "forwarder", "forward_anthropic", err, nil)
			record.ErrorCode = "gateway_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
		}
		resp := result.Response
		applyForwardInfo(record, result.Info)
		defer resp.Body.Close()

		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			logger.Error(traceID, "handler", "read_openai_response", err, nil)
			record.ErrorCode = "read_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "read response failed"})
		}
		record.ResponseBody = truncateBody(string(rawBody))

		openAIResp, cacheUsage, err := decodeOpenAIResponseAndCache(rawBody)
		if err != nil {
			logger.Error(traceID, "handler", "decode_openai_response", err, nil)
			record.ErrorCode = "decode_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "decode response failed"})
		}

		anthropicResp := adapter.OpenAIToAnthropic(&openAIResp)

		outputHash := hashAnthropicResponseContent(anthropicResp)
		record.PromptTokens = anthropicResp.Usage.InputTokens
		record.OutputTokens = anthropicResp.Usage.OutputTokens
		record.TotalTokens = anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens
		record.CacheReadTokens = cacheUsage.CacheReadTokens
		record.CacheWriteTokens = cacheUsage.CacheWriteTokens
		record.CacheHitRate = cacheUsage.CacheHitRate
		record.OutputHash = outputHash
		logger.Audit(traceID, "llm_call", map[string]interface{}{
			"model":              anthropicReq.Model,
			"model_version":      anthropicReq.Model,
			"prompt_version":     "",
			"input_tokens":       anthropicResp.Usage.InputTokens,
			"output_tokens":      anthropicResp.Usage.OutputTokens,
			"cache_read_tokens":  cacheUsage.CacheReadTokens,
			"cache_write_tokens": cacheUsage.CacheWriteTokens,
			"cache_hit_rate":     cacheUsage.CacheHitRate,
			"input_hash":         hashAnthropicMessages(anthropicReq.Messages, anthropicReq.System),
			"output_hash":        outputHash,
		})

		return c.Status(resp.StatusCode).JSON(anthropicResp)
	}
}

func writeAnthropicSSE(w io.Writer, event string, payload interface{}) {
	b, _ := json.Marshal(payload)
	writeAnthropicSSEBytes(w, event, b)
}

func writeAnthropicSSEBytes(w io.Writer, event string, payload []byte) {
	if event != "" {
		_, _ = w.Write([]byte(fmt.Sprintf("event: %s\n", event)))
	}
	_, _ = w.Write([]byte(fmt.Sprintf("data: %s\n\n", payload)))
}

func writeAnthropicResponseSSE(w io.Writer, resp *models.AnthropicMessageResponse) {
	writeAnthropicSSE(w, "message_start", fiber.Map{
		"type": "message_start",
		"message": fiber.Map{
			"id":            resp.ID,
			"type":          "message",
			"role":          resp.Role,
			"model":         resp.Model,
			"content":       []interface{}{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": fiber.Map{
				"input_tokens":  resp.Usage.InputTokens,
				"output_tokens": 0,
			},
		},
	})

	for i, block := range resp.Content {
		writeAnthropicContentBlockSSE(w, i, block)
	}

	writeAnthropicSSE(w, "message_delta", fiber.Map{
		"type": "message_delta",
		"delta": fiber.Map{
			"stop_reason":   resp.StopReason,
			"stop_sequence": resp.StopSequence,
		},
		"usage": fiber.Map{"output_tokens": resp.Usage.OutputTokens},
	})
	writeAnthropicSSE(w, "message_stop", fiber.Map{"type": "message_stop"})
}

func writeAnthropicContentBlockSSE(w io.Writer, index int, block models.AnthropicContent) {
	switch block.Type {
	case "thinking":
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": index,
			"content_block": fiber.Map{
				"type":     "thinking",
				"thinking": "",
			},
		})
		if block.Thinking != "" {
			writeAnthropicSSE(w, "content_block_delta", fiber.Map{
				"type":  "content_block_delta",
				"index": index,
				"delta": fiber.Map{
					"type":     "thinking_delta",
					"thinking": block.Thinking,
				},
			})
		}
	case "tool_use":
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": index,
			"content_block": fiber.Map{
				"type":  "tool_use",
				"id":    block.ID,
				"name":  block.Name,
				"input": fiber.Map{},
			},
		})
		inputJSON := "{}"
		if block.Input != nil {
			if raw, ok := block.Input.(string); ok {
				raw = strings.TrimSpace(raw)
				if strings.HasPrefix(raw, "{") || strings.HasPrefix(raw, "[") {
					inputJSON = raw
				} else if b, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(b)
				}
			} else {
				if b, err := json.Marshal(block.Input); err == nil {
					inputJSON = string(b)
				}
			}
		}
		writeAnthropicSSE(w, "content_block_delta", fiber.Map{
			"type":  "content_block_delta",
			"index": index,
			"delta": fiber.Map{
				"type":         "input_json_delta",
				"partial_json": inputJSON,
			},
		})
	default:
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": index,
			"content_block": fiber.Map{
				"type": "text",
				"text": "",
			},
		})
		if block.Text != "" {
			writeAnthropicSSE(w, "content_block_delta", fiber.Map{
				"type":  "content_block_delta",
				"index": index,
				"delta": fiber.Map{
					"type": "text_delta",
					"text": block.Text,
				},
			})
		}
	}

	writeAnthropicSSE(w, "content_block_stop", fiber.Map{
		"type":  "content_block_stop",
		"index": index,
	})
}

type anthropicStreamBlockState struct {
	index   int
	kind    string
	id      string
	name    string
	started bool
	stopped bool
}

type anthropicStreamState struct {
	started        bool
	messageStopped bool
	messageID      string
	model          string
	nextIndex      int
	textBlock      *anthropicStreamBlockState
	reasoningBlock *anthropicStreamBlockState
	toolBlocks     map[int]*anthropicStreamBlockState
	sawToolUse     bool
}

func newAnthropicStreamState() *anthropicStreamState {
	return &anthropicStreamState{
		toolBlocks: make(map[int]*anthropicStreamBlockState),
	}
}

func (s *anthropicStreamState) start(w io.Writer, chunk *models.OpenAIStreamChunk, req models.AnthropicMessageRequest) {
	if s.started {
		return
	}
	s.started = true
	s.messageID = chunk.ID
	if s.messageID == "" {
		s.messageID = "msg_stream"
	}
	s.model = chunk.Model
	if s.model == "" {
		s.model = req.Model
	}
	writeAnthropicSSE(w, "message_start", fiber.Map{
		"type": "message_start",
		"message": fiber.Map{
			"id":            s.messageID,
			"type":          "message",
			"role":          "assistant",
			"model":         s.model,
			"content":       []interface{}{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": fiber.Map{
				"input_tokens":  0,
				"output_tokens": 0,
			},
		},
	})
}

func (s *anthropicStreamState) emitText(w io.Writer, text string) {
	if text == "" {
		return
	}
	if s.textBlock == nil {
		s.textBlock = s.newBlock("text")
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": s.textBlock.index,
			"content_block": fiber.Map{
				"type": "text",
				"text": "",
			},
		})
	}
	writeAnthropicSSE(w, "content_block_delta", fiber.Map{
		"type":  "content_block_delta",
		"index": s.textBlock.index,
		"delta": fiber.Map{
			"type": "text_delta",
			"text": text,
		},
	})
}

func (s *anthropicStreamState) emitReasoning(w io.Writer, reasoning string) {
	if reasoning == "" {
		return
	}
	if s.reasoningBlock == nil {
		s.reasoningBlock = s.newBlock("thinking")
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": s.reasoningBlock.index,
			"content_block": fiber.Map{
				"type":     "thinking",
				"thinking": "",
			},
		})
	}
	writeAnthropicSSE(w, "content_block_delta", fiber.Map{
		"type":  "content_block_delta",
		"index": s.reasoningBlock.index,
		"delta": fiber.Map{
			"type":     "thinking_delta",
			"thinking": reasoning,
		},
	})
}

func (s *anthropicStreamState) emitToolCall(w io.Writer, tc models.ToolCall) {
	name := tc.Function.Name
	key := tc.Index
	if key == 0 && tc.ID != "" {
		for existingKey, existing := range s.toolBlocks {
			if existing.id == tc.ID {
				key = existingKey
				break
			}
		}
	}
	block := s.toolBlocks[key]
	if block == nil {
		if name == "" {
			return
		}
		block = s.newBlock("tool_use")
		block.id = tc.ID
		if block.id == "" {
			block.id = "toolu_" + randomID()
		}
		block.name = name
		s.toolBlocks[key] = block
		s.sawToolUse = true
		writeAnthropicSSE(w, "content_block_start", fiber.Map{
			"type":  "content_block_start",
			"index": block.index,
			"content_block": fiber.Map{
				"type":  "tool_use",
				"id":    block.id,
				"name":  block.name,
				"input": fiber.Map{},
			},
		})
	} else if name != "" && block.name == "" {
		block.name = name
	}
	if tc.Function.Arguments != "" {
		writeAnthropicSSE(w, "content_block_delta", fiber.Map{
			"type":  "content_block_delta",
			"index": block.index,
			"delta": fiber.Map{
				"type":         "input_json_delta",
				"partial_json": tc.Function.Arguments,
			},
		})
	}
}

func (s *anthropicStreamState) finish(w io.Writer, finishReason string, record *usage.RequestLog) {
	if s.messageStopped {
		return
	}
	s.stopBlocks(w)
	stopReason := "end_turn"
	switch finishReason {
	case "tool_calls":
		stopReason = "tool_use"
	case "length":
		stopReason = "max_tokens"
	case "stop":
		stopReason = "end_turn"
	}
	if s.sawToolUse && stopReason == "end_turn" {
		stopReason = "tool_use"
	}
	if record != nil {
		record.FinishReason = stopReason
	}
	writeAnthropicSSE(w, "message_delta", fiber.Map{
		"type":  "message_delta",
		"delta": fiber.Map{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": fiber.Map{"output_tokens": 0},
	})
	writeAnthropicSSE(w, "message_stop", fiber.Map{"type": "message_stop"})
	s.messageStopped = true
}

func (s *anthropicStreamState) stopBlocks(w io.Writer) {
	blocks := make([]*anthropicStreamBlockState, 0, 3+len(s.toolBlocks))
	if s.textBlock != nil && !s.textBlock.stopped {
		blocks = append(blocks, s.textBlock)
	}
	if s.reasoningBlock != nil && !s.reasoningBlock.stopped {
		blocks = append(blocks, s.reasoningBlock)
	}
	for _, block := range s.toolBlocks {
		if block != nil && !block.stopped {
			blocks = append(blocks, block)
		}
	}
	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].index < blocks[j].index
	})
	for _, block := range blocks {
		writeAnthropicSSE(w, "content_block_stop", fiber.Map{
			"type":  "content_block_stop",
			"index": block.index,
		})
		block.stopped = true
	}
}

func (s *anthropicStreamState) newBlock(kind string) *anthropicStreamBlockState {
	block := &anthropicStreamBlockState{
		index: s.nextIndex,
		kind:  kind,
	}
	s.nextIndex++
	return block
}

func findDoubleNewline(data []byte) int {
	for i := 0; i < len(data)-1; i++ {
		if data[i] == '\n' && data[i+1] == '\n' {
			return i
		}
	}
	return -1
}

func handleResponses(fwd *forwarder.Forwarder, logger *mw.Logger, r *router.Router, usageStore *usage.Store) fiber.Handler {
	return func(c *fiber.Ctx) error {
		traceID := mw.ExtractTraceID(c)
		ctx := c.Context()
		start := time.Now()
		record := newUsageRecord(c, "responses")
		defer finishUsageRecord(c, usageStore, start, record)

		var respReq models.ResponsesRequest
		if err := c.BodyParser(&respReq); err != nil {
			logger.Error(traceID, "handler", "parse_responses_request", err, nil)
			record.ErrorCode = "invalid_request"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
		}
		record.Model = respReq.Model
		record.Stream = respReq.Stream

		// 模型权限检查
		vk, _ := auth.ExtractVirtualKey(c)
		if vk != nil && len(vk.AllowedModels) > 0 {
			allowed := false
			for _, m := range vk.AllowedModels {
				if m == respReq.Model {
					allowed = true
					break
				}
			}
			if !allowed {
				logger.Warn(traceID, "handler", "model_not_allowed", map[string]interface{}{
					"model":    respReq.Model,
					"key_hash": vk.KeyHash,
				})
				record.ErrorCode = "model_not_allowed"
				record.ErrorMessage = "model not allowed"
				return c.Status(fiber.StatusForbidden).JSON(fiber.Map{"error": "model not allowed"})
			}
		}

		logger.Info(traceID, "handler", "responses_request", map[string]interface{}{
			"model":  respReq.Model,
			"stream": respReq.Stream,
		})

		// 转换为 OpenAI Chat Completions 请求
		openAIReq := adapter.ResponsesToOpenAI(&respReq)
		record.HasImage = openAIReq.HasImageContent()
		record.InputHash = hashMessages(openAIReq.Messages)

		if respReq.Stream {
			// 流式路径：Codex CLI 默认 stream=true，将上游 Chat Completions SSE
			// 实时转换为 Responses API SSE 格式
			openAIReq.Stream = true
			result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, openAIReq)
			if err != nil {
				logger.Error(traceID, "forwarder", "forward_responses_stream", err, nil)
				record.ErrorCode = "gateway_error"
				record.ErrorMessage = err.Error()
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
			}
			resp := result.Response
			applyForwardInfo(record, result.Info)
			if resp.StatusCode >= fiber.StatusBadRequest {
				defer resp.Body.Close()
				body, _ := io.ReadAll(resp.Body)
				record.ErrorCode = "upstream_error"
				record.ErrorMessage = string(body)
				c.Set("Content-Type", resp.Header.Get("Content-Type"))
				finishUsageRecord(c, usageStore, start, record)
				if len(body) == 0 {
					return c.Status(resp.StatusCode).JSON(fiber.Map{"error": "upstream request failed"})
				}
				return c.Status(resp.StatusCode).Send(body)
			}

			c.Set("Content-Type", "text/event-stream")
			c.Set("Cache-Control", "no-cache")
			c.Set("Connection", "keep-alive")

			// 检查上游 200 响应是否实际是 JSON 错误（某些 Provider 会这样做）
			reader, errBody, peekErr := peekUpstreamStreamError(resp.Body)
			if peekErr != nil && errBody != nil {
				logger.Error(traceID, "handler", "upstream_stream_error", peekErr, map[string]interface{}{
					"provider": result.Info.Provider,
					"key_id":   result.Info.ProviderKeyID,
				})
				// 标记 key 为失败，让后续重试能选择其他 key
				r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
				record.ErrorCode = "upstream_error"
				record.ErrorMessage = string(errBody)
				record.ResponseBody = truncateBody(string(errBody))
				finishUsageRecord(c, usageStore, start, record)
				return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
			}

			// 使用 fasthttp 原生流式 body，避免 SendStream 阻塞
			record.StreamProcessing = true
			c.Context().Response.SetBodyStreamWriter(func(w *bufio.Writer) {
				defer resp.Body.Close()
				adapter.OpenAIStreamToResponsesSSEWithUsage(reader, &ttftMarkWriter{
					w: w,
					mark: func() {
						markTTFT(record, start)
					},
				}, func(usage models.OpenAIUsage) {
					record.PromptTokens = usage.PromptTokens
					record.OutputTokens = usage.CompletionTokens
					record.TotalTokens = usage.TotalTokens
				})
				w.Flush()
				finishUsageRecordWithStatus(usageStore, start, record, fiber.StatusOK)
			})

			return nil
		}

		// 非流式路径
		openAIReq.Stream = false
		result, err := fwd.ForwardOpenAIRequestWithInfo(ctx, openAIReq)
		if err != nil {
			logger.Error(traceID, "forwarder", "forward_responses", err, nil)
			record.ErrorCode = "gateway_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{"error": "service temporarily unavailable"})
		}
		resp := result.Response
		applyForwardInfo(record, result.Info)
		defer resp.Body.Close()

		rawBody, err := io.ReadAll(resp.Body)
		if err != nil {
			logger.Error(traceID, "handler", "read_openai_response", err, nil)
			record.ErrorCode = "read_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "read response failed"})
		}
		record.ResponseBody = truncateBody(string(rawBody))

		// 检查上游 200 响应是否实际是 JSON 错误（某些 Provider 会这样做）
		trimmed := strings.TrimSpace(string(rawBody))
		if strings.HasPrefix(trimmed, "{\"error\"") || strings.HasPrefix(trimmed, "{\"type\"") {
			logger.Error(traceID, "handler", "upstream_responses_nonstream_error", fmt.Errorf("upstream returned 200 with error body"), map[string]interface{}{
				"provider": result.Info.Provider,
				"key_id":   result.Info.ProviderKeyID,
			})
			r.MarkKeyResult(result.Info.Provider, result.Info.ProviderKeyID, http.StatusTooManyRequests)
			record.ErrorCode = "upstream_error"
			record.ErrorMessage = string(rawBody)
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{"error": "upstream rate limit or quota exceeded"})
		}

		openAIResp, cacheUsage, err := decodeOpenAIResponseAndCache(rawBody)
		if err != nil {
			logger.Error(traceID, "handler", "decode_openai_response", err, nil)
			record.ErrorCode = "decode_error"
			record.ErrorMessage = err.Error()
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "decode response failed"})
		}

		responsesResp := adapter.OpenAIToResponses(&openAIResp)

		outputHash := hashOpenAIResponseContent(&openAIResp)
		record.PromptTokens = openAIResp.Usage.PromptTokens
		record.OutputTokens = openAIResp.Usage.CompletionTokens
		record.TotalTokens = openAIResp.Usage.TotalTokens
		record.CacheReadTokens = cacheUsage.CacheReadTokens
		record.CacheWriteTokens = cacheUsage.CacheWriteTokens
		record.CacheHitRate = cacheUsage.CacheHitRate
		record.OutputHash = outputHash
		record.FinishReason = firstOpenAIFinishReason(openAIResp.Choices)
		logger.Audit(traceID, "llm_call", map[string]interface{}{
			"model":              respReq.Model,
			"model_version":      respReq.Model,
			"prompt_version":     "",
			"prompt_tokens":      openAIResp.Usage.PromptTokens,
			"completion_tokens":  openAIResp.Usage.CompletionTokens,
			"cache_read_tokens":  cacheUsage.CacheReadTokens,
			"cache_write_tokens": cacheUsage.CacheWriteTokens,
			"cache_hit_rate":     cacheUsage.CacheHitRate,
			"input_hash":         hashMessages(openAIReq.Messages),
			"output_hash":        outputHash,
		})

		return c.Status(resp.StatusCode).JSON(responsesResp)
	}
}

func decodeOpenAIResponseAndCache(raw []byte) (models.OpenAIChatCompletionResponse, usage.CacheUsage, error) {
	var resp models.OpenAIChatCompletionResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return resp, usage.CacheUsage{}, err
	}

	var rawEnvelope struct {
		Usage map[string]json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(raw, &rawEnvelope); err != nil {
		return resp, usage.CacheUsage{}, err
	}

	cacheUsage := usage.ParseOpenAICacheUsage(rawEnvelope.Usage, resp.Usage.PromptTokens)
	return resp, cacheUsage, nil
}

func authMiddleware(authService *auth.Auth, usageStore *usage.Store) fiber.Handler {
	base := authService.Middleware()
	return func(c *fiber.Ctx) error {
		start := time.Now()
		err := base(c)
		if usageStore != nil && c.Response().StatusCode() == fiber.StatusUnauthorized {
			record := newUsageRecord(c, protocolFromPath(c.Path()))
			record.StatusCode = fiber.StatusUnauthorized
			record.ErrorCode = "unauthorized"
			record.ErrorMessage = "unauthorized"
			record.LatencyMS = time.Since(start).Milliseconds()
			_ = usageStore.Record(context.Background(), *record)
		}
		return err
	}
}

func newUsageRecord(c *fiber.Ctx, protocol string) *usage.RequestLog {
	record := &usage.RequestLog{
		TraceID:        mw.ExtractTraceID(c),
		CreatedAt:      time.Now().UTC(),
		Method:         c.Method(),
		Path:           c.Path(),
		Protocol:       protocol,
		VirtualKeyHash: stringLocal(c, "virtual_key_hash"),
		User:           stringLocal(c, "virtual_key_user"),
		APIKey:         stringLocal(c, "virtual_key_api_key"),
		ClientIP:       requestClientIP(c),
		RequestBody:    truncateBody(string(c.Body())),
		Provider:       "unknown",
		StatusCode:     fiber.StatusOK,
	}
	if record.User == "" {
		record.User = "unknown"
	}
	return record
}

func finishUsageRecord(c *fiber.Ctx, store *usage.Store, start time.Time, record *usage.RequestLog) {
	if store == nil || record == nil || record.StreamProcessing {
		return
	}
	status := c.Response().StatusCode()
	if record.StatusCode != 0 && status == fiber.StatusOK && record.StatusCode >= fiber.StatusBadRequest {
		status = record.StatusCode
	}
	finishUsageRecordWithStatus(store, start, record, status)
}

func finishUsageRecordWithStatus(store *usage.Store, start time.Time, record *usage.RequestLog, status int) {
	if store == nil || record == nil || record.Recorded {
		return
	}
	record.Recorded = true
	record.StreamCompleted = true
	record.StatusCode = status
	record.LatencyMS = time.Since(start).Milliseconds()
	applyPerformanceFields(record)
	if record.TTFTMS == 0 && !record.Stream && record.StatusCode < fiber.StatusBadRequest {
		record.TTFTMS = record.LatencyMS
	}
	if record.TotalTokens == 0 {
		record.TotalTokens = record.PromptTokens + record.OutputTokens
	}
	if record.ErrorCode == "" && status >= fiber.StatusBadRequest {
		record.ErrorCode = defaultErrorCode(status)
	}
	if err := store.Record(context.Background(), *record); err != nil {
		fmt.Fprintf(os.Stderr, "usage record failed: %v\n", err)
	}
}

func applyForwardInfo(record *usage.RequestLog, info forwarder.ResponseInfo) {
	if record == nil {
		return
	}
	record.Provider = info.Provider
	record.ProviderKeyID = info.ProviderKeyID
	record.UpstreamModel = info.UpstreamModel
	record.RouterDecision = info.RouterDecision
	record.FallbackCount = info.FallbackCount
}

func requestClientIP(c *fiber.Ctx) string {
	if c == nil {
		return ""
	}
	if forwarded := strings.TrimSpace(c.Get("X-Forwarded-For")); forwarded != "" {
		parts := strings.Split(forwarded, ",")
		if len(parts) > 0 && strings.TrimSpace(parts[0]) != "" {
			return strings.TrimSpace(parts[0])
		}
	}
	if realIP := strings.TrimSpace(c.Get("X-Real-IP")); realIP != "" {
		return realIP
	}
	return c.IP()
}

func truncateBody(body string) string {
	return truncateText(body, 16000)
}

func truncateText(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

type ttftMarkWriter struct {
	w      io.Writer
	mark   func()
	marked bool
}

func (w *ttftMarkWriter) Write(p []byte) (int, error) {
	if !w.marked && len(p) > 0 {
		w.marked = true
		if w.mark != nil {
			w.mark()
		}
	}
	return w.w.Write(p)
}

type limitedBodyCapture struct {
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func (c *limitedBodyCapture) Write(p []byte) (int, error) {
	if c == nil {
		return len(p), nil
	}
	if c.limit <= 0 {
		return len(p), nil
	}
	remaining := c.limit - c.buf.Len()
	if remaining <= 0 {
		c.truncated = true
		return len(p), nil
	}
	if len(p) > remaining {
		_, _ = c.buf.Write(p[:remaining])
		c.truncated = true
		return len(p), nil
	}
	_, _ = c.buf.Write(p)
	return len(p), nil
}

func (c *limitedBodyCapture) String() string {
	if c == nil {
		return ""
	}
	s := c.buf.String()
	if c.truncated {
		if s != "" {
			return s + "\n... [truncated]"
		}
		return "... [truncated]"
	}
	return s
}

func stringLocal(c *fiber.Ctx, key string) string {
	if value, ok := c.Locals(key).(string); ok {
		return value
	}
	return ""
}

func protocolFromPath(path string) string {
	switch path {
	case "/v1/chat/completions":
		return "openai_chat"
	case "/v1/messages":
		return "anthropic_messages"
	case "/v1/responses":
		return "responses"
	default:
		return ""
	}
}

func defaultErrorCode(status int) string {
	switch status {
	case fiber.StatusUnauthorized:
		return "unauthorized"
	case fiber.StatusForbidden:
		return "forbidden"
	case fiber.StatusNotFound:
		return "model_not_found"
	case fiber.StatusTooManyRequests:
		return "rate_limited"
	case fiber.StatusBadGateway, fiber.StatusServiceUnavailable:
		return "gateway_error"
	default:
		if status >= 500 {
			return "provider_error"
		}
		return "request_error"
	}
}

func firstOpenAIFinishReason(choices []models.OpenAIChoice) string {
	if len(choices) == 0 {
		return ""
	}
	return choices[0].FinishReason
}

func dashboardAuthorized(c *fiber.Ctx, cfg *config.Config) bool {
	if cfg.Dashboard.Token == "" {
		return true
	}
	token := c.Get("X-Dashboard-Token")
	if token == "" {
		token = c.Query("token")
	}
	return token == cfg.Dashboard.Token
}

func requireDashboardToken(cfg *config.Config) fiber.Handler {
	return func(c *fiber.Ctx) error {
		if !dashboardAuthorized(c, cfg) {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{"error": "invalid dashboard token"})
		}
		return c.Next()
	}
}

func registerDashboard(app *fiber.App, cfg *config.Config, store *usage.Store, authService *auth.Auth, r *router.Router) {
	app.Get("/dashboard", func(c *fiber.Ctx) error {
		c.Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
		c.Set("Content-Type", "text/html; charset=utf-8")
		if !dashboardAuthorized(c, cfg) {
			return c.SendString(dashboardTokenHTML(cfg.Dashboard.Token))
		}
		return c.SendString(dashboardHTML())
	})

	app.Get("/dashboard/requests/:trace_id", func(c *fiber.Ctx) error {
		c.Set("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
		c.Set("Content-Type", "text/html; charset=utf-8")
		if !dashboardAuthorized(c, cfg) {
			return c.SendString(dashboardTokenHTML(cfg.Dashboard.Token))
		}
		return c.SendString(requestDetailHTML(c.Params("trace_id")))
	})

	api := app.Group("/dashboard/api", requireDashboardToken(cfg))
	api.Get("/summary", func(c *fiber.Ctx) error {
		hours, _ := strconv.Atoi(c.Query("hours", "24"))
		data, err := store.Summary(c.Context(), hours)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Get("/error-breakdown", func(c *fiber.Ctx) error {
		hours, _ := strconv.Atoi(c.Query("hours", "24"))
		data, err := store.ErrorBreakdown(c.Context(), hours)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Get("/daily", func(c *fiber.Ctx) error {
		days, _ := strconv.Atoi(c.Query("days", "14"))
		data, err := store.Daily(c.Context(), days)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Get("/usage", func(c *fiber.Ctx) error {
		days, _ := strconv.Atoi(c.Query("days", "14"))
		limit, _ := strconv.Atoi(c.Query("limit", "20"))
		data, err := store.UsageDashboard(c.Context(), days, limit)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Get("/requests", func(c *fiber.Ctx) error {
		data, err := store.Requests(c.Context(), requestFilterFromQuery(c))
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Get("/requests/:trace_id", func(c *fiber.Ctx) error {
		row, err := store.RequestByTraceID(c.Context(), c.Params("trace_id"))
		if err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "request not found"})
			}
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(row)
	})
	api.Get("/requests.csv", func(c *fiber.Ctx) error {
		rows, err := store.Requests(c.Context(), requestFilterFromQuery(c))
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		c.Set("Content-Type", "text/csv; charset=utf-8")
		c.Set("Content-Disposition", "attachment; filename=requests.csv")
		var sb strings.Builder
		w := csv.NewWriter(&sb)
		_ = w.Write([]string{"trace_id", "created_at", "user", "api_key", "status_code", "model", "provider", "provider_key_id", "prompt_tokens", "output_tokens", "cost_cents", "router_decision", "fallback_count", "latency_ms", "error_code", "error_message"})
		for _, row := range rows {
			_ = w.Write([]string{row.TraceID, row.CreatedAt.Format(time.RFC3339Nano), row.User, row.APIKey, strconv.Itoa(row.StatusCode), row.Model, row.Provider, row.ProviderKeyID, strconv.Itoa(row.PromptTokens), strconv.Itoa(row.OutputTokens), strconv.FormatInt(row.CostCents, 10), row.RouterDecision, strconv.Itoa(row.FallbackCount), strconv.FormatInt(row.LatencyMS, 10), row.ErrorCode, row.ErrorMessage})
		}
		w.Flush()
		return c.SendString(sb.String())
	})
	api.Get("/keys", func(c *fiber.Ctx) error {
		hours, _ := strconv.Atoi(c.Query("hours", "24"))
		stats, err := store.KeyStats(c.Context(), hours)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"keys": r.AllKeyStatuses(), "stats": stats})
	})
	api.Get("/providers", func(c *fiber.Ctx) error {
		hours, _ := strconv.Atoi(c.Query("hours", "24"))
		stats, err := store.KeyStats(c.Context(), hours)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"keys": r.AllKeyStatuses(), "stats": stats})
	})
	api.Post("/keys/:provider/:key/:action", func(c *fiber.Ctx) error {
		if !r.SetKeyState(c.Params("provider"), c.Params("key"), c.Params("action")) {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid key action"})
		}
		return c.JSON(fiber.Map{"ok": true})
	})
	api.Get("/virtual-keys", func(c *fiber.Ctx) error {
		keys, err := store.ListVirtualKeys(c.Context())
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(keys)
	})
	api.Get("/keys-overview", func(c *fiber.Ctx) error {
		keys, err := store.ListVirtualKeys(c.Context())
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(fiber.Map{"virtual_keys": keys})
	})
	api.Post("/virtual-keys", func(c *fiber.Ctx) error {
		var req usage.CreateVirtualKeyRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
		}
		key, apiKey, err := store.CreateVirtualKey(c.Context(), req)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		authService.AddOrUpdateByHash(auth.VirtualKeyInfo{
			KeyHash:              key.KeyHash,
			User:                 key.User,
			APIKey:               key.APIKey,
			AllowedModels:        key.AllowedModels,
			RPMLimit:             key.RPMLimit,
			TPMLimit:             key.TPMLimit,
			ConcurrencyLimit:     key.ConcurrencyLimit,
			DailySpendLimitCents: key.DailySpendLimitCents,
		})
		return c.JSON(fiber.Map{"id": key.ID, "api_key": apiKey, "key": key})
	})
	api.Post("/virtual-keys/:id/:action", func(c *fiber.Ctx) error {
		id, _ := strconv.ParseInt(c.Params("id"), 10, 64)
		key, err := store.SetVirtualKeyState(c.Context(), id, c.Params("action"))
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
		}
		if key.Enabled && key.RevokedAt == nil {
			authService.AddOrUpdateByHash(auth.VirtualKeyInfo{
				KeyHash:              key.KeyHash,
				User:                 key.User,
				APIKey:               key.APIKey,
				AllowedModels:        key.AllowedModels,
				RPMLimit:             key.RPMLimit,
				TPMLimit:             key.TPMLimit,
				ConcurrencyLimit:     key.ConcurrencyLimit,
				DailySpendLimitCents: key.DailySpendLimitCents,
			})
		} else {
			authService.RemoveByHash(key.KeyHash)
		}
		return c.JSON(fiber.Map{"ok": true})
	})
	api.Get("/errors", func(c *fiber.Ctx) error {
		hours, _ := strconv.Atoi(c.Query("hours", "24"))
		limit, _ := strconv.Atoi(c.Query("limit", "100"))
		data, err := store.ErrorDashboard(c.Context(), hours, limit)
		if err != nil {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": err.Error()})
		}
		return c.JSON(data)
	})
	api.Post("/reload", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"ok": true})
	})
}

func requestFilterFromQuery(c *fiber.Ctx) usage.RequestFilter {
	limit, _ := strconv.Atoi(c.Query("limit", "100"))
	return usage.RequestFilter{
		Provider:      c.Query("provider"),
		ProviderKeyID: c.Query("provider_key_id"),
		Model:         c.Query("model"),
		User:          c.Query("user"),
		Status:        c.Query("status"),
		Limit:         limit,
	}
}

func dashboardTokenHTML(token string) string {
	masked := auth.MaskKey(token)
	return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM Gateway Dashboard</title><style>
body{font-family:Arial,sans-serif;margin:0;background:#f6f7f9;color:#20242a;min-height:100vh;display:flex;align-items:center;justify-content:center}.panel{width:min(640px,calc(100vw - 24px));background:#fff;border:1px solid #dde1e7;border-radius:8px;padding:20px}.title{font-size:22px;font-weight:700;margin:0 0 8px}.desc{color:#657080;font-size:13px;line-height:1.5;margin:0 0 16px}.tokenbox{border:1px solid #dde1e7;border-radius:6px;background:#f8fafc;padding:12px 14px;margin:14px 0 10px}.label{font-size:12px;color:#657080;margin-bottom:6px}.token{font-family:Consolas,monospace;font-size:14px;word-break:break-all}.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}.btn{height:36px;border:1px solid #20242a;background:#20242a;color:#fff;border-radius:6px;padding:0 14px;cursor:pointer}.btn.secondary{background:#fff;color:#20242a}.note{margin-top:12px;font-size:12px;color:#657080;word-break:break-all}.err{color:#b42318}.ok{color:#16794c}</style></head><body><div class="panel"><div class="title">LLM Gateway Dashboard</div><p class="desc">This page shows the current dashboard token. Click open to enter the dashboard, or copy the token into other devices if needed.</p><div class="tokenbox"><div class="label">Current token</div><div class="token" id="tokenText">` + masked + `</div></div><div class="row"><button id="open" class="btn">Open Dashboard</button><button id="copy" class="btn secondary">Copy Token</button></div><div id="msg" class="note"></div></div><script>
const fullToken = ` + strconv.Quote(token) + `;const el=id=>document.getElementById(id);function setMsg(text,err){const m=el('msg');m.textContent=text;m.className='note '+(err?'err':'ok');}function go(token){localStorage.setItem('llm_gateway_dashboard_token', token); location.href='/dashboard?token='+encodeURIComponent(token);}el('open').addEventListener('click',()=>go(fullToken));el('copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(fullToken); setMsg('Token copied.', false);}catch(e){setMsg('Copy failed: '+e.message, true);}});setTimeout(()=>go(fullToken),0);</script><script>
(function(){
  function body(label, value){return '<div class="detail-group"><div class="detail-title">'+label+'</div><pre style="margin:0;padding:10px 12px;border:1px solid var(--line2);border-radius:6px;background:#fafbfc;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word;max-height:280px;overflow:auto" class="mono">'+esc(value||'')+'</pre></div>'}
  function renderDetailV2(r){
    const box=el('requestDetail'); if(!box) return;
    if(!r){ box.innerHTML='<div class="hint">Select a request to inspect details.</div>'; return; }
    box.innerHTML = '<div class="detail-head"><div><div class="detail-title">'+esc(r.model||'Request')+'</div><div class="detail-sub mono">'+esc(r.trace_id)+'</div></div><div class="detail-sub">'+dt(r.created_at)+'</div></div>'
      + group('Basic Info',[field('ID',esc(r.trace_id)),field('Time',dt(r.created_at)),field('Model',esc(r.model)),field('Protocol',esc(r.protocol)),field('Status Code',v(r.status_code)),field('Stream',r.stream?'yes':'no'),field('Has Image',r.has_image?'yes':'no'),field('Total Time',v(r.latency_ms)+' ms'),field('First Token',v(r.ttft_ms)+' ms'),field('Request Cost',money(v(r.cost_cents))),field('Token Rate',v(r.output_tokens_per_second)),field('Client IP',esc(r.client_ip)),field('User API Key',esc(r.api_key))])
      + group('Usage',[field('Prompt',v(r.prompt_tokens)),field('Completion',v(r.output_tokens)),field('Total',v(r.total_tokens)),field('Cache Read',v(r.cache_read_tokens)),field('Cache Write',v(r.cache_write_tokens)),field('Cache Hit',pct(r.cache_hit_rate)),field('Finish Reason',esc(r.finish_reason)),field('Prompt Bucket',esc(r.prompt_bucket))])
      + group('Routing',[field('User',esc(r.user)),field('Provider',esc(r.provider)),field('Provider Key',esc(r.provider_key_id)),field('Upstream Model',esc(r.upstream_model)),field('Router Decision',esc(r.router_decision)),field('Fallback Count',v(r.fallback_count)),field('Slow Reason',esc(r.slow_reason)),field('Generation',v(r.generation_ms)+' ms')])
      + body('Request Body', r.request_body)
      + body('Response Body', r.response_body)
      + group('Error',[field('Error Code',esc(r.error_code)),field('Error Message',esc(r.error_message))]);
  }
  renderDetail = renderDetailV2;
  if (typeof load === 'function') { load().catch(function(){}); }
})();
</script></body></html>`
}

func dashboardHTML() string {
	return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM Gateway</title><style>
/* ─── Design Tokens ─── */
:root{
  --bg:#fafafa;--surface:#fff;--surface-2:#f5f5f5;--surface-3:#f0f0f0;
  --text:#111;--text-2:#666;--text-3:#999;--border:#e5e5e5;--border-2:#eee;
  --accent:#0070f3;--accent-2:#0060df;--green:#17c964;--green-bg:#f0fdf4;
  --amber:#f5a623;--amber-bg:#fffbeb;--red:#ef4444;--red-bg:#fef2f2;
  --radius:12px;--radius-sm:8px;--shadow:0 1px 2px rgba(0,0,0,.04),0 1px 3px rgba(0,0,0,.02);
  --shadow-lg:0 4px 24px rgba(0,0,0,.06);--font-sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,"Helvetica Neue",sans-serif;
  --font-mono:"SF Mono",Monaco,Consolas,"Liberation Mono","Courier New",monospace;
  --sidebar-w:240px;--topbar-h:56px;
}
*{box-sizing:border-box;margin:0}
html,body{height:100%;font-family:var(--font-sans);font-size:14px;line-height:1.5;color:var(--text);background:var(--bg);-webkit-font-smoothing:antialiased}

/* ─── Layout ─── */
.app{min-height:100vh;display:flex}
.sidebar{width:var(--sidebar-w);background:var(--surface);border-right:1px solid var(--border);position:fixed;inset:0 auto 0 0;display:flex;flex-direction:column;z-index:50}
.sidebar-head{padding:20px 20px 12px}
.sidebar-brand{font-size:15px;font-weight:650;letter-spacing:-.01em;display:flex;align-items:center;gap:8px}
.sidebar-brand svg{width:20px;height:20px;color:var(--accent)}
.sidebar-nav{flex:1;padding:0 12px 16px;overflow-y:auto}
.nav-item{width:100%;display:flex;align-items:center;gap:10px;padding:8px 12px;border-radius:var(--radius-sm);border:none;background:transparent;color:var(--text-2);font-size:13px;cursor:pointer;transition:all .15s;margin:2px 0}
.nav-item:hover{background:var(--surface-2);color:var(--text)}
.nav-item.active{background:var(--surface-3);color:var(--text);font-weight:500}
.nav-item svg{width:16px;height:16px;flex-shrink:0;opacity:.7}
.nav-item.active svg{opacity:1}
.sidebar-foot{padding:16px 20px;border-top:1px solid var(--border);font-size:12px;color:var(--text-3);line-height:1.6}
.main{flex:1;margin-left:var(--sidebar-w);min-width:0}
.topbar{height:var(--topbar-h);border-bottom:1px solid var(--border);background:var(--surface);display:flex;align-items:center;justify-content:space-between;padding:0 28px;position:sticky;top:0;z-index:40}
.topbar-left{display:flex;align-items:center;gap:16px}
.page-title{font-size:18px;font-weight:600;letter-spacing:-.01em}
.env-badge{font-size:11px;font-weight:500;padding:3px 8px;border-radius:999px;background:var(--green-bg);color:var(--green);border:1px solid rgba(23,201,100,.2)}
.topbar-right{display:flex;align-items:center;gap:10px}
.btn-icon{width:32px;height:32px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);display:inline-flex;align-items:center;justify-content:center;cursor:pointer;color:var(--text-2);transition:all .15s}
.btn-icon:hover{border-color:var(--text-3);color:var(--text)}
.btn-icon svg{width:16px;height:16px}
.btn{height:32px;padding:0 14px;border-radius:var(--radius-sm);border:1px solid var(--border);background:var(--surface);font-size:13px;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;gap:6px;color:var(--text);transition:all .15s;text-decoration:none}
.btn:hover{border-color:var(--text-3)}
.btn-primary{background:var(--text);color:#fff;border-color:var(--text)}
.btn-primary:hover{background:#333}
.content{padding:24px 28px 40px;max-width:1400px}

/* ─── Status Indicators ─── */
.status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
.status-ok{background:var(--green);box-shadow:0 0 0 3px rgba(23,201,100,.15)}
.status-warn{background:var(--amber);box-shadow:0 0 0 3px rgba(245,166,35,.15)}
.status-bad{background:var(--red);box-shadow:0 0 0 3px rgba(239,68,68,.15)}
.status-offline{background:var(--text-3);box-shadow:0 0 0 3px rgba(153,153,153,.15)}

/* ─── Cards ─── */
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);overflow:hidden}
.card-pad{padding:20px}
.card-head{display:flex;align-items:center;justify-content:space-between;padding:16px 20px;border-bottom:1px solid var(--border-2)}
.card-title{font-size:14px;font-weight:600}
.card-sub{font-size:13px;color:var(--text-2);font-weight:400}
.card-body{padding:16px 20px}

/* ─── Metric Cards ─── */
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:24px}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow);transition:box-shadow .2s}
.metric-card:hover{box-shadow:var(--shadow-lg)}
.metric-label{font-size:12px;color:var(--text-3);font-weight:500;text-transform:uppercase;letter-spacing:.03em;margin-bottom:8px;display:flex;align-items:center;gap:6px}
.metric-value{font-size:28px;font-weight:650;letter-spacing:-.02em;line-height:1.2;margin-bottom:4px}
.metric-hint{font-size:12px;color:var(--text-2)}
.metric-delta{font-size:12px;font-weight:500;margin-left:auto}
.metric-delta.up{color:var(--green)}
.metric-delta.down{color:var(--red)}

/* ─── Health Cards ─── */
.health-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-bottom:24px}
.health-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow);display:flex;align-items:flex-start;gap:14px}
.health-icon{width:40px;height:40px;border-radius:var(--radius-sm);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.health-icon.ok{background:var(--green-bg);color:var(--green)}
.health-icon.warn{background:var(--amber-bg);color:var(--amber)}
.health-icon.bad{background:var(--red-bg);color:var(--red)}
.health-icon svg{width:20px;height:20px}
.health-content{flex:1;min-width:0}
.health-title{font-size:14px;font-weight:600;margin-bottom:2px}
.health-desc{font-size:13px;color:var(--text-2);line-height:1.5}
.health-meta{font-size:12px;color:var(--text-3);margin-top:6px}

/* ─── Provider Cards ─── */
.provider-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;margin-bottom:24px}
.provider-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:20px;box-shadow:var(--shadow);cursor:pointer;transition:all .2s}
.provider-card:hover{box-shadow:var(--shadow-lg);border-color:var(--text-3)}
.provider-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.provider-name{font-size:15px;font-weight:600;display:flex;align-items:center;gap:8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.provider-name .status-dot{flex-shrink:0}
.provider-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.provider-stat{text-align:center}
.provider-stat-value{font-size:18px;font-weight:600}
.provider-stat-label{font-size:11px;color:var(--text-3);margin-top:2px;text-transform:uppercase;letter-spacing:.03em}
.provider-bar{height:4px;background:var(--surface-2);border-radius:999px;margin-top:16px;overflow:hidden}
.provider-bar-fill{height:100%;border-radius:999px;transition:width .5s ease}
.provider-table-row{display:grid;grid-template-columns:200px 80px 100px 100px 100px 80px;gap:12px;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border-2);font-size:13px}
.provider-table-row:hover{background:var(--surface-2)}
.provider-table-head{font-size:11px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.03em;padding:10px 16px;border-bottom:1px solid var(--border-2);display:grid;grid-template-columns:200px 80px 100px 100px 100px 80px;gap:12px}

/* ─── Tables ─── */
.table-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:13px}
thaad,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--border-2);white-space:nowrap}
thaad{font-size:11px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:.03em;background:var(--surface-2)}
tbody tr{cursor:pointer;transition:background .1s}
tbody tr:hover{background:var(--surface-2)}
tbody tr.active{background:var(--surface-3)}
.cell-mono{font-family:var(--font-mono);font-size:12px}
.cell-badge{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:500}
.badge-green{background:var(--green-bg);color:var(--green)}
.badge-amber{background:var(--amber-bg);color:var(--amber)}
.badge-red{background:var(--red-bg);color:var(--red)}
.badge-gray{background:var(--surface-3);color:var(--text-2)}

/* ─── Charts ─── */
.chart-container{height:180px;position:relative;margin-top:12px}
.chart-bar{display:flex;align-items:flex-end;gap:3px;height:100%;padding:0 4px}
.chart-bar-item{flex:1;background:var(--accent);border-radius:3px 3px 0 0;opacity:.7;transition:opacity .2s;min-width:4px}
.chart-bar-item:hover{opacity:1}
.chart-labels{display:flex;gap:3px;padding:0 4px;margin-top:6px}
.chart-label{flex:1;text-align:center;font-size:10px;color:var(--text-3);min-width:4px;overflow:hidden;text-overflow:ellipsis}
.chart-empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-3);font-size:13px}

/* ─── Drawers ─── */
.drawer{position:fixed;inset:0 0 0 auto;width:min(520px,100vw);background:var(--surface);box-shadow:var(--shadow-lg);z-index:100;transform:translateX(100%);transition:transform .3s cubic-bezier(.16,1,.3,1);display:flex;flex-direction:column}
.drawer.open{transform:translateX(0)}
.drawer-overlay{position:fixed;inset:0;background:rgba(0,0,0,.2);z-index:99;opacity:0;pointer-events:none;transition:opacity .3s}
.drawer-overlay.open{opacity:1;pointer-events:auto}
.drawer-head{padding:20px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.drawer-title{font-size:16px;font-weight:600}
.drawer-body{flex:1;overflow-y:auto;padding:20px 24px}
.drawer-foot{padding:16px 24px;border-top:1px solid var(--border);display:flex;gap:10px;justify-content:flex-end}

/* ─── Filters ─── */
.filter-bar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}
.filter-bar input,.filter-bar select{height:36px;padding:0 12px;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);font-size:13px;color:var(--text);outline:none;min-width:140px}
.filter-bar input:focus,.filter-bar select:focus{border-color:var(--text-3)}

/* ─── Tabs ─── */
.tabs{display:flex;gap:4px;border-bottom:1px solid var(--border);margin-bottom:20px}
.tab-btn{padding:8px 16px;border:none;background:transparent;font-size:13px;color:var(--text-2);cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;transition:all .15s}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--text);border-bottom-color:var(--text);font-weight:500}

/* ─── Timeline ─── */
.timeline{position:relative;padding-left:24px}
.timeline::before{content:'';position:absolute;left:7px;top:4px;bottom:4px;width:2px;background:var(--border-2);border-radius:999px}
.timeline-item{position:relative;padding-bottom:20px}
.timeline-item::before{content:'';position:absolute;left:-20px;top:4px;width:12px;height:12px;border-radius:50%;background:var(--surface);border:2px solid var(--border)}
.timeline-item.ok::before{border-color:var(--green)}
.timeline-item.warn::before{border-color:var(--amber)}
.timeline-item.bad::before{border-color:var(--red)}
.timeline-time{font-size:11px;color:var(--text-3);margin-bottom:4px}
.timeline-title{font-size:13px;font-weight:500;margin-bottom:2px}
.timeline-desc{font-size:12px;color:var(--text-2);line-height:1.5}

/* ─── Key Status Cards ─── */
.key-status-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:20px}
.key-status-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius-sm);padding:16px;text-align:center}
.key-status-count{font-size:24px;font-weight:650;margin-bottom:4px}
.key-status-count.ok{color:var(--green)}
.key-status-count.warn{color:var(--amber)}
.key-status-count.bad{color:var(--red)}
.key-status-label{font-size:12px;color:var(--text-3)}

/* ─── Misc ─── */
.empty-state{text-align:center;padding:60px 20px;color:var(--text-3)}
.empty-state svg{width:48px;height:48px;margin-bottom:16px;opacity:.4}
.empty-state-title{font-size:15px;font-weight:500;color:var(--text-2);margin-bottom:4px}
.section-title{font-size:13px;font-weight:600;color:var(--text-2);text-transform:uppercase;letter-spacing:.04em;margin-bottom:12px}
.divider{height:1px;background:var(--border);margin:24px 0}
.pre-wrap{font-family:var(--font-mono);font-size:12px;line-height:1.6;background:var(--surface-2);padding:16px;border-radius:var(--radius-sm);overflow-x:auto;white-space:pre-wrap;word-break:break-word;max-height:400px;overflow-y:auto}

/* ─── Views ─── */
.view{display:none}
.view.active{display:block}

/* ─── Responsive ─── */
@media(max-width:900px){
  .sidebar{transform:translateX(-100%);transition:transform .3s}
  .sidebar.open{transform:translateX(0)}
  .main{margin-left:0}
  .metric-grid{grid-template-columns:repeat(2,1fr)}
  .health-grid,.provider-grid{grid-template-columns:1fr}
}
@media(max-width:640px){
  .content{padding:16px}
  .metric-grid{grid-template-columns:1fr}
  .topbar{padding:0 16px}
  .filter-bar input,.filter-bar select{min-width:100px}
}
</style></head><body>

<div class="app">
<!-- Sidebar -->
<aside class="sidebar" id="sidebar">
  <div class="sidebar-head">
    <div class="sidebar-brand">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
      LLM Gateway
    </div>
  </div>
  <nav class="sidebar-nav">
    <button class="nav-item active" data-view="overview">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>
      Overview
    </button>
    <button class="nav-item" data-view="requests">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      Requests
    </button>
    <button class="nav-item" data-view="usage">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
      Usage
    </button>
    <button class="nav-item" data-view="providers">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
      Providers
    </button>
    <button class="nav-item" data-view="keys">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-1.5 3.5L19 4"/></svg>
      Virtual Keys
    </button>
    <button class="nav-item" data-view="errors">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      Errors
    </button>
  </nav>
  <div class="sidebar-foot">
    <div id="sideStatus">Loading...</div>
    <div style="margin-top:4px">v2.0</div>
  </div>
</aside>

<!-- Main -->
<main class="main">
  <!-- Topbar -->
  <header class="topbar">
    <div class="topbar-left">
      <div class="page-title" id="pageTitle">Overview</div>
      <span class="env-badge" id="envBadge">Production</span>
    </div>
    <div class="topbar-right">
      <button class="btn-icon" id="refreshBtn" title="Refresh">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 11-2.12-9.36L23 10"/></svg>
      </button>
      <a class="btn" id="exportBtn" href="#" style="display:none">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Export
      </a>
    </div>
  </header>

  <!-- Content -->
  <div class="content">
    <!-- Status Banner -->
    <div id="statusBanner" style="display:none;margin-bottom:20px"></div>

    <!-- OVERVIEW -->
    <section id="overview" class="view active">
      <!-- System Health -->
      <div class="section-title">System Health</div>
      <div class="health-grid" id="healthCards"></div>

      <!-- Core Metrics -->
      <div class="section-title">Core Metrics <span style="font-weight:400;color:var(--text-3)">· Last 24h</span></div>
      <div class="metric-grid" id="overviewMetrics"></div>

      <!-- Provider Health -->
      <div class="section-title">Providers</div>
      <div class="provider-grid" id="providerCards"></div>

      <!-- Usage Trend -->
      <div class="section-title">Usage Trend</div>
      <div class="card" style="margin-bottom:24px">
        <div class="card-body">
          <div class="chart-container" id="usageChart"></div>
        </div>
      </div>

      <!-- Recent Incidents -->
      <div class="section-title">Recent Incidents</div>
      <div class="card card-pad" id="incidentsCard">
        <div class="timeline" id="incidentsTimeline"></div>
      </div>
    </section>

    <!-- REQUESTS -->
    <section id="requests" class="view">
      <div class="metric-grid" id="requestMetrics" style="grid-template-columns:repeat(4,1fr)"></div>
      <div class="filter-bar">
        <input id="rProvider" placeholder="Provider">
        <input id="rKey" placeholder="Key ID">
        <input id="rModel" placeholder="Model">
        <input id="rUser" placeholder="User">
        <select id="rStatus"><option value="">All Status</option><option value="success">Success</option><option value="error">Error</option></select>
        <button class="btn btn-primary" id="rFilterBtn">Filter</button>
      </div>
      <div class="card">
        <div class="table-wrap">
          <table>
            <thead><tr>
              <th>Time</th><th>Trace</th><th>User</th><th>Status</th><th>Model</th><th>Prompt</th><th>Completion</th><th>Cache</th><th>Latency</th><th>Error</th>
            </tr></thead>
            <tbody id="requestRows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- USAGE -->
    <section id="usage" class="view">
      <div class="tabs">
        <button class="tab-btn active" data-period="today">Today</button>
        <button class="tab-btn" data-period="week">Week</button>
        <button class="tab-btn" data-period="month">Month</button>
      </div>
      <div class="metric-grid" id="usageMetrics"></div>
      <div class="card" style="margin-bottom:24px">
        <div class="card-head"><div class="card-title">Token Distribution</div></div>
        <div class="card-body">
          <div class="chart-container" id="tokenChart"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div class="card-title">Daily Usage</div></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Date</th><th>Requests</th><th>Errors</th><th>Prompt</th><th>Completion</th><th>Total</th><th>Cost</th></tr></thead>
            <tbody id="usageRows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- PROVIDERS -->
    <section id="providers" class="view">
      <div class="provider-grid" id="providerDetailCards"></div>
    </section>

    <!-- VIRTUAL KEYS -->
    <section id="keys" class="view">
      <div class="key-status-grid" id="keyStatusGrid"></div>
      <div class="card card-pad" style="margin-bottom:24px">
        <div class="filter-bar" style="margin-bottom:0">
          <input id="vkUser" placeholder="User" value="default">
          <input id="vkModels" placeholder="Models" value="kimi-for-coding,gpt-5.5">
          <input id="vkRpm" placeholder="RPM" value="100" style="width:80px">
          <input id="vkTpm" placeholder="TPM" value="200000" style="width:100px">
          <input id="vkConc" placeholder="Concurrency" value="10" style="width:100px">
          <button class="btn btn-primary" id="vkGenerateBtn">Generate Key</button>
        </div>
        <div id="vkGenerated" style="display:none;margin-top:12px"></div>
      </div>
      <div class="card">
        <div class="table-wrap">
          <table>
            <thead><tr><th>ID</th><th>Key</th><th>User</th><th>Models</th><th>RPM</th><th>TPM</th><th>Concurrency</th><th>Status</th><th>Last Used</th><th>Actions</th></tr></thead>
            <tbody id="vkRows"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ERRORS -->
    <section id="errors" class="view">
      <div class="metric-grid" id="errorMetrics" style="grid-template-columns:repeat(4,1fr)"></div>
      <div class="card" style="margin-bottom:24px">
        <div class="card-head"><div class="card-title">Error Trend</div></div>
        <div class="card-body">
          <div class="chart-container" id="errorChart"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-head"><div class="card-title">Recent Errors</div></div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Time</th><th>Trace</th><th>Status</th><th>Code</th><th>Message</th></tr></thead>
            <tbody id="errorRows"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
</main>
</div>

<!-- Drawer -->
<div class="drawer-overlay" id="drawerOverlay"></div>
<div class="drawer" id="drawer">
  <div class="drawer-head">
    <div class="drawer-title" id="drawerTitle">Request Detail</div>
    <button class="btn-icon" id="drawerClose">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>
  <div class="drawer-body" id="drawerBody"></div>
</div>

<script>
// ─── State ───
const token=new URLSearchParams(location.search).get('token')||localStorage.getItem('llm_gateway_dashboard_token')||'';
if(token)localStorage.setItem('llm_gateway_dashboard_token',token);
const h=token?{'X-Dashboard-Token':token}:{};
const el=id=>document.getElementById(id);
let lastRows=[],currentPeriod='today';

function api(path){const u=new URL(path,location.origin);if(token)u.searchParams.set('token',token);return u.pathname+u.search}
function qs(){const u=new URLSearchParams();['provider','provider_key_id','model','user','status'].forEach(id=>{const n=el('r'+id.charAt(0).toUpperCase()+id.slice(1));if(!n)return;const v=n?n.value:'';if(v)u.set(id,v)});u.set('limit','100');if(token)u.set('token',token);return u}
function money(c){return'$'+((c||0)/100).toFixed(4)}
function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function v(x,d){return x==null?(d==null?0:d):x}
function num(x){return Number(x)||0}
function dt(x){return x?new Date(x).toLocaleString('en-US',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):''}
function shortTrace(x){return x?String(x).slice(0,8):''}
function pct(x){return x==null?'0%':(Math.round(num(x)*1000)/10)+'%'}
function compact(n){n=num(n);if(n>=1e9)return(n/1e9).toFixed(1)+'B';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return String(Math.round(n))}
function jsonHeaders(){return Object.assign({'Content-Type':'application/json'},h)}
async function j(url,opt){const r=await fetch(url,opt||{headers:h});const text=await r.text();if(!r.ok)throw new Error(text||r.statusText);return text?JSON.parse(text):{}}

// ─── Navigation ───
function setView(name){document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===name));document.querySelectorAll('.nav-item').forEach(x=>x.classList.toggle('active',x.dataset.view===name));el('pageTitle').textContent=name.charAt(0).toUpperCase()+name.slice(1);el('exportBtn').style.display=name==='requests'?'inline-flex':'none'}
document.querySelectorAll('.nav-item').forEach(n=>n.addEventListener('click',()=>setView(n.dataset.view)));

// ─── Drawer ───
function openDrawer(title,html){el('drawerTitle').textContent=title;el('drawerBody').innerHTML=html;el('drawer').classList.add('open');el('drawerOverlay').classList.add('open')}
function closeDrawer(){el('drawer').classList.remove('open');el('drawerOverlay').classList.remove('open')}
el('drawerClose').addEventListener('click',closeDrawer);el('drawerOverlay').addEventListener('click',closeDrawer);

// ─── Render Helpers ───
function statusBadge(code){if(!code||code>=400)return'<span class="cell-badge badge-red">'+code+'</span>';if(code>=300)return'<span class="cell-badge badge-amber">'+code+'</span>';return'<span class="cell-badge badge-green">'+code+'</span>'}
function stateBadge(state){const map={healthy:['badge-green','Healthy'],active:['badge-green','Active'],cooldown:['badge-amber','Cooldown'],disabled:['badge-red','Disabled'],revoked:['badge-red','Revoked']};const[m,l]=map[state]||['badge-gray',state];return'<span class="cell-badge '+m+'">'+(l||state)+'</span>'}
function healthCard(icon,status,title,desc,meta){const cls=status==='ok'?'ok':status==='warn'?'warn':'bad';return'<div class="health-card"><div class="health-icon '+cls+'">'+icon+'</div><div class="health-content"><div class="health-title">'+title+'</div><div class="health-desc">'+desc+'</div><div class="health-meta">'+meta+'</div></div></div>'}
function metricCard(label,value,hint,delta){return'<div class="metric-card"><div class="metric-label">'+label+'</div><div class="metric-value">'+value+'</div><div class="metric-hint">'+hint+(delta?'<span class="metric-delta '+delta.cls+'">'+delta.text+'</span>':'')+'</div></div>'}

// ─── Overview ───
function renderOverview(s,rows,keydata,errors,usage){const avgCache=rows.length?rows.reduce((a,r)=>a+num(r.cache_hit_rate),0)/rows.length:0;const slow=rows.filter(r=>r.slow_reason||num(r.status_code)>=400).length;const errorRate=rows.length?Math.round(slow/rows.length*1000)/10:0;

// Health Cards
const providers=(keydata.keys||[]).map(k=>k.provider).filter((v,i,a)=>a.indexOf(v)===i);
const healthyKeys=(keydata.keys||[]).filter(k=>k.state==='healthy').length;
const totalKeys=(keydata.keys||[]).length;
const exhaustedKeys=(keydata.keys||[]).filter(k=>k.state==='disabled').length;

el('healthCards').innerHTML=
  healthCard('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>','ok','Gateway Online','All systems operational','Uptime · 99.9%')+
  healthCard('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',exhaustedKeys>0?'warn':'ok','Providers',''+providers.join(', ')+'',''+healthyKeys+'/'+totalKeys+' keys healthy')+
  healthCard('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>',avgCache<0.5?'warn':'ok','Cache Status','Hit rate '+pct(avgCache)+'',''+compact(s.cache_hits||0)+' hits / '+compact(s.cache_misses||0)+' misses')+
  (exhaustedKeys>0?healthCard('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>','bad',''+exhaustedKeys+' Keys Exhausted','Provider keys need attention','Review in Virtual Keys'):'');

// Metrics
el('overviewMetrics').innerHTML=
  metricCard('Requests',compact(s.requests),'last 24h',{cls:errorRate>5?'up':'down',text:errorRate+'% error'})+
  metricCard('Prompt Tokens',compact(s.prompt_tokens),'input',{cls:'down',text:''})+
  metricCard('Completion',compact(s.output_tokens),'output',{cls:'down',text:''})+
  metricCard('Avg Latency',compact(s.avg_latency_ms)+' ms','24h',{cls:'down',text:''})+
  metricCard('Avg TTFT',compact(s.avg_ttft_ms)+' ms','24h',{cls:'down',text:''})+
  metricCard('Cache Hit',pct(avgCache),'recent',{cls:'down',text:''})+
  metricCard('Cost',money(s.cost_cents),'total',{cls:'down',text:''})+
  metricCard('Slow/Error',compact(slow),'recent 100',{cls:slow>10?'up':'down',text:''});

// Provider Cards - use compact table when > 4 providers
const stats={};(keydata.stats||[]).forEach(x=>stats[x.provider+'/'+x.provider_key_id]=x);
const providerKeys = keydata.keys || [];
if (providerKeys.length > 4) {
  // Compact table view
  el('providerCards').innerHTML = '<div class="card"><div class="card-head"><div class="card-title">Provider Keys</div></div><div class="card-body" style="padding:0">' +
    '<div class="provider-table-head"><div>Provider/Key</div><div>State</div><div>Success</div><div>Tokens</div><div>Requests</div><div>Actions</div></div>' +
    providerKeys.map(k => {
      const st = stats[k.provider + '/' + k.id] || {};
      const rate = Math.round(num(st.success_rate) * 1000) / 10;
      const cls = k.state === 'healthy' ? 'ok' : k.state === 'cooldown' ? 'warn' : 'bad';
      return '<div class="provider-table-row">' +
        '<div style="display:flex;align-items:center;gap:8px"><span class="status-dot status-' + cls + '"></span><span style="font-weight:500">' + esc(k.provider) + ' / ' + esc(k.id) + '</span></div>' +
        '<div>' + stateBadge(k.state) + '</div>' +
        '<div style="font-weight:600">' + rate + '%</div>' +
        '<div>' + compact(st.total_tokens || 0) + '</div>' +
        '<div>' + v(st.requests) + '</div>' +
        '<div><button class="btn" data-p="' + esc(k.provider) + '" data-k="' + esc(k.id) + '" data-a="enable">Enable</button></div>' +
      '</div>';
    }).join('') + '</div></div>';
} else {
  // Card view
  el('providerCards').innerHTML = providerKeys.map(k => {
    const st = stats[k.provider + '/' + k.id] || {};
    const rate = Math.round(num(st.success_rate) * 1000) / 10;
    const cls = k.state === 'healthy' ? 'ok' : k.state === 'cooldown' ? 'warn' : 'bad';
    return '<div class="provider-card" data-provider="' + esc(k.provider) + '" data-key="' + esc(k.id) + '">' +
      '<div class="provider-top"><div class="provider-name"><span class="status-dot status-' + cls + '"></span>' + esc(k.provider) + ' / ' + esc(k.id) + '</div>' + stateBadge(k.state) + '</div>' +
      '<div class="provider-stats">' +
        '<div class="provider-stat"><div class="provider-stat-value">' + rate + '%</div><div class="provider-stat-label">Success</div></div>' +
        '<div class="provider-stat"><div class="provider-stat-value">' + compact(st.total_tokens || 0) + '</div><div class="provider-stat-label">Tokens</div></div>' +
        '<div class="provider-stat"><div class="provider-stat-value">' + v(st.requests) + '</div><div class="provider-stat-label">Requests</div></div>' +
      '</div>' +
      '<div class="provider-bar"><div class="provider-bar-fill" style="width:' + rate + '%;background:' + (rate > 90 ? 'var(--green)' : rate > 70 ? 'var(--amber)' : 'var(--red)') + '"></div></div>' +
    '</div>';
  }).join('') || '<div class="empty-state"><div class="empty-state-title">No providers</div></div>';
}

// Usage Chart
renderBarChart('usageChart',((usage&&usage.daily)||[]).slice(-14).map(d=>({label:d.date.slice(5),value:d.requests})));

// Incidents
const incidents=(errors.error_requests||[]).slice(0,5).map(r=>({time:r.created_at,title:r.error_code||'Error',desc:(r.error_message||'').slice(0,100),status:'bad'}));
el('incidentsTimeline').innerHTML=incidents.length?incidents.map(i=>'<div class="timeline-item '+i.status+'"><div class="timeline-time">'+dt(i.time)+'</div><div class="timeline-title">'+esc(i.title)+'</div><div class="timeline-desc">'+esc(i.desc)+'</div></div>').join(''):'<div class="empty-state"><div class="empty-state-title">No incidents</div><div>System is running smoothly</div></div>';

el('sideStatus').textContent=compact(s.requests)+' requests · '+compact(s.errors)+' errors';
}

// ─── Charts ───
function renderBarChart(id,data){const max=Math.max(...data.map(d=>d.value),1);const elChart=el(id);if(!elChart)return;if(!data.length){elChart.innerHTML='<div class="chart-empty">No data available</div>';return}elChart.innerHTML='<div class="chart-bar">'+data.map(d=>'<div class="chart-bar-item" style="height:'+(d.value/max*100)+'%" title="'+d.label+': '+compact(d.value)+'"></div>').join('')+'</div><div class="chart-labels">'+data.map(d=>'<div class="chart-label">'+d.label+'</div>').join('')+'</div>';}

// ─── Requests ───
function renderRequests(rows){lastRows=rows||[];
const success=lastRows.filter(r=>r.status_code<400).length;
const errors=lastRows.filter(r=>r.status_code>=400).length;
const avgLatency=lastRows.length?Math.round(lastRows.reduce((a,r)=>a+num(r.latency_ms),0)/lastRows.length):0;
const avgTTFT=lastRows.length?Math.round(lastRows.reduce((a,r)=>a+num(r.ttft_ms),0)/lastRows.filter(r=>r.ttft_ms).length||1):0;
el('requestMetrics').innerHTML=
  metricCard('Total',compact(lastRows.length),'requests')+
  metricCard('Success Rate',pct(success/lastRows.length),'last 100')+
  metricCard('Avg Latency',avgLatency+' ms','p50')+
  metricCard('Avg TTFT',avgTTFT+' ms','p50');

el('requestRows').innerHTML=lastRows.map((r,i)=>'<tr data-idx="'+i+'"><td>'+dt(r.created_at)+'</td><td class="cell-mono" title="'+esc(r.trace_id)+'">'+esc(shortTrace(r.trace_id))+'</td><td>'+esc(r.user)+'</td><td>'+statusBadge(r.status_code)+'</td><td>'+esc(r.model)+'</td><td>'+compact(r.prompt_tokens)+'</td><td>'+compact(r.output_tokens)+'</td><td>'+pct(r.cache_hit_rate)+'</td><td>'+v(r.latency_ms)+' ms</td><td>'+esc(r.error_code)+'</td></tr>').join('')||'<tr><td colspan="10" class="empty-state">No requests</td></tr>';
}

// ─── Usage ───
function renderUsage(usage){const mix=usage.token_mix||{};
const today=usage.daily?.[usage.daily.length-1]||{};
el('usageMetrics').innerHTML=
  metricCard('Prompt Tokens',compact(today.prompt_tokens||0),'today')+
  metricCard('Completion',compact(today.output_tokens||0),'today')+
  metricCard('Cache Read',compact(today.cache_read_tokens||0),'today')+
  metricCard('Cost',money(today.cost_cents),'today');

renderBarChart('tokenChart',[
  {label:'Prompt',value:mix.prompt_tokens||0},
  {label:'Completion',value:mix.output_tokens||0},
  {label:'Cache R',value:mix.cache_read_tokens||0},
  {label:'Cache W',value:mix.cache_write_tokens||0}
]);

el('usageRows').innerHTML=(usage.daily||[]).map(d=>'<tr><td>'+esc(d.date)+'</td><td>'+compact(d.requests)+'</td><td>'+compact(d.errors)+'</td><td>'+compact(d.prompt_tokens)+'</td><td>'+compact(d.output_tokens)+'</td><td>'+compact(num(d.prompt_tokens)+num(d.output_tokens))+'</td><td>'+money(d.cost_cents)+'</td></tr>').join('')||'<tr><td colspan="7" class="empty-state">No data</td></tr>';
}

// ─── Providers ───
function renderProviders(keydata){const stats={};(keydata.stats||[]).forEach(x=>stats[x.provider+'/'+x.provider_key_id]=x);
const providers=[...new Set((keydata.keys||[]).map(k=>k.provider))];
el('providerDetailCards').innerHTML=providers.map(p=>{
  const pkeys=(keydata.keys||[]).filter(k=>k.provider===p);
  const healthy=pkeys.filter(k=>k.state==='healthy').length;
  const total=pkeys.length;
  const pstats=pkeys.map(k=>stats[k.provider+'/'+k.id]||{});
  const totalReq=pstats.reduce((a,s)=>a+num(s.requests),0);
  const totalTokens=pstats.reduce((a,s)=>a+num(s.total_tokens),0);
  const avgRate=Math.round(pstats.reduce((a,s)=>a+num(s.success_rate),0)/(pstats.length||1)*1000)/10;
  return'<div class="card card-pad">'+
    '<div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">'+
      '<span class="status-dot status-'+(healthy===total?'ok':healthy>0?'warn':'bad')+'"></span>'+
      '<div style="font-size:16px;font-weight:600">'+esc(p)+'</div>'+
      '<div style="margin-left:auto">'+stateBadge(healthy===total?'healthy':healthy>0?'cooldown':'disabled')+'</div>'+
    '</div>'+
    '<div class="metric-grid" style="grid-template-columns:repeat(4,1fr);margin-bottom:0">'+
      metricCard('Keys',healthy+'/'+total,'healthy')+
      metricCard('Requests',compact(totalReq),'24h')+
      metricCard('Tokens',compact(totalTokens),'24h')+
      metricCard('Success',avgRate+'%','rate')+
    '</div>'+
    '<div class="divider"></div>'+
    '<div class="table-wrap">'+
      '<table><thead><tr><th>Key</th><th>State</th><th>Requests</th><th>Success</th><th>Errors</th><th>Tokens</th><th>Actions</th></tr></thead>'+
      '<tbody>'+pkeys.map(k=>{
        const st=stats[k.provider+'/'+k.id]||{};const rate=Math.round(num(st.success_rate)*1000)/10;
        return'<tr><td class="cell-mono">'+esc(k.id)+'</td><td>'+stateBadge(k.state)+'</td><td>'+v(st.requests)+'</td><td>'+rate+'%</td><td>'+v(st.errors)+'</td><td>'+compact(st.total_tokens||0)+'</td><td><button class="btn" data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="enable">Enable</button> <button class="btn" data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="cooldown">Cooldown</button> <button class="btn" data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="disable">Disable</button></td></tr>';
      }).join('')+'</tbody></table>'+
    '</div>'+
  '</div>';
}).join('')||'<div class="empty-state"><div class="empty-state-title">No providers</div></div>';
}

// ─── Virtual Keys ───
function renderVirtualKeys(vkeys){const active=(vkeys||[]).filter(k=>k.enabled&&!k.revoked_at).length;const disabled=(vkeys||[]).filter(k=>!k.enabled&&!k.revoked_at).length;const revoked=(vkeys||[]).filter(k=>k.revoked_at).length;
el('keyStatusGrid').innerHTML=
  '<div class="key-status-card"><div class="key-status-count ok">'+active+'</div><div class="key-status-label">Active</div></div>'+
  '<div class="key-status-card"><div class="key-status-count warn">'+disabled+'</div><div class="key-status-label">Disabled</div></div>'+
  '<div class="key-status-card"><div class="key-status-count bad">'+revoked+'</div><div class="key-status-label">Revoked</div></div>'+
  '<div class="key-status-card"><div class="key-status-count">'+(vkeys||[]).length+'</div><div class="key-status-label">Total</div></div>';

el('vkRows').innerHTML=(vkeys||[]).map(k=>{
  const state=k.state||((k.revoked_at)?'revoked':(k.enabled?'active':'disabled'));
  const actions=state==='revoked'?'<span class="cell-badge badge-red">Revoked</span>':'<button class="btn" data-id="'+k.id+'" data-a="'+(state==='active'?'disable':'enable')+'">'+(state==='active'?'Disable':'Enable')+'</button> <button class="btn" data-id="'+k.id+'" data-a="revoke">Revoke</button>';
  return'<tr><td>'+k.id+'</td><td class="cell-mono" title="'+esc(k.api_key)+'">'+esc(k.api_key)+'</td><td>'+esc(k.user)+'</td><td>'+esc((k.allowed_models||[]).join(', '))+'</td><td>'+v(k.rpm_limit)+'</td><td>'+v(k.tpm_limit)+'</td><td>'+v(k.concurrency_limit)+'</td><td>'+stateBadge(state)+'</td><td>'+dt(k.last_used_at)+'</td><td>'+actions+'</td></tr>';
}).join('')||'<tr><td colspan="10" class="empty-state">No virtual keys</td></tr>';
}

// ─── Errors ───
function renderErrors(errors){const breakdown=errors.breakdown||[];const total=breakdown.reduce((a,b)=>a+num(b.count),0);
const byCode={};breakdown.forEach(b=>{byCode[b.status_code]=num(byCode[b.status_code])+num(b.count)});
el('errorMetrics').innerHTML=
  metricCard('Total Errors',compact(total),'24h')+
  metricCard('401 Unauthorized',compact(byCode['401']||0),'auth')+
  metricCard('429 Rate Limit',compact(byCode['429']||0),'rate')+
  metricCard('500+ Server',compact(byCode['500']||0),'server');

renderBarChart('errorChart',breakdown.slice(0,10).map(b=>({label:String(b.status_code),value:b.count})));

const errorRows=[].concat(errors.error_requests||[],errors.slow_requests||[]).filter((r,i,a)=>a.findIndex(x=>x.trace_id===r.trace_id)===i).slice(0,50);
el('errorRows').innerHTML=errorRows.map(r=>'<tr><td>'+dt(r.created_at)+'</td><td class="cell-mono" title="'+esc(r.trace_id)+'">'+esc(shortTrace(r.trace_id))+'</td><td>'+statusBadge(r.status_code)+'</td><td>'+esc(r.error_code)+'</td><td title="'+esc(r.error_message)+'">'+esc((r.error_message||'').slice(0,80))+'</td></tr>').join('')||'<tr><td colspan="5" class="empty-state">No errors</td></tr>';
}

// ─── Request Detail Drawer ───
function showRequestDetail(r){
  const sections=[
    {title:'Basic Info',items:[['Trace ID',esc(r.trace_id)],['Time',dt(r.created_at)],['Model',esc(r.model)],['Protocol',esc(r.protocol)],['Status',r.status_code],['Stream',r.stream?'Yes':'No'],['Client IP',esc(r.client_ip)]]},
    {title:'Usage',items:[['Prompt',compact(r.prompt_tokens)],['Completion',compact(r.output_tokens)],['Total',compact(r.total_tokens)],['Cache Read',compact(r.cache_read_tokens)],['Cache Write',compact(r.cache_write_tokens)],['Hit Rate',pct(r.cache_hit_rate)],['Finish Reason',esc(r.finish_reason)]]},
    {title:'Performance',items:[['Latency',v(r.latency_ms)+' ms'],['TTFT',v(r.ttft_ms)+' ms'],['Generation',v(r.generation_ms)+' ms'],['Token Rate',v(r.output_tokens_per_second)]]},
    {title:'Routing',items:[['User',esc(r.user)],['Provider',esc(r.provider)],['Key',esc(r.provider_key_id)],['Upstream',esc(r.upstream_model)],['Decision',esc(r.router_decision)],['Fallbacks',v(r.fallback_count)]]},
    {title:'Cost',items:[['Cost',money(r.cost_cents)]]}
  ];
  if(r.error_code)sections.push({title:'Error',items:[['Code',esc(r.error_code)],['Message',esc(r.error_message)]]});

  const html=sections.map(s=>'<div style="margin-bottom:20px"><div class="section-title">'+s.title+'</div><div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px">'+s.items.map(i=>'<div style="background:var(--surface-2);padding:10px 12px;border-radius:var(--radius-sm)"><div style="font-size:11px;color:var(--text-3);margin-bottom:2px">'+i[0]+'</div><div style="font-size:13px;font-weight:500">'+i[1]+'</div></div>').join('')+'</div></div>').join('')+
    (r.request_body?'<div style="margin-bottom:20px"><div class="section-title">Request Body</div><div class="pre-wrap">'+esc(r.request_body)+'</div></div>':'')+
    (r.response_body?'<div style="margin-bottom:20px"><div class="section-title">Response Body</div><div class="pre-wrap">'+esc(r.response_body)+'</div></div>':'');

  openDrawer(r.model+' · '+shortTrace(r.trace_id),html);
}

// ─── Actions ───
async function keyAction(p,k,a){await fetch(api('/dashboard/api/keys/'+encodeURIComponent(p)+'/'+encodeURIComponent(k)+'/'+a),{method:'POST',headers:h});await load()}
async function vkeyAction(id,a){await fetch(api('/dashboard/api/virtual-keys/'+id+'/'+a),{method:'POST',headers:h});await load()}
async function createVKey(){const btn=el('vkGenerateBtn');btn.disabled=true;btn.textContent='...';try{const body={user:el('vkUser').value||'default',allowed_models:el('vkModels').value.split(',').map(x=>x.trim()).filter(Boolean),rpm_limit:parseInt(el('vkRpm').value||'100'),tpm_limit:parseInt(el('vkTpm').value||'200000'),concurrency_limit:parseInt(el('vkConc').value||'10')};const data=await j(api('/dashboard/api/virtual-keys'),{method:'POST',headers:jsonHeaders(),body:JSON.stringify(body)});if(data&&data.api_key){el('vkGenerated').innerHTML='<div style="background:var(--green-bg);border:1px solid rgba(23,201,100,.2);border-radius:var(--radius-sm);padding:12px 16px;display:flex;align-items:center;justify-content:space-between;gap:12px"><div><div style="font-size:12px;font-weight:500;color:var(--green);margin-bottom:4px">New key generated</div><div class="cell-mono" style="font-size:13px">'+esc(data.api_key)+'</div></div><button class="btn btn-primary" id="vkCopyBtn">Copy</button></div>';el('vkGenerated').style.display='block';el('vkCopyBtn').addEventListener('click',()=>navigator.clipboard.writeText(data.api_key));load();return}throw new Error('No key')}catch(e){el('vkGenerated').innerHTML='<div style="background:var(--red-bg);border:1px solid rgba(239,68,68,.2);border-radius:var(--radius-sm);padding:12px 16px;color:var(--red);font-size:13px">'+esc(e.message)+'</div>';el('vkGenerated').style.display='block'}finally{btn.disabled=false;btn.textContent='Generate Key'}}

// ─── Load ───
async function load(){const s=await j(api('/dashboard/api/summary?hours=24'));const usage=await j(api('/dashboard/api/usage?days=14&limit=20'));const keydata=await j(api('/dashboard/api/providers?hours=24'));const keyOverview=await j(api('/dashboard/api/keys-overview'));const errors=await j(api('/dashboard/api/errors?hours=24&limit=100'));const rows=await j(api('/dashboard/api/requests?'+qs().toString()));
renderOverview(s,rows||[],keydata,errors,usage);renderRequests(rows||[]);renderUsage(usage);renderProviders(keydata);renderVirtualKeys(keyOverview.virtual_keys||[]);renderErrors(errors);}

// ─── Events ───
document.addEventListener('click',e=>{
  const n=e.target.closest('[data-view]');if(n)setView(n.dataset.view);
  const p=e.target.closest('[data-p][data-a]');if(p)keyAction(p.dataset.p,p.dataset.k,p.dataset.a).catch(()=>{});
  const k=e.target.closest('[data-id][data-a]');if(k)vkeyAction(k.dataset.id,k.dataset.a).catch(()=>{});
  const tr=e.target.closest('tr[data-idx]');if(tr){const r=lastRows[Number(tr.dataset.idx)];if(r)showRequestDetail(r)}
  const pc=e.target.closest('.provider-card');if(pc){/* TODO: provider detail */}
});

el('refreshBtn').addEventListener('click',()=>load().catch(()=>{}));
el('rFilterBtn').addEventListener('click',()=>load().catch(()=>{}));
el('vkGenerateBtn').addEventListener('click',createVKey);

// Tab switching
document.querySelectorAll('.tab-btn').forEach(t=>t.addEventListener('click',()=>{document.querySelectorAll('.tab-btn').forEach(x=>x.classList.toggle('active',x===t));currentPeriod=t.dataset.period;load().catch(()=>{})}));

// Init
load().catch(e=>{el('statusBanner').innerHTML='<div style="background:var(--red-bg);border:1px solid rgba(239,68,68,.2);border-radius:var(--radius-sm);padding:12px 16px;color:var(--red);font-size:13px">'+esc(e.message)+'</div>';el('statusBanner').style.display='block'});
setInterval(()=>load().catch(()=>{}),15000);
</script>
</body></html>`
}

func requestDetailHTML(traceID string) string {
	return `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Request Detail</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--line:#dce2ea;--line2:#edf1f5;--text:#1f2933;--muted:#667085;--blue:#2563eb;--green:#14804a;--red:#b42318;--amber:#a15c00;--nav:#111827;--nav2:#1f2937;--r:8px}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}.wrap{min-height:100vh;padding:20px 22px 28px}.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;color:var(--muted);margin-top:4px}.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.notice{background:#fff;border:1px solid var(--line);border-radius:7px;padding:9px 11px;white-space:normal;word-break:break-word;display:none}.notice.bad{border-color:#efb2aa;background:#fff7f6}.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:14px}.detail-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}.detail-title{font-size:16px;font-weight:700}.detail-sub{font-size:12px;color:var(--muted)}.detail-group{margin-top:12px}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.kv{border:1px solid var(--line2);border-radius:6px;padding:8px 10px;background:#fafbfc}.k{font-size:11px;color:var(--muted);margin-bottom:4px}.v{font-size:13px;word-break:break-word;white-space:normal}.bodybox{margin:0;padding:10px 12px;border:1px solid var(--line2);border-radius:6px;background:#fafbfc;font-size:12px;line-height:1.45;white-space:pre-wrap;word-break:break-word;max-height:280px;overflow:auto}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.mono{font-family:Consolas,monospace}.btn{height:34px;border:1px solid #aeb8c5;border-radius:6px;background:#fff;padding:0 10px;display:inline-flex;align-items:center;text-decoration:none;color:var(--text);cursor:pointer;font-size:12px}.hint{font-size:12px;color:var(--muted)}@media(max-width:640px){.detail-grid{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><div class="wrap"><div class="top"><div><div class="title">Request Detail</div><div class="subtitle mono" id="traceLabel">` + traceID + `</div></div><div class="actions"><a class="btn" id="backBtn" href="#">Back</a></div></div><div id="statusBanner" class="notice" style="display:block;margin-bottom:12px"></div><div id="detail" class="panel"><div class="hint">Loading...</div></div></div><script>
const traceID=` + strconv.Quote(traceID) + `;const token=new URLSearchParams(location.search).get('token')||localStorage.getItem('llm_gateway_dashboard_token')||'';if(token)localStorage.setItem('llm_gateway_dashboard_token',token);const h=token?{'X-Dashboard-Token':token}:{};const el=id=>document.getElementById(id);function api(path){const u=new URL(path,location.origin);if(token)u.searchParams.set('token',token);return u.pathname+u.search}function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]})}function v(x,d){return x==null?(d==null?0:d):x}function num(x){return Number(x)||0}function dt(x){return x?new Date(x).toLocaleString():''}function pct(x){return x==null?'0%':(Math.round(num(x)*1000)/10)+'%'}function money(c){return '$'+((c||0)/100).toFixed(4)}function field(label,value){return '<div class="kv"><div class="k">'+label+'</div><div class="v">'+value+'</div></div>'}function group(title,items){return '<div class="detail-group"><div class="detail-title">'+title+'</div><div class="detail-grid">'+items.join('')+'</div></div>'}function body(title,value){if(!value)return '<div class="detail-group"><div class="detail-title">'+title+'</div><div class="hint">No body captured.</div></div>';return '<div class="detail-group"><div class="detail-title">'+title+'</div><div class="bodybox mono">'+esc(value)+'</div></div>'}function showBanner(text,isError){const box=el('statusBanner');box.style.display='block';box.className='notice'+(isError?' bad':'');box.textContent=text}function render(r){const box=el('detail');if(!box)return;if(!r){box.innerHTML='<div class="hint">Request not found.</div>';return}box.innerHTML='<div class="detail-head"><div><div class="detail-title">'+esc(r.model||'Request')+'</div><div class="detail-sub mono">'+esc(r.trace_id)+'</div></div><div class="detail-sub">'+dt(r.created_at)+'</div></div>'+group('Basic Info',[field('ID',esc(r.trace_id)),field('Time',dt(r.created_at)),field('Model',esc(r.model)),field('Protocol',esc(r.protocol)),field('Status Code',v(r.status_code)),field('Stream',r.stream?'yes':'no'),field('Has Image',r.has_image?'yes':'no'),field('Total Time',v(r.latency_ms)+' ms'),field('First Token',v(r.ttft_ms)+' ms'),field('Request Cost',money(v(r.cost_cents))),field('Token Rate',v(r.output_tokens_per_second)),field('Client IP',esc(r.client_ip)),field('User API Key',esc(r.api_key))])+group('Usage',[field('Prompt',v(r.prompt_tokens)),field('Completion',v(r.output_tokens)),field('Total',v(r.total_tokens)),field('Cache Read',v(r.cache_read_tokens)),field('Cache Write',v(r.cache_write_tokens)),field('Cache Hit',pct(r.cache_hit_rate)),field('Finish Reason',esc(r.finish_reason)),field('Prompt Bucket',esc(r.prompt_bucket))])+group('Routing',[field('User',esc(r.user)),field('Provider',esc(r.provider)),field('Provider Key',esc(r.provider_key_id)),field('Upstream Model',esc(r.upstream_model)),field('Router Decision',esc(r.router_decision)),field('Fallback Count',v(r.fallback_count)),field('Slow Reason',esc(r.slow_reason)),field('Generation',v(r.generation_ms)+' ms')])+body('Request Body',r.request_body)+body('Response Body',r.response_body)+group('Error',[field('Error Code',esc(r.error_code)),field('Error Message',esc(r.error_message))])}async function load(){showBanner(token?'Dashboard token loaded.':'Dashboard token missing. Open with ?token=...',false);el('backBtn').href=api('/dashboard?token='+encodeURIComponent(token));const r=await fetch(api('/dashboard/api/requests/'+encodeURIComponent(traceID)),{headers:h});const text=await r.text();if(!r.ok){showBanner(text||r.statusText,true);render(null);return}render(text?JSON.parse(text):null)}load().catch(function(e){showBanner(e.message,true);render(null)});
</script></body></html>`
}

// hashMessages 对 OpenAI messages 计算 SHA256。
func hashMessages(messages []models.OpenAIMessage) string {
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(m.Role)
		sb.WriteString(":")
		sb.WriteString(m.Content)
		sb.WriteString("\n")
	}
	h := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(h[:])
}

// hashAnthropicMessages 对 Anthropic messages + system 计算 SHA256。
func hashAnthropicMessages(messages []models.AnthropicMessage, system string) string {
	var sb strings.Builder
	if system != "" {
		sb.WriteString("system:")
		sb.WriteString(system)
		sb.WriteString("\n")
	}
	for _, m := range messages {
		sb.WriteString(m.Role)
		sb.WriteString(":")
		content := ""
		if s, ok := m.Content.(string); ok {
			content = s
		}
		sb.WriteString(content)
		sb.WriteString("\n")
	}
	h := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(h[:])
}

// hashOpenAIResponseContent 对 OpenAI 响应内容计算 SHA256。
func hashOpenAIResponseContent(resp *models.OpenAIChatCompletionResponse) string {
	var sb strings.Builder
	for _, c := range resp.Choices {
		sb.WriteString(c.Message.Content)
	}
	h := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(h[:])
}

func randomID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano())
}

// hashAnthropicResponseContent 对 Anthropic 响应内容计算 SHA256。
func hashAnthropicResponseContent(resp *models.AnthropicMessageResponse) string {
	var sb strings.Builder
	for _, c := range resp.Content {
		switch c.Type {
		case "text":
			sb.WriteString(c.Text)
		case "thinking":
			sb.WriteString(c.Thinking)
		case "tool_use":
			sb.WriteString(c.ID)
			sb.WriteString(c.Name)
			if c.Input != nil {
				if b, err := json.Marshal(c.Input); err == nil {
					sb.WriteString(string(b))
				}
			}
		default:
			sb.WriteString(c.Text)
			sb.WriteString(c.Thinking)
		}
	}
	h := sha256.Sum256([]byte(sb.String()))
	return hex.EncodeToString(h[:])
}

// hashKey 对 Key 进行脱敏，返回前 8 位 + "***" 的哈希标识。
func hashKey(key string) string {
	if len(key) <= 8 {
		return "***"
	}
	return key[:8] + "***"
}

// fiberResponseWriter wraps fiber.Ctx to implement http.ResponseWriter for Prometheus.
type fiberResponseWriter struct {
	ctx *fiber.Ctx
}

func (w *fiberResponseWriter) Header() http.Header {
	return http.Header{}
}

func (w *fiberResponseWriter) Write(p []byte) (int, error) {
	return w.ctx.Write(p)
}

func (w *fiberResponseWriter) WriteHeader(statusCode int) {
	w.ctx.Status(statusCode)
}
