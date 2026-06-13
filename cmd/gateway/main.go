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
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LLM Gateway Dashboard</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--panel2:#f8fafc;--line:#dce2ea;--line2:#edf1f5;--text:#1f2933;--muted:#667085;--blue:#2563eb;--green:#14804a;--red:#b42318;--amber:#a15c00;--nav:#111827;--nav2:#1f2937;--r:8px}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}.app{min-height:100vh;display:grid;grid-template-columns:232px 1fr}.side{background:var(--nav);color:#d1d5db;padding:18px 14px;position:sticky;top:0;height:100vh}.brand{color:#fff;font-weight:700;font-size:16px;margin:4px 8px 18px}.navbtn{width:100%;height:38px;border:0;border-radius:7px;background:transparent;color:#d1d5db;text-align:left;padding:0 11px;margin:3px 0;cursor:pointer;font-size:13px}.navbtn:hover{background:var(--nav2);color:#fff}.navbtn.active{background:#fff;color:#111827}.sidefoot{position:absolute;left:14px;right:14px;bottom:16px;font-size:11px;color:#9ca3af;line-height:1.5}.main{min-width:0;padding:20px 22px 28px}.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px}.title{font-size:22px;font-weight:700}.subtitle{font-size:12px;color:var(--muted);margin-top:4px}.actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.notice{background:#fff;border:1px solid var(--line);border-radius:7px;padding:9px 11px;white-space:normal;word-break:break-word;display:none}.notice.bad{border-color:#efb2aa;background:#fff7f6}.cards{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin-bottom:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);padding:12px;min-height:76px}.label{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.value{font-size:21px;font-weight:700;margin-top:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.hint{font-size:11px;color:var(--muted);margin-top:4px}.grid2{display:grid;grid-template-columns:1.15fr .85fr;gap:14px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:var(--r);overflow:hidden}.panel.pad{padding:14px}.panelhd{height:42px;padding:0 13px;border-bottom:1px solid var(--line2);display:flex;align-items:center;justify-content:space-between;gap:10px}.paneltitle{font-size:14px;font-weight:700}.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:0 0 12px}.bar input,.bar select{height:34px;border:1px solid #cfd6df;border-radius:6px;padding:0 8px;background:#fff;color:var(--text)}button,a.btn{height:34px;border:1px solid #aeb8c5;border-radius:6px;background:#fff;padding:0 10px;display:inline-flex;align-items:center;text-decoration:none;color:var(--text);cursor:pointer;font-size:12px}.primary{background:var(--text);color:white;border-color:var(--text)}table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:8px 9px;border-bottom:1px solid var(--line2);text-align:left;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}th{background:#f0f3f7;color:#4b5563;font-weight:700}.view{display:none}.view.active{display:block}.request-detail{margin-top:12px;border:1px solid var(--line);border-radius:var(--r);background:#fff;padding:12px}.detail-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap;margin-bottom:10px}.detail-title{font-size:16px;font-weight:700}.detail-sub{font-size:12px;color:var(--muted)}.detail-group{margin-top:12px}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.kv{border:1px solid var(--line2);border-radius:6px;padding:8px 10px;background:#fafbfc}.k{font-size:11px;color:var(--muted);margin-bottom:4px}.v{font-size:13px;word-break:break-word;white-space:normal}.request-row{cursor:pointer}.request-row:hover td{background:#f7faff}.request-row.active td{background:#eef4ff}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.mono{font-family:Consolas,monospace}.metriclist{display:grid;gap:9px}.metricrow{display:grid;grid-template-columns:120px 1fr auto;gap:10px;align-items:center;font-size:12px}.track{height:8px;background:#eef2f7;border-radius:999px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:999px}.breakrow{display:grid;grid-template-columns:54px 1fr auto;gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid var(--line2);font-size:12px}.code{font-family:Consolas,monospace;color:#4b5563}.count{font-weight:700}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}.generated{margin-top:8px}.wide{grid-column:1/-1}@media(max-width:1100px){.cards{grid-template-columns:repeat(4,1fr)}.grid2,.split{grid-template-columns:1fr}.app{grid-template-columns:1fr}.side{position:relative;height:auto}.sidefoot{position:static;margin:16px 8px 0}.nav{display:flex;gap:6px;overflow:auto}.navbtn{width:auto;white-space:nowrap}.main{padding:14px}}@media(max-width:640px){.cards{grid-template-columns:repeat(2,1fr)}.detail-grid{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><div class="app"><aside class="side"><div class="brand">LLM Gateway</div><nav class="nav"><button class="navbtn active" data-view="overview">Overview</button><button class="navbtn" data-view="requests">Requests</button><button class="navbtn" data-view="usage">Usage</button><button class="navbtn" data-view="providers">Providers</button><button class="navbtn" data-view="keys">Virtual Keys</button><button class="navbtn" data-view="errors">Errors</button></nav><div class="sidefoot"><div>api.jsnanshuo.com</div><div id="sideStatus">Waiting for data</div></div></aside><main class="main"><div class="top"><div><div class="title">Gateway Dashboard</div><div class="subtitle">Live request, cache, latency and provider health.</div></div><div class="actions"><button id="reloadBtn">Reload</button><a class="btn" id="export" href="#">Export CSV</a></div></div><div id="statusBanner" class="notice" style="display:block;margin-bottom:12px"></div><section class="cards" id="summaryCards"></section><section id="overview" class="view active"><div class="grid2"><div class="panel"><div class="panelhd"><div class="paneltitle">Recent Requests</div><button data-view-go="requests">Open</button></div><table><thead><tr><th>Time</th><th>Status</th><th>Model</th><th>Prompt</th><th>TTFT</th><th>Latency</th></tr></thead><tbody id="overviewRows"></tbody></table></div><div class="panel pad"><div class="paneltitle">Provider Health</div><div id="providerSummary" class="metriclist" style="margin-top:12px"></div></div><div class="panel pad"><div class="paneltitle">Daily Usage</div><div id="dailyBars" class="metriclist" style="margin-top:12px"></div></div><div class="panel pad"><div class="paneltitle">Error Breakdown</div><div id="errorBreakdown" style="margin-top:8px"></div></div></div></section><section id="requests" class="view"><div class="panel pad"><div class="bar"><input id="provider" placeholder="provider"><input id="provider_key_id" placeholder="provider key"><input id="model" placeholder="model"><input id="user" placeholder="user"><select id="status"><option value="">all</option><option value="success">success</option><option value="error">error</option></select><button id="filterBtn">Filter</button></div><table><thead><tr><th>Time</th><th>Trace</th><th>User</th><th>API Key</th><th>Status</th><th>Model</th><th>Prompt</th><th>Completion</th><th>Cache Read</th><th>Cache Write</th><th>Hit Rate</th><th>Latency</th><th>Error</th></tr></thead><tbody id="rows"></tbody></table><div id="requestDetail" class="request-detail"></div></div></section><section id="usage" class="view"><div class="split"><div class="panel pad"><div class="paneltitle">Daily Usage</div><div id="usageDailyBars" class="metriclist" style="margin-top:12px"></div></div><div class="panel pad"><div class="paneltitle">Token Mix</div><div id="usageTokenMix" class="metriclist" style="margin-top:12px"></div></div></div><div class="panel" style="margin-top:14px"><div class="panelhd"><div class="paneltitle">Daily Usage Table</div></div><table><thead><tr><th>Date</th><th>Requests</th><th>Errors</th><th>Prompt</th><th>Completion</th><th>Total</th><th>Cost</th></tr></thead><tbody id="usageDailyRows"></tbody></table></div><div class="panel" style="margin-top:14px"><div class="panelhd"><div class="paneltitle">Recent Usage</div><button data-view-go="requests">Open Requests</button></div><table><thead><tr><th>Time</th><th>Model</th><th>Prompt</th><th>Completion</th><th>Cache Read</th><th>Hit Rate</th><th>TTFT</th><th>Latency</th></tr></thead><tbody id="usageRecentRows"></tbody></table></div></section><section id="providers" class="view"><div class="panel"><div class="panelhd"><div class="paneltitle">Provider Keys</div></div><table><thead><tr><th>Provider</th><th>Key</th><th>State</th><th>Requests</th><th>Success Rate</th><th>Errors</th><th>Tokens</th><th>Actions</th></tr></thead><tbody id="keyrows"></tbody></table></div></section><section id="keys" class="view"><div class="panel pad"><div class="paneltitle">Virtual API Keys</div><div class="bar" style="margin-top:12px"><input id="vuser" placeholder="user" value="default"><input id="vmodels" placeholder="models, comma separated" value="kimi-for-coding,gpt-5.5"><input id="vrpm" placeholder="rpm" value="100"><input id="vtpm" placeholder="tpm" value="200000"><input id="vconc" placeholder="concurrency" value="10"><button id="generateKeyBtn" class="primary">Generate</button></div><pre id="generatedKey" class="notice mono generated"></pre><table><thead><tr><th>ID</th><th>API Key</th><th>User</th><th>Models</th><th>RPM</th><th>TPM</th><th>Concurrency</th><th>State</th><th>Last Used</th><th>Actions</th></tr></thead><tbody id="vkeyrows"></tbody></table></div></section><section id="errors" class="view"><div class="split"><div class="panel pad"><div class="paneltitle">Error Breakdown</div><div id="errorBreakdownFull" style="margin-top:8px"></div></div><div class="panel"><div class="panelhd"><div class="paneltitle">Slow / Error Requests</div></div><table><thead><tr><th>Time</th><th>Trace</th><th>Status</th><th>Slow</th><th>Latency</th><th>Error Code</th><th>Error Message</th></tr></thead><tbody id="errorRows"></tbody></table></div></div></section></main></div><script>
const token=new URLSearchParams(location.search).get('token')||localStorage.getItem('llm_gateway_dashboard_token')||'';if(token)localStorage.setItem('llm_gateway_dashboard_token',token);const h=token?{'X-Dashboard-Token':token}:{};const el=id=>document.getElementById(id);let lastRows=[];function api(path){const u=new URL(path,location.origin);if(token)u.searchParams.set('token',token);return u.pathname+u.search}function qs(){const u=new URLSearchParams();['provider','provider_key_id','model','user','status'].forEach(id=>{const n=el(id);const value=n?n.value:'';if(value)u.set(id,value)});u.set('limit','100');if(token)u.set('token',token);return u}function money(c){return '$'+((c||0)/100).toFixed(4)}function esc(v){return String(v==null?'':v).replace(/[&<>"']/g,function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]})}function v(x,d){return x==null?(d==null?0:d):x}function num(x){return Number(x)||0}function dt(x){return x?new Date(x).toLocaleString():''}function shortTrace(x){return x?String(x).slice(0,8):''}function pct(x){return x==null?'0%':(Math.round(num(x)*1000)/10)+'%'}function compact(n){n=num(n);if(n>=1000000)return (n/1000000).toFixed(1)+'M';if(n>=1000)return (n/1000).toFixed(1)+'K';return String(Math.round(n))}function jsonHeaders(){return Object.assign({'Content-Type':'application/json'},h)}function showGenerated(text,isError,copyValue){const box=el('generatedKey');box.style.display='block';box.className='notice mono generated'+(isError?' bad':'');if(copyValue){box.innerHTML='<div>'+esc(text)+'</div><button data-copy-key="'+esc(copyValue)+'" class="copy-key">Copy</button>'}else{box.textContent=text}}function showBanner(text,isError){const box=el('statusBanner');box.style.display='block';box.className='notice'+(isError?' bad':'');box.textContent=text}async function j(url,opt){const r=await fetch(url,opt||{headers:h});const text=await r.text();if(!r.ok)throw new Error(text||r.statusText);return text?JSON.parse(text):{}}function setView(name){document.querySelectorAll('.view').forEach(function(x){x.classList.toggle('active',x.id===name)});document.querySelectorAll('.navbtn').forEach(function(x){x.classList.toggle('active',x.dataset.view===name)})}function card(label,value,hint){return '<div class="card"><div class="label">'+label+'</div><div class="value">'+value+'</div><div class="hint">'+hint+'</div></div>'}function renderSummary(s,rows){const avgCache=rows.length?rows.reduce(function(a,r){return a+num(r.cache_hit_rate)},0)/rows.length:0;const slow=rows.filter(function(r){return r.slow_reason||num(r.status_code)>=400}).length;el('summaryCards').innerHTML=[card('Requests',compact(s.requests),'last 24h'),card('Errors',compact(s.errors),'last 24h'),card('Prompt Tokens',compact(s.prompt_tokens),'input'),card('Completion',compact(s.output_tokens),'output'),card('Avg Latency',compact(s.avg_latency_ms)+' ms','24h'),card('Avg TTFT',compact(s.avg_ttft_ms)+' ms','24h'),card('Cache Hit',pct(avgCache),'recent 100'),card('Slow/Error',compact(slow),'recent 100')].join('');el('sideStatus').textContent=compact(s.requests)+' requests / '+compact(s.errors)+' errors'}function renderErrors(eb){const html=(eb||[]).map(function(x){return '<div class="breakrow"><span class="code">'+esc(x.status_code)+'</span><span>'+esc(x.label)+'</span><span class="count">'+v(x.count)+'</span></div>'}).join('')||'<div class="hint">No errors in this window.</div>';el('errorBreakdown').innerHTML=html;el('errorBreakdownFull').innerHTML=html}function renderErrorRows(rows){const html=(rows||[]).map(function(r){return '<tr><td>'+dt(r.created_at)+'</td><td class="mono" title="'+esc(r.trace_id)+'">'+esc(shortTrace(r.trace_id))+'</td><td>'+v(r.status_code)+'</td><td>'+esc(r.slow_reason)+'</td><td>'+v(r.latency_ms)+' ms</td><td>'+esc(r.error_code)+'</td><td title="'+esc(r.error_message)+'">'+esc(r.error_message)+'</td></tr>'}).join('')||'<tr><td colspan="7" class="hint">No error requests in this window.</td></tr>';el('errorRows').innerHTML=html}function renderDaily(daily){const max=Math.max.apply(null,(daily||[]).map(function(d){return num(d.requests)}).concat([1]));const bars=(daily||[]).slice(0,10).map(function(d){const w=Math.max(4,Math.round(num(d.requests)/max*100));return '<div class="metricrow"><span>'+esc(d.date)+'</span><div class="track"><div class="fill" style="width:'+w+'%"></div></div><b>'+compact(d.requests)+'</b></div>'}).join('')||'<div class="hint">No usage records.</div>';el('dailyBars').innerHTML=bars;el('usageDailyBars').innerHTML=bars;el('usageDailyRows').innerHTML=(daily||[]).map(function(d){return '<tr><td>'+esc(d.date)+'</td><td>'+compact(d.requests)+'</td><td>'+compact(d.errors)+'</td><td>'+compact(d.prompt_tokens)+'</td><td>'+compact(d.output_tokens)+'</td><td>'+compact(num(d.prompt_tokens)+num(d.output_tokens))+'</td><td>'+money(d.cost_cents)+'</td></tr>'}).join('')||'<tr><td colspan="7" class="hint">No usage records.</td></tr>'}function renderTokenMix(mix){mix=mix||{};const rows=[['Prompt',mix.prompt_tokens],['Completion',mix.output_tokens],['Cache Read',mix.cache_read_tokens],['Cache Write',mix.cache_write_tokens]];const max=Math.max.apply(null,rows.map(function(x){return num(x[1])}).concat([1]));el('usageTokenMix').innerHTML=rows.map(function(x){const w=Math.max(4,Math.round(num(x[1])/max*100));return '<div class="metricrow"><span>'+esc(x[0])+'</span><div class="track"><div class="fill" style="width:'+w+'%"></div></div><b>'+compact(x[1])+'</b></div>'}).join('')}function renderRecentUsage(rows){el('usageRecentRows').innerHTML=(rows||[]).map(function(r){return '<tr><td>'+dt(r.created_at)+'</td><td>'+esc(r.model)+'</td><td>'+compact(r.prompt_tokens)+'</td><td>'+compact(r.output_tokens)+'</td><td>'+compact(r.cache_read_tokens)+'</td><td>'+pct(r.cache_hit_rate)+'</td><td>'+v(r.ttft_ms)+' ms</td><td>'+v(r.latency_ms)+' ms</td></tr>'}).join('')||'<tr><td colspan="8" class="hint">No recent usage.</td></tr>'}function renderProviderSummary(keydata){const stats={};(keydata.stats||[]).forEach(function(x){stats[x.provider+'/'+x.provider_key_id]=x});el('providerSummary').innerHTML=(keydata.keys||[]).map(function(k){const st=stats[k.provider+'/'+k.id]||{};const rate=Math.round(num(st.success_rate)*1000)/10;return '<div class="metricrow"><span>'+esc(k.provider+'/'+k.id)+'</span><div class="track"><div class="fill" style="width:'+Math.max(2,rate)+'%"></div></div><b class="'+(k.state==='healthy'?'ok':k.state==='cooldown'?'warn':'bad')+'">'+esc(k.state)+' '+rate+'%</b></div>'}).join('')||'<div class="hint">No provider keys configured.</div>'}function renderProviderRows(keydata){const stats={};(keydata.stats||[]).forEach(function(x){stats[x.provider+'/'+x.provider_key_id]=x});el('keyrows').innerHTML=(keydata.keys||[]).map(function(k){const st=stats[k.provider+'/'+k.id]||{};const rate=Math.round(num(st.success_rate)*1000)/10;return '<tr><td>'+esc(k.provider)+'</td><td>'+esc(k.id)+'</td><td class="'+(k.state==='healthy'?'ok':k.state==='cooldown'?'warn':'bad')+'">'+esc(k.state)+'</td><td>'+v(st.requests)+'</td><td>'+rate+'%</td><td>'+v(st.errors)+'</td><td>'+v(st.total_tokens)+'</td><td><button data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="enable" class="provider-action">Enable</button> <button data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="cooldown" class="provider-action">Cooldown</button> <button data-p="'+esc(k.provider)+'" data-k="'+esc(k.id)+'" data-a="disable" class="provider-action">Disable</button></td></tr>'}).join('')||'<tr><td colspan="8" class="hint">No provider keys configured.</td></tr>'}function renderVKeys(vkeys){el('vkeyrows').innerHTML=(vkeys||[]).map(function(k){const state=k.state||((k.revoked_at)?'revoked':(k.enabled?'active':'disabled'));const cls=state==='active'?'ok':state==='disabled'?'warn':'bad';const actions=state==='revoked'?'<span class="bad">Revoked</span>':'<button data-id="'+k.id+'" data-a="'+(state==='active'?'disable':'enable')+'" class="vkey-action">'+(state==='active'?'Disable':'Enable')+'</button> <button data-id="'+k.id+'" data-a="revoke" class="vkey-action">Revoke</button>';const keyCell='<span class="mono" title="Keys are masked after creation">'+esc(k.api_key)+'</span>';return '<tr><td>'+k.id+'</td><td>'+keyCell+'</td><td>'+esc(k.user)+'</td><td>'+esc((k.allowed_models||[]).join(','))+'</td><td>'+v(k.rpm_limit)+'</td><td>'+v(k.tpm_limit)+'</td><td>'+v(k.concurrency_limit)+'</td><td class="'+cls+'">'+esc(state)+'</td><td>'+dt(k.last_used_at)+'</td><td>'+actions+'</td></tr>'}).join('')||'<tr><td colspan="10" class="hint">No virtual keys.</td></tr>'}function renderRequestRows(rows){lastRows=rows||[];el('overviewRows').innerHTML=lastRows.slice(0,8).map(function(r){return '<tr><td>'+dt(r.created_at)+'</td><td>'+v(r.status_code)+'</td><td>'+esc(r.model)+'</td><td>'+compact(r.prompt_tokens)+'</td><td>'+v(r.ttft_ms)+' ms</td><td>'+v(r.latency_ms)+' ms</td></tr>'}).join('')||'<tr><td colspan="6" class="hint">No requests.</td></tr>';el('rows').innerHTML=lastRows.map(function(r,i){return '<tr class="request-row'+(i===0?' active':'')+'" data-idx="'+i+'"><td>'+dt(r.created_at)+'</td><td class="mono" title="'+esc(r.trace_id)+'">'+esc(shortTrace(r.trace_id))+'</td><td>'+esc(r.user)+'</td><td class="mono" title="'+esc(r.api_key)+'">'+esc(r.api_key)+'</td><td>'+r.status_code+'</td><td>'+esc(r.model)+'</td><td>'+v(r.prompt_tokens)+'</td><td>'+v(r.output_tokens)+'</td><td>'+v(r.cache_read_tokens)+'</td><td>'+v(r.cache_write_tokens)+'</td><td>'+pct(r.cache_hit_rate)+'</td><td>'+v(r.latency_ms)+' ms</td><td>'+esc(r.error_code)+'</td></tr>'}).join('')||'<tr><td colspan="13" class="hint">No requests.</td></tr>';renderDetail(lastRows[0]||null)}function field(label,value){return '<div class="kv"><div class="k">'+label+'</div><div class="v">'+value+'</div></div>'}function group(title,items){return '<div class="detail-group"><div class="detail-title">'+title+'</div><div class="detail-grid">'+items.join('')+'</div></div>'}function renderDetail(r){const box=el('requestDetail');if(!box)return;if(!r){box.innerHTML='<div class="hint">Select a request to inspect details.</div>';return}box.innerHTML='<div class="detail-head"><div><div class="detail-title">'+esc(r.model||'Request')+'</div><div class="detail-sub mono">'+esc(r.trace_id)+'</div></div><div class="detail-sub">'+dt(r.created_at)+'</div></div>'+group('Basic Info',[field('ID',esc(r.trace_id)),field('Time',dt(r.created_at)),field('Model',esc(r.model)),field('Protocol',esc(r.protocol)),field('Status Code',v(r.status_code)),field('Stream',r.stream?'yes':'no'),field('Has Image',r.has_image?'yes':'no'),field('Total Time',v(r.latency_ms)+' ms'),field('First Token',v(r.ttft_ms)+' ms'),field('Request Cost',money(v(r.cost_cents))),field('Token Rate',v(r.output_tokens_per_second)),field('Client IP',esc(r.client_ip)),field('User API Key',esc(r.api_key))])+group('Usage',[field('Prompt',v(r.prompt_tokens)),field('Completion',v(r.output_tokens)),field('Total',v(r.total_tokens)),field('Cache Read',v(r.cache_read_tokens)),field('Cache Write',v(r.cache_write_tokens)),field('Cache Hit',pct(r.cache_hit_rate)),field('Finish Reason',esc(r.finish_reason)),field('Prompt Bucket',esc(r.prompt_bucket))])+group('Routing',[field('User',esc(r.user)),field('Provider',esc(r.provider)),field('Provider Key',esc(r.provider_key_id)),field('Upstream Model',esc(r.upstream_model)),field('Router Decision',esc(r.router_decision)),field('Fallback Count',v(r.fallback_count)),field('Slow Reason',esc(r.slow_reason)),field('Generation',v(r.generation_ms)+' ms')])+group('Request Body',[field('Body',r.request_body||'')])+group('Response Body',[field('Body',r.response_body||'')])+group('Error',[field('Error Code',esc(r.error_code)),field('Error Message',esc(r.error_message))])}function uniqueRows(rows){const seen={};return (rows||[]).filter(function(r){const k=r.trace_id||JSON.stringify(r);if(seen[k])return false;seen[k]=true;return true})}async function keyAction(p,k,a){await fetch(api('/dashboard/api/keys/'+encodeURIComponent(p)+'/'+encodeURIComponent(k)+'/'+a),{method:'POST',headers:h});await load()}async function vkeyAction(id,a){await fetch(api('/dashboard/api/virtual-keys/'+id+'/'+a),{method:'POST',headers:h});await load()}async function createVKey(){const btn=el('generateKeyBtn');btn.disabled=true;btn.textContent='Generating';try{const body={user:el('vuser').value||'default',allowed_models:el('vmodels').value.split(',').map(function(x){return x.trim()}).filter(Boolean),rpm_limit:parseInt(el('vrpm').value||'100'),tpm_limit:parseInt(el('vtpm').value||'200000'),concurrency_limit:parseInt(el('vconc').value||'10')};const data=await j(api('/dashboard/api/virtual-keys'),{method:'POST',headers:jsonHeaders(),body:JSON.stringify(body)});if(data&&data.api_key){showGenerated('New key, copy once: '+data.api_key,false,data.api_key);load().catch(function(){}) ;return}throw new Error('missing api_key in response')}catch(e){showGenerated('Generate failed: '+e.message,true)}finally{btn.disabled=false;btn.textContent='Generate'}}async function load(){showBanner(token?'Dashboard token loaded. Data refreshes every 15s.':'Dashboard token missing. Open this page with ?token=...',!token);const q=qs();el('export').href='/dashboard/api/requests.csv?'+q.toString();const s=await j(api('/dashboard/api/summary?hours=24'));const usage=await j(api('/dashboard/api/usage?days=14&limit=20'));const keydata=await j(api('/dashboard/api/providers?hours=24'));const keyOverview=await j(api('/dashboard/api/keys-overview'));const errors=await j(api('/dashboard/api/errors?hours=24&limit=100'));const rows=await j(api('/dashboard/api/requests?'+q.toString()));const errorRows=uniqueRows([].concat(errors.error_requests||[],errors.slow_requests||[]));renderRequestRows(rows||[]);renderErrorRows(errorRows);renderSummary(s,rows||[]);renderErrors(errors.breakdown||[]);renderDaily(usage.daily||[]);renderTokenMix(usage.token_mix||{});renderRecentUsage(usage.recent||[]);renderProviderSummary(keydata);renderProviderRows(keydata);renderVKeys(keyOverview.virtual_keys||[])}document.addEventListener('click',function(e){const n=e.target.closest('[data-view]');if(n)setView(n.dataset.view);const g=e.target.closest('[data-view-go]');if(g)setView(g.dataset.viewGo);const p=e.target.closest('.provider-action');if(p)keyAction(p.dataset.p,p.dataset.k,p.dataset.a).catch(function(err){showGenerated('Provider action failed: '+err.message,true)});const k=e.target.closest('.vkey-action');if(k)vkeyAction(k.dataset.id,k.dataset.a).catch(function(err){showGenerated('Key action failed: '+err.message,true)});const tr=e.target.closest('tr[data-idx]');if(tr){const idx=Number(tr.dataset.idx);Array.from(el('rows').querySelectorAll('tr')).forEach(function(x){x.classList.remove('active')});tr.classList.add('active');location.href=api('/dashboard/requests/'+encodeURIComponent(lastRows[idx].trace_id))}});el('generateKeyBtn').addEventListener('click',createVKey);el('filterBtn').addEventListener('click',function(){load().catch(function(e){showGenerated('Load failed: '+e.message,true)})});el('reloadBtn').addEventListener('click',function(){load().catch(function(e){showGenerated('Load failed: '+e.message,true)})});load().catch(function(e){showGenerated('Load failed: '+e.message,true)});setInterval(function(){load().catch(function(){})},15000);
</script><script>
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
