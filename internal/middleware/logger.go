// Package middleware 提供日志、Metrics 和 TraceID 注入功能。
package middleware

import (
	"encoding/json"
	"os"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

// Logger 是结构化 JSON 日志记录器。
type Logger struct {
	level  string
	output *os.File
}

// LogEntry 是单条日志的结构。
type LogEntry struct {
	Timestamp string                 `json:"timestamp"`
	Level     string                 `json:"level"`
	TraceID   string                 `json:"trace_id"`
	Module    string                 `json:"module"`
	Event     string                 `json:"event"`
	LatencyMs int64                  `json:"latency_ms,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// NewLogger 创建 Logger 实例。
func NewLogger(level string) *Logger {
	return &Logger{
		level:  level,
		output: os.Stdout,
	}
}

// Log 输出一条结构化日志。
func (l *Logger) Log(entry LogEntry) {
	if !l.shouldLog(entry.Level) {
		return
	}
	entry.Timestamp = time.Now().UTC().Format(time.RFC3339Nano)
	b, _ := json.Marshal(entry)
	l.output.Write(b)
	l.output.Write([]byte("\n"))
}

// Info 输出 INFO 级别日志。
func (l *Logger) Info(traceID, module, event string, data map[string]interface{}) {
	l.Log(LogEntry{Level: "INFO", TraceID: traceID, Module: module, Event: event, Data: data})
}

// Warn 输出 WARN 级别日志。
func (l *Logger) Warn(traceID, module, event string, data map[string]interface{}) {
	l.Log(LogEntry{Level: "WARN", TraceID: traceID, Module: module, Event: event, Data: data})
}

// Error 输出 ERROR 级别日志。
func (l *Logger) Error(traceID, module, event string, err error, data map[string]interface{}) {
	entry := LogEntry{Level: "ERROR", TraceID: traceID, Module: module, Event: event, Data: data}
	if err != nil {
		entry.Error = err.Error()
	}
	l.Log(entry)
}

// Audit 输出 AUDIT 级别日志（AI 调用专用）。
func (l *Logger) Audit(traceID, event string, data map[string]interface{}) {
	l.Log(LogEntry{Level: "AUDIT", TraceID: traceID, Module: "llm_gateway", Event: event, Data: data})
}

func (l *Logger) shouldLog(level string) bool {
	levels := map[string]int{"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "AUDIT": 4}
	configured := levels[l.level]
	incoming := levels[level]
	return incoming >= configured
}

// TraceIDMiddleware 为每个请求注入 trace_id。
func TraceIDMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		traceID := c.Get("X-Trace-ID")
		if traceID == "" {
			traceID = uuid.New().String()
		}
		c.Locals("trace_id", traceID)
		c.Set("X-Trace-ID", traceID)
		return c.Next()
	}
}

// RequestLogMiddleware 记录请求日志。
func RequestLogMiddleware(logger *Logger) fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		traceID := c.Locals("trace_id").(string)

		path := c.Path()
		method := c.Method()
		if shouldLogRequestStart(method, path) {
			logger.Info(traceID, "http", "request_start", map[string]interface{}{
				"method": method,
				"path":   path,
				"ip":     c.IP(),
			})
		}

		err := c.Next()

		latency := time.Since(start).Milliseconds()
		status := c.Response().StatusCode()
		if shouldLogRequestEnd(method, path, status, latency) {
			logger.Info(traceID, "http", "request_end", map[string]interface{}{
				"method":     method,
				"path":       path,
				"status":     status,
				"latency_ms": latency,
			})
		}

		return err
	}
}

func shouldLogRequestStart(method, path string) bool {
	return false
}

func shouldLogRequestEnd(method, path string, status int, latencyMs int64) bool {
	if isDashboardPath(path) {
		return false
	}
	if status >= 400 {
		return true
	}
	if method == fiber.MethodPost && path == "/v1/messages" {
		return true
	}
	if isLLMPath(path) && latencyMs >= 5000 {
		return true
	}
	return false
}

func isDashboardPath(path string) bool {
	return path == "/" || path == "/dashboard" || strings.HasPrefix(path, "/dashboard/")
}

func isLLMPath(path string) bool {
	return path == "/v1/messages" || path == "/v1/chat/completions" || path == "/v1/responses" || strings.HasPrefix(path, "/v1/messages/")
}

// ExtractTraceID 从 Fiber Context 提取 trace_id。
func ExtractTraceID(c *fiber.Ctx) string {
	if id, ok := c.Locals("trace_id").(string); ok {
		return id
	}
	return ""
}
