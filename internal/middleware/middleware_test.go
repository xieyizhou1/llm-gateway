package middleware

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gofiber/fiber/v2"
)

func TestNewLogger(t *testing.T) {
	l := NewLogger("INFO")
	if l == nil {
		t.Fatal("NewLogger returned nil")
	}
	if l.level != "INFO" {
		t.Errorf("level: got %s, want INFO", l.level)
	}
}

func TestLogger_shouldLog(t *testing.T) {
	l := NewLogger("INFO")
	if !l.shouldLog("INFO") {
		t.Error("INFO should be logged at INFO level")
	}
	if !l.shouldLog("ERROR") {
		t.Error("ERROR should be logged at INFO level")
	}
	if l.shouldLog("DEBUG") {
		t.Error("DEBUG should not be logged at INFO level")
	}
}

func TestLogger_Info(t *testing.T) {
	l := NewLogger("INFO")
	l.Info("test-trace", "test-module", "test_event", map[string]interface{}{
		"key": "value",
	})
}

func TestLogger_Audit(t *testing.T) {
	l := NewLogger("INFO")
	l.Audit("test-trace", "llm_call", map[string]interface{}{
		"model":          "claude-sonnet",
		"prompt_version": "gateway.v1.0",
		"model_version":  "claude-3-5-sonnet-20241022",
		"input_tokens":   100,
		"output_tokens":  50,
	})
}

func TestLogger_Warn(t *testing.T) {
	l := NewLogger("WARN")
	l.Info("test-trace", "module", "event", nil) // 低于 WARN，不应输出
	l.Warn("test-trace", "module", "warn_event", nil)
}

func TestLogger_Error(t *testing.T) {
	l := NewLogger("ERROR")
	l.Error("test-trace", "module", "error_event", nil, map[string]interface{}{
		"status": 500,
	})
}

func TestTraceIDMiddleware(t *testing.T) {
	app := fiber.New()
	app.Use(TraceIDMiddleware())
	app.Get("/test", func(c *fiber.Ctx) error {
		traceID := ExtractTraceID(c)
		if traceID == "" {
			return c.Status(500).SendString("missing trace_id")
		}
		return c.JSON(fiber.Map{"trace_id": traceID})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
	if resp.Header.Get("X-Trace-ID") == "" {
		t.Error("expected X-Trace-ID header in response")
	}
}

func TestTraceIDMiddlewareWithHeader(t *testing.T) {
	app := fiber.New()
	app.Use(TraceIDMiddleware())
	app.Get("/test", func(c *fiber.Ctx) error {
		traceID := ExtractTraceID(c)
		return c.JSON(fiber.Map{"trace_id": traceID})
	})

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Trace-ID", "custom-trace-123")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
	if resp.Header.Get("X-Trace-ID") != "custom-trace-123" {
		t.Errorf("expected custom-trace-123, got %s", resp.Header.Get("X-Trace-ID"))
	}
}

func TestRequestLogMiddleware(t *testing.T) {
	app := fiber.New()
	logger := NewLogger("INFO")
	app.Use(TraceIDMiddleware())
	app.Use(RequestLogMiddleware(logger))
	app.Get("/test", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestRequestLogMiddlewareErrorPath(t *testing.T) {
	app := fiber.New()
	logger := NewLogger("INFO")
	app.Use(TraceIDMiddleware())
	app.Use(RequestLogMiddleware(logger))
	app.Get("/test", func(c *fiber.Ctx) error {
		return c.Status(500).SendString("error")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 500 {
		t.Errorf("expected 500, got %d", resp.StatusCode)
	}
}

func TestRequestLogFiltering(t *testing.T) {
	if shouldLogRequestStart("GET", "/dashboard/api/summary") {
		t.Fatal("dashboard request_start should be suppressed")
	}
	if shouldLogRequestEnd("GET", "/dashboard/api/summary", 200, 1) {
		t.Fatal("dashboard request_end should be suppressed")
	}
	if shouldLogRequestStart("POST", "/v1/messages") {
		t.Fatal("request_start should be suppressed")
	}
	if !shouldLogRequestEnd("POST", "/v1/messages", 200, 100) {
		t.Fatal("/v1/messages request_end should be logged")
	}
	if !shouldLogRequestEnd("GET", "/health", 500, 1) {
		t.Fatal("errors should be logged")
	}
	if !shouldLogRequestEnd("POST", "/v1/responses", 200, 5000) {
		t.Fatal("slow LLM requests should be logged")
	}
}

func TestExtractTraceIDMissing(t *testing.T) {
	app := fiber.New()
	app.Get("/test", func(c *fiber.Ctx) error {
		traceID := ExtractTraceID(c)
		if traceID != "" {
			return c.Status(500).SendString("should be empty")
		}
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestExtractTraceIDTypeAssertion(t *testing.T) {
	app := fiber.New()
	app.Get("/test", func(c *fiber.Ctx) error {
		c.Locals("trace_id", 12345) // wrong type
		traceID := ExtractTraceID(c)
		if traceID != "" {
			return c.Status(500).SendString("should be empty for wrong type")
		}
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestMetricsMiddleware(t *testing.T) {
	app := fiber.New()
	app.Use(MetricsMiddleware())
	app.Get("/test", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestRecordProviderRequest(t *testing.T) {
	RecordProviderRequest("kimi", "moonshot-v1-8k")
}

func TestRecordProviderError(t *testing.T) {
	RecordProviderError("kimi", 429)
}

func TestUpdateKeyPoolHealth(t *testing.T) {
	UpdateKeyPoolHealth("kimi", 2)
	UpdateKeyPoolHealth("deepseek", 1)
}

func TestLoggerWriteError(t *testing.T) {
	// 测试日志写入到无效 writer 的场景
	l := NewLogger("INFO")
	l.output = nil
	// 不应 panic
	l.Info("trace", "module", "event", nil)
}

func TestLoggerLogLevelFilter(t *testing.T) {
	l := NewLogger("ERROR")
	l.Info("trace", "module", "event", nil)     // 不应输出
	l.Warn("trace", "module", "warn", nil)      // 不应输出
	l.Error("trace", "module", "err", nil, nil) // 应输出
	l.Audit("trace", "event", nil)              // AUDIT(4) >= ERROR(3)，应输出
}

func TestLoggerJSONOutput(t *testing.T) {
	app := fiber.New()
	logger := NewLogger("INFO")
	app.Use(TraceIDMiddleware())
	app.Use(RequestLogMiddleware(logger))
	app.Get("/test", func(c *fiber.Ctx) error {
		logger.Info(ExtractTraceID(c), "test", "json_event", map[string]interface{}{"foo": "bar"})
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestSerializeAnthropicStreamChunkError(t *testing.T) {
	// 测试无效的 SSE 数据行
	l := NewLogger("INFO")
	l.Info("trace", "module", "event", map[string]interface{}{"data": "invalid"})
}

func TestMetricsMiddlewareWithRoutePath(t *testing.T) {
	app := fiber.New()
	app.Use(MetricsMiddleware())
	app.Get("/v1/:name", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/v1/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestRequestLogMiddlewareBodyLimit(t *testing.T) {
	app := fiber.New()
	logger := NewLogger("INFO")
	app.Use(TraceIDMiddleware())
	app.Use(RequestLogMiddleware(logger))
	app.Post("/test", func(c *fiber.Ctx) error {
		return c.SendString("ok")
	})

	body := strings.NewReader(`{"test":"data"}`)
	req := httptest.NewRequest("POST", "/test", body)
	req.Header.Set("Content-Type", "application/json")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}
