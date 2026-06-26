package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gofiber/fiber/v2"
	"llm-gateway/internal/auth"
	"llm-gateway/internal/config"
	"llm-gateway/internal/forwarder"
	mw "llm-gateway/internal/middleware"
	"llm-gateway/internal/models"
	"llm-gateway/internal/router"
	"llm-gateway/internal/usage"
)

const testVirtualKey = "sk-virtual-test"

func setupRealGatewayApp(t *testing.T, upstream *httptest.Server, modelMap map[string]string) (*fiber.App, *usage.Store) {
	t.Helper()

	store, err := usage.Open(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatalf("open usage store: %v", err)
	}

	authService := auth.NewAuth(&config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           testVirtualKey,
				AllowedModels: []string{"claude-sonnet", "gpt-5.5", "gpt-4o"},
				RPMLimit:      100,
			},
		},
	})

	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL:  upstream.URL,
				ModelMap: modelMap,
				Keys:     []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
			},
			DeepSeek: config.ProviderConfig{},
		},
		Router: config.RouterConfig{
			Strategy:        "round_robin",
			RetryCount:      1,
			TimeoutSeconds:  30,
			CooldownSeconds: 60,
			ProviderOrder:   []string{"kimi"},
		},
		ModelCapabilities: map[string]config.ModelCapability{
			"claude-sonnet": {SupportsCache: true},
		},
	}

	r := router.NewRouter(cfg, nil)
	r.DisableRateLimitForTest()

	fwd := forwarder.NewForwarder(r, 10*time.Second, cfg.Router.RetryCount)
	logger := mw.NewLogger("INFO")

	app := fiber.New(fiber.Config{
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	})
	app.Use(mw.TraceIDMiddleware())
	app.Use(mw.RequestLogMiddleware(logger))

	app.Post("/v1/messages", authService.Middleware(), handleAnthropicMessages(fwd, logger, r, store))
	app.Post("/v1/responses", authService.Middleware(), handleResponses(fwd, logger, r, store))

	return app, store
}

func setupTestApp() *fiber.App {
	app := fiber.New(fiber.Config{
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
	})

	logger := mw.NewLogger("INFO")
	app.Use(mw.TraceIDMiddleware())
	app.Use(mw.RequestLogMiddleware(logger))

	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           "sk-virtual-test",
				AllowedModels: []string{"gpt-4", "claude-sonnet"},
				RPMLimit:      100,
			},
		},
	}
	authService := auth.NewAuth(cfg)

	// Health endpoint
	app.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{"status": "ok"})
	})

	// Mock OpenAI endpoint
	app.Post("/v1/chat/completions", authService.Middleware(), func(c *fiber.Ctx) error {
		var req models.OpenAIChatCompletionRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid body"})
		}

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
				return c.Status(fiber.StatusForbidden).JSON(fiber.Map{"error": "model not allowed"})
			}
		}

		return c.JSON(models.OpenAIChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   req.Model,
			Choices: []models.OpenAIChoice{
				{
					Index: 0,
					Message: models.OpenAIMessage{
						Role:    "assistant",
						Content: "Test response",
					},
					FinishReason: "stop",
				},
			},
		})
	})

	// Mock Anthropic endpoint
	app.Post("/v1/messages", authService.Middleware(), func(c *fiber.Ctx) error {
		var req models.AnthropicMessageRequest
		if err := c.BodyParser(&req); err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid body"})
		}
		return c.JSON(models.AnthropicMessageResponse{
			ID:      "msg-test",
			Type:    "message",
			Role:    "assistant",
			Model:   req.Model,
			Content: []models.AnthropicContent{{Type: "text", Text: "Test response"}},
			Usage:   models.AnthropicUsage{InputTokens: 10, OutputTokens: 5},
		})
	})

	return app
}

func TestAnthropicMessagesEndToEndNonStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected /chat/completions, got %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req models.OpenAIChatCompletionRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("invalid request body: %v", err)
		}
		if req.Model != "claude-sonnet-mapped" {
			t.Errorf("expected mapped model claude-sonnet-mapped, got %s", req.Model)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:      "chatcmpl-test",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "claude-sonnet-mapped",
			Choices: []models.OpenAIChoice{
				{
					Index:        0,
					Message:      models.OpenAIMessage{Role: "assistant", Content: "Hello from Anthropic"},
					FinishReason: "stop",
				},
			},
			Usage: models.OpenAIUsage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		})
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"claude-sonnet": "claude-sonnet-mapped"})

	reqBody, _ := json.Marshal(models.AnthropicMessageRequest{
		Model:     "claude-sonnet",
		Messages:  []models.AnthropicMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 100,
	})
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	var result models.AnthropicMessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Role != "assistant" {
		t.Errorf("expected role assistant, got %s", result.Role)
	}
	if len(result.Content) != 1 || result.Content[0].Text != "Hello from Anthropic" {
		t.Errorf("unexpected content: %+v", result.Content)
	}
	if result.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", result.StopReason)
	}
	if result.Usage.InputTokens != 10 || result.Usage.OutputTokens != 5 {
		t.Errorf("unexpected usage: %+v", result.Usage)
	}
}

func TestAnthropicMessagesEndToEndStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"claude-sonnet-mapped\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"chatcmpl-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"claude-sonnet-mapped\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"claude-sonnet": "claude-sonnet-mapped"})

	reqBody, _ := json.Marshal(models.AnthropicMessageRequest{
		Model:     "claude-sonnet",
		Stream:    true,
		Messages:  []models.AnthropicMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 100,
	})
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	body, _ := io.ReadAll(resp.Body)
	got := string(body)
	if !strings.Contains(got, "Hello") || !strings.Contains(got, " world") {
		t.Errorf("expected streamed text 'Hello world', got %s", got)
	}
	if !strings.Contains(got, `"type":"message_stop"`) {
		t.Errorf("expected message_stop event, got %s", got)
	}
}

func TestAnthropicMessagesEndToEndToolUse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:      "chatcmpl-tool",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "claude-sonnet-mapped",
			Choices: []models.OpenAIChoice{
				{
					Index: 0,
					Message: models.OpenAIMessage{
						Role:    "assistant",
						Content: "",
						ToolCalls: []models.ToolCall{
							{
								ID:   "call_123",
								Type: "function",
								Function: models.FunctionCall{
									Name:      "shell_command",
									Arguments: `{"command":"pwd"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: models.OpenAIUsage{PromptTokens: 20, CompletionTokens: 10, TotalTokens: 30},
		})
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"claude-sonnet": "claude-sonnet-mapped"})

	reqBody, _ := json.Marshal(models.AnthropicMessageRequest{
		Model:     "claude-sonnet",
		MaxTokens: 100,
		Messages:  []models.AnthropicMessage{{Role: "user", Content: "Run pwd"}},
		Tools: []models.AnthropicTool{
			{
				Name:        "shell_command",
				Description: "Run shell command",
				InputSchema: map[string]interface{}{"type": "object"},
			},
		},
	})
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	var result models.AnthropicMessageResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.StopReason != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %s", result.StopReason)
	}
	if len(result.Content) != 1 || result.Content[0].Type != "tool_use" {
		t.Fatalf("expected tool_use content, got %+v", result.Content)
	}
	if result.Content[0].Name != "shell_command" {
		t.Errorf("expected tool name shell_command, got %s", result.Content[0].Name)
	}
}

func TestAnthropicMessagesEndToEndModelNotAllowed(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream should not be called for disallowed model")
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"claude-sonnet": "claude-sonnet-mapped"})

	reqBody, _ := json.Marshal(models.AnthropicMessageRequest{
		Model:     "claude-opus",
		Messages:  []models.AnthropicMessage{{Role: "user", Content: "Hi"}},
		MaxTokens: 100,
	})
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 403 {
		t.Errorf("expected 403, got %d", resp.StatusCode)
	}
}

func TestResponsesEndToEndNonStreamText(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected /chat/completions, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:      "chatcmpl-resp",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "kimi-for-coding",
			Choices: []models.OpenAIChoice{
				{
					Index:        0,
					Message:      models.OpenAIMessage{Role: "assistant", Content: "Hello from Responses"},
					FinishReason: "stop",
				},
			},
			Usage: models.OpenAIUsage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
		})
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"gpt-5.5": "kimi-for-coding"})

	reqBody, _ := json.Marshal(models.ResponsesRequest{
		Model:        "gpt-5.5",
		Instructions: "Be helpful",
		Input:        "Hi",
	})
	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	var result models.ResponsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Model != "kimi-for-coding" {
		t.Errorf("expected model kimi-for-coding, got %s", result.Model)
	}
	if len(result.Output) != 1 || result.Output[0].Type != "message" {
		t.Fatalf("expected 1 message output, got %+v", result.Output)
	}
	if len(result.Output[0].Content) != 1 || result.Output[0].Content[0].Text != "Hello from Responses" {
		t.Errorf("unexpected output content: %+v", result.Output[0].Content)
	}
}

func TestResponsesEndToEndFunctionCall(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:      "chatcmpl-func",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "kimi-for-coding",
			Choices: []models.OpenAIChoice{
				{
					Index: 0,
					Message: models.OpenAIMessage{
						Role:    "assistant",
						Content: "",
						ToolCalls: []models.ToolCall{
							{
								ID:   "call_456",
								Type: "function",
								Function: models.FunctionCall{
									Name:      "shell_command",
									Arguments: `{"command":"ls"}`,
								},
							},
						},
					},
					FinishReason: "tool_calls",
				},
			},
			Usage: models.OpenAIUsage{PromptTokens: 15, CompletionTokens: 8, TotalTokens: 23},
		})
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"gpt-5.5": "kimi-for-coding"})

	reqBody, _ := json.Marshal(models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "List files"},
		},
		Tools: []models.ResponsesTool{
			{
				Type:        "function",
				Name:        "shell_command",
				Description: "Run a command",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"command": map[string]interface{}{"type": "string"},
					},
					"required": []string{"command"},
				},
			},
		},
	})
	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	var result models.ResponsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(result.Output) != 2 {
		t.Fatalf("expected 2 outputs (message + function_call), got %d", len(result.Output))
	}
	if result.Output[1].Type != "function_call" {
		t.Errorf("expected function_call output, got %s", result.Output[1].Type)
	}
	if result.Output[1].Name != "shell_command" || result.Output[1].Arguments != `{"command":"ls"}` {
		t.Errorf("unexpected function_call: %+v", result.Output[1])
	}
}

func TestResponsesEndToEndStream(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"chatcmpl-resp-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi-for-coding\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"chatcmpl-resp-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"kimi-for-coding\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" Responses\"},\"finish_reason\":\"stop\"}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	app, _ := setupRealGatewayApp(t, upstream, map[string]string{"gpt-5.5": "kimi-for-coding"})

	reqBody, _ := json.Marshal(models.ResponsesRequest{
		Model:  "gpt-5.5",
		Input:  "Hi",
		Stream: true,
	})
	req := httptest.NewRequest("POST", "/v1/responses", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+testVirtualKey)

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	body, _ := io.ReadAll(resp.Body)
	got := string(body)
	if !strings.Contains(got, "Hello") || !strings.Contains(got, "Responses") {
		t.Errorf("expected streamed text, got %s", got)
	}
	if !strings.Contains(got, "[DONE]") {
		t.Errorf("expected [DONE] terminator, got %s", got)
	}
}

func TestHealthEndpoint(t *testing.T) {
	app := setupTestApp()
	reqBody, _ := json.Marshal(models.OpenAIChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hello"}},
	})
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 401 {
		t.Errorf("expected status 401, got %d", resp.StatusCode)
	}
}

func TestOpenAIChatCompletionsWithAuth(t *testing.T) {
	app := setupTestApp()
	reqBody, _ := json.Marshal(models.OpenAIChatCompletionRequest{
		Model:    "gpt-4",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hello"}},
	})
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-virtual-test")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		t.Errorf("expected status 200, got %d, body: %s", resp.StatusCode, string(body))
	}

	var result models.OpenAIChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&result)
	if result.Model != "gpt-4" {
		t.Errorf("expected model gpt-4, got %s", result.Model)
	}
}

func TestOpenAIChatCompletionsModelNotAllowed(t *testing.T) {
	app := setupTestApp()
	reqBody, _ := json.Marshal(models.OpenAIChatCompletionRequest{
		Model:    "gpt-3.5",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hello"}},
	})
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-virtual-test")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 403 {
		t.Errorf("expected status 403, got %d", resp.StatusCode)
	}
}

func TestAnthropicMessagesWithAuth(t *testing.T) {
	app := setupTestApp()
	reqBody, _ := json.Marshal(models.AnthropicMessageRequest{
		Model:    "claude-sonnet",
		Messages: []models.AnthropicMessage{{Role: "user", Content: "Hello"}},
	})
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-virtual-test")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
}

func TestAnthropicMessagesInvalidBody(t *testing.T) {
	app := setupTestApp()
	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte("invalid")))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-virtual-test")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 400 {
		t.Errorf("expected status 400, got %d", resp.StatusCode)
	}
}

func TestAnthropicStreamToolCallKeepsArgumentContinuation(t *testing.T) {
	app := fiber.New()
	app.Get("/stream", func(c *fiber.Ctx) error {
		state := newAnthropicStreamState()
		state.start(c, &models.OpenAIStreamChunk{ID: "chatcmpl-test", Model: "claude-sonnet"}, models.AnthropicMessageRequest{Model: "claude-sonnet"})
		state.emitToolCall(c, models.ToolCall{
			Index: 0,
			ID:    "call_123",
			Type:  "function",
			Function: models.FunctionCall{
				Name: "Bash",
			},
		})
		state.emitToolCall(c, models.ToolCall{
			Index: 0,
			Function: models.FunctionCall{
				Arguments: `{"command":"pwd"}`,
			},
		})
		state.finish(c, "tool_calls", nil)
		return nil
	})

	resp, err := app.Test(httptest.NewRequest("GET", "/stream", nil))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	got := string(body)
	if !strings.Contains(got, `"partial_json":"{\"command\":\"pwd\"}"`) {
		t.Fatalf("expected continued tool arguments in stream, got:\n%s", got)
	}
	if !strings.Contains(got, `"stop_reason":"tool_use"`) {
		t.Fatalf("expected tool_use stop reason, got:\n%s", got)
	}
}

func TestRequestShapeLoggingPolicy(t *testing.T) {
	if !shouldLogRequestShape("trace-error", &usage.RequestLog{StatusCode: 500}) {
		t.Fatal("error request shapes should be logged")
	}
	if !shouldLogRequestShape("trace-slow", &usage.RequestLog{StatusCode: 200, LatencyMS: 10000}) {
		t.Fatal("slow request shapes should be logged")
	}
	if !shouldLogRequestShape("trace-large", &usage.RequestLog{StatusCode: 200, PromptTokens: 50000}) {
		t.Fatal("large request shapes should be logged")
	}
	if isShapeSample("stable-trace") != isShapeSample("stable-trace") {
		t.Fatal("shape sampling should be deterministic")
	}
}

func TestRequestShapeIncludesToolSchemaSize(t *testing.T) {
	anthropicShape := anthropicRequestShape(&models.AnthropicMessageRequest{
		Tools: []models.AnthropicTool{{
			Name:        "Bash",
			Description: "run command",
			InputSchema: map[string]interface{}{
				"type": "object",
			},
		}},
	})
	if anthropicShape["tool_schema_chars"].(int) <= 0 {
		t.Fatalf("expected anthropic tool schema chars, got %+v", anthropicShape)
	}

	openAIShape := openAIRequestShape(&models.OpenAIChatCompletionRequest{
		Tools: []models.OpenAITool{{
			Type: "function",
			Function: models.OpenAIFunction{
				Name: "Bash",
				Parameters: map[string]interface{}{
					"type": "object",
				},
			},
		}},
	})
	if openAIShape["tool_schema_chars"].(int) <= 0 {
		t.Fatalf("expected openai tool schema chars, got %+v", openAIShape)
	}
}

func TestTraceIDHeader(t *testing.T) {
	app := setupTestApp()
	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Set("X-Trace-ID", "test-trace-123")

	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected status 200, got %d", resp.StatusCode)
	}
	// Fiber 的 Test 返回的 Response.Header 包含服务端设置的 Header
	traceID := resp.Header.Get("X-Trace-ID")
	if traceID == "" {
		t.Error("expected X-Trace-ID header in response")
	}
}

func TestSetupGatewayInvalidConfigPath(t *testing.T) {
	_, err := setupGateway("/nonexistent/config.yaml", "INFO")
	if err == nil {
		t.Error("expected error for invalid config path")
	}
}

func TestSetupGatewaySuccess(t *testing.T) {
	// 创建临时配置文件
	configContent := `
server:
  host: "127.0.0.1"
  port: 18080
auth:
  virtual_keys:
    - key: "sk-test"
      allowed_models: ["gpt-4"]
      rpm_limit: 100
providers:
  kimi:
    base_url: "http://localhost:18081"
    keys:
      - id: "k1"
        key: "sk-k1"
        weight: 1
        rpm_limit: 60
    model_map:
      "gpt-4": "moonshot-v1-8k"
  deepseek:
    base_url: "http://localhost:18082"
    keys:
      - id: "d1"
        key: "sk-d1"
        weight: 1
        rpm_limit: 60
    model_map:
      "gpt-4o": "deepseek-chat"
redis:
  host: "localhost"
  port: 6379
  db: 0
router:
  strategy: "round_robin"
  retry_count: 1
  timeout_seconds: 5
logging:
  level: "INFO"
  format: "json"
`
	tmpFile := t.TempDir() + "/test-config.yaml"
	if err := os.WriteFile(tmpFile, []byte(configContent), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}

	deps, err := setupGateway(tmpFile, "INFO")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if deps == nil {
		t.Fatal("expected non-nil deps")
	}
	if deps.cfg == nil {
		t.Fatal("expected non-nil cfg")
	}
	if deps.cfg.Server.Port != 18080 {
		t.Errorf("expected port 18080, got %d", deps.cfg.Server.Port)
	}
	if deps.authService == nil {
		t.Fatal("expected non-nil authService")
	}
	if deps.router == nil {
		t.Fatal("expected non-nil router")
	}
	if deps.forwarder == nil {
		t.Fatal("expected non-nil forwarder")
	}
	if deps.logger == nil {
		t.Fatal("expected non-nil logger")
	}
	deps.forwarder.Close()
}

func TestHashMessages(t *testing.T) {
	messages := []models.OpenAIMessage{
		{Role: "system", Content: "Be helpful"},
		{Role: "user", Content: "Hello"},
	}
	h := hashMessages(messages)
	if h == "" {
		t.Error("expected non-empty hash")
	}
	// 相同输入应产生相同输出
	h2 := hashMessages(messages)
	if h != h2 {
		t.Error("expected same hash for same input")
	}
}

func TestHashAnthropicMessages(t *testing.T) {
	messages := []models.AnthropicMessage{
		{Role: "user", Content: "Hello"},
	}
	h := hashAnthropicMessages(messages, "Be helpful")
	if h == "" {
		t.Error("expected non-empty hash")
	}
	// 相同输入应产生相同输出
	h2 := hashAnthropicMessages(messages, "Be helpful")
	if h != h2 {
		t.Error("expected same hash for same input")
	}
	// system 为空时应不同
	h3 := hashAnthropicMessages(messages, "")
	if h == h3 {
		t.Error("expected different hash for different system")
	}
}

func TestHashOpenAIResponseContent(t *testing.T) {
	resp := &models.OpenAIChatCompletionResponse{
		Choices: []models.OpenAIChoice{
			{Message: models.OpenAIMessage{Content: "Hello world"}},
		},
	}
	h := hashOpenAIResponseContent(resp)
	if h == "" {
		t.Error("expected non-empty hash")
	}
}

func TestHashAnthropicResponseContent(t *testing.T) {
	resp := &models.AnthropicMessageResponse{
		Content: []models.AnthropicContent{{Type: "text", Text: "Hello world"}},
	}
	h := hashAnthropicResponseContent(resp)
	if h == "" {
		t.Error("expected non-empty hash")
	}
}

func TestHashKey(t *testing.T) {
	if hashKey("short") != "***" {
		t.Errorf("expected *** for short key, got %s", hashKey("short"))
	}
	result := hashKey("sk-test-key-12345")
	expected := "sk-test-***"
	if result != expected {
		t.Errorf("expected %s, got %s", expected, result)
	}
}

func TestDashboardModuleAPIs(t *testing.T) {
	app := fiber.New()
	store, err := usage.Open(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatalf("open usage store: %v", err)
	}
	defer store.Close()

	cfg := &config.Config{
		Auth: config.AuthConfig{
			VirtualKeys: []config.VirtualKey{{
				Key:           "sk-virtual-test",
				User:          "team-a",
				AllowedModels: []string{"gpt-5.5"},
				RPMLimit:      100,
			}},
		},
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL:  "http://kimi.example/v1",
				Keys:     []config.ProviderKey{{ID: "kimi-1", Key: "sk-kimi"}},
				ModelMap: map[string]string{"gpt-5.5": "kimi-for-coding"},
			},
			DeepSeek: config.ProviderConfig{
				BaseURL: "http://deepseek.example/v1",
				Keys:    []config.ProviderKey{{ID: "deepseek-1", Key: "sk-deepseek"}},
			},
		},
		Router: config.RouterConfig{
			ProviderOrder:  []string{"kimi", "deepseek"},
			TimeoutSeconds: 30,
			RetryCount:     1,
		},
	}
	if err := store.SeedConfigVirtualKeys(context.Background(), cfg.Auth.VirtualKeys); err != nil {
		t.Fatalf("seed virtual keys: %v", err)
	}
	if err := store.Record(context.Background(), usage.RequestLog{
		TraceID:         "trace-dashboard",
		CreatedAt:       time.Now().UTC(),
		Method:          "POST",
		Path:            "/v1/responses",
		Protocol:        "responses",
		VirtualKeyHash:  auth.HashKey("sk-virtual-test"),
		User:            "team-a",
		APIKey:          "sk-...test",
		ClientIP:        "203.0.113.20",
		Model:           "gpt-5.5",
		Provider:        "kimi",
		ProviderKeyID:   "kimi-1",
		UpstreamModel:   "kimi-for-coding",
		RouterDecision:  "kimi/kimi-1",
		StatusCode:      502,
		LatencyMS:       2500,
		SlowReason:      "slow_request",
		PromptTokens:    100,
		OutputTokens:    20,
		TotalTokens:     120,
		CacheReadTokens: 10,
		HasImage:        true,
		RequestBody:     `{"model":"gpt-5.5"}`,
		ResponseBody:    `{"error":"provider failed"}`,
		ErrorCode:       "gateway_error",
		ErrorMessage:    "provider failed",
	}); err != nil {
		t.Fatalf("record request: %v", err)
	}

	registerDashboard(app, cfg, store, auth.NewAuth(&cfg.Auth), router.NewRouter(cfg, nil))
	for _, path := range []string{
		"/dashboard/api/usage?days=14&limit=5",
		"/dashboard/api/providers?hours=24",
		"/dashboard/api/keys-overview",
		"/dashboard/api/errors?hours=24&limit=5",
	} {
		req := httptest.NewRequest("GET", path, nil)
		resp, err := app.Test(req)
		if err != nil {
			t.Fatalf("%s request failed: %v", path, err)
		}
		if resp.StatusCode != 200 {
			t.Fatalf("%s expected 200, got %d", path, resp.StatusCode)
		}
		var body map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
			t.Fatalf("%s returned invalid json: %v", path, err)
		}
		if len(body) == 0 {
			t.Fatalf("%s returned empty object", path)
		}
	}
}

func TestDashboardHTMLUsesModuleAPIsAndStoredToken(t *testing.T) {
	html := dashboardHTML()
	for _, want := range []string{
		"localStorage.getItem('llm_gateway_dashboard_token')",
		"/dashboard/api/usage",
		"/dashboard/api/providers",
		"/dashboard/api/keys-overview",
		"/dashboard/api/errors",
		"<section id=\"usage\" class=\"view\">",
	} {
		if !strings.Contains(html, want) {
			t.Fatalf("dashboard html missing %q", want)
		}
	}
}

func TestRequestDetailHTMLUsesTraceID(t *testing.T) {
	html := requestDetailHTML("trace-123")
	for _, want := range []string{
		"Request Detail",
		"trace-123",
		"/dashboard/api/requests/",
		"Basic Info",
		"Request Body",
		"Response Body",
	} {
		if !strings.Contains(html, want) {
			t.Fatalf("detail html missing %q", want)
		}
	}
}

func TestRequestDetailRouteServesHTML(t *testing.T) {
	app := fiber.New()
	cfg := &config.Config{}
	store, err := usage.Open(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatalf("open usage store: %v", err)
	}
	defer store.Close()
	registerDashboard(app, cfg, store, auth.NewAuth(&cfg.Auth), router.NewRouter(cfg, nil))
	req := httptest.NewRequest("GET", "/dashboard/requests/trace-123", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("request detail route failed: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Request Detail") {
		t.Fatalf("detail route missing html body")
	}
}
