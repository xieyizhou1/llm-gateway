package forwarder

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"llm-gateway/internal/config"
	"llm-gateway/internal/models"
	"llm-gateway/internal/router"
)

func TestForwarder_ForwardOpenAIRequest(t *testing.T) {
	// 模拟上游 Provider
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:      "test-id",
			Object:  "chat.completion",
			Created: 1234567890,
			Model:   "deepseek-chat",
			Choices: []models.OpenAIChoice{
				{
					Index: 0,
					Message: models.OpenAIMessage{
						Role:    "assistant",
						Content: "Hello from test",
					},
					FinishReason: "stop",
				},
			},
			Usage: models.OpenAIUsage{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		})
	}))
	defer upstream.Close()

	// 创建带有一个健康 Key 的 Router
	pool := router.NewKeyPool("test", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60},
		},
	})
	r := &router.Router{}
	// 直接替换 pools（测试专用）
	r.SetPoolForTest("kimi", pool)
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	resp, err := fwd.ForwardOpenAIRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("ForwardOpenAIRequest failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var result models.OpenAIChatCompletionResponse
	if err := json.Unmarshal(body, &result); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if result.Choices[0].Message.Content != "Hello from test" {
		t.Errorf("content: got %s", result.Choices[0].Message.Content)
	}
}

func TestForwarder_RetryOn500(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:     "test-id",
			Object: "chat.completion",
			Choices: []models.OpenAIChoice{
				{Message: models.OpenAIMessage{Role: "assistant", Content: "OK"}},
			},
		})
	}))
	defer upstream.Close()

	pool := router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60},
		},
	})
	r := &router.Router{}
	r.SetPoolForTest("kimi", pool)
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 2)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	resp, err := fwd.ForwardOpenAIRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("ForwardOpenAIRequest failed: %v", err)
	}
	defer resp.Body.Close()

	if callCount != 2 {
		t.Errorf("retry count: got %d, want 2", callCount)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}
}

func TestForwarder_RetryOn429SwitchesKey(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"error":{"message":"quota exhausted","type":"exceeded_current_quota_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:     "test-id",
			Object: "chat.completion",
			Choices: []models.OpenAIChoice{
				{Message: models.OpenAIMessage{Role: "assistant", Content: "OK"}},
			},
		})
	}))
	defer upstream.Close()

	r := &router.Router{}
	r.SetPoolForTest("kimi", router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-test-1", Weight: 1, RPMLimit: 60},
			{ID: "key-2", Key: "sk-test-2", Weight: 1, RPMLimit: 60},
		},
	}))
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	resp, err := fwd.ForwardOpenAIRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("ForwardOpenAIRequest failed: %v", err)
	}
	defer resp.Body.Close()

	if callCount != 2 {
		t.Fatalf("expected retry on 429 to reach second key, got %d calls", callCount)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 after retry, got %d", resp.StatusCode)
	}
}

func TestForwarder_ForwardOpenAIRequestRaw(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var reqMap map[string]interface{}
		json.Unmarshal(body, &reqMap)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:     "raw-test",
			Object: "chat.completion",
			Choices: []models.OpenAIChoice{
				{Message: models.OpenAIMessage{Role: "assistant", Content: "Raw OK"}},
			},
		})
	}))
	defer upstream.Close()

	pool := router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60},
		},
	})
	r := &router.Router{}
	r.SetPoolForTest("kimi", pool)
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	rawBody := []byte(`{"model":"gpt-4o","messages":[{"role":"user","content":"Hi"}]}`)
	resp, err := fwd.ForwardOpenAIRequestRaw(context.Background(), rawBody, "gpt-4o")
	if err != nil {
		t.Fatalf("ForwardOpenAIRequestRaw failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("status: got %d, want 200", resp.StatusCode)
	}

	var result models.OpenAIChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.Choices[0].Message.Content != "Raw OK" {
		t.Errorf("content: got %s", result.Choices[0].Message.Content)
	}
}

func TestForwarder_NormalizeHTTPImageURL(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Write([]byte{
			0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
			0x00, 0x00, 0x00, 0x00,
		})
	}))
	defer imageServer.Close()

	raw := []byte(`[
		{"type":"image_url","image_url":{"url":"` + imageServer.URL + `/test.png"}},
		{"type":"text","text":"describe this"}
	]`)

	fwd := NewForwarder(&router.Router{}, 10*time.Second, 0)
	defer fwd.Close()
	normalized, changed, err := fwd.normalizeMessageImageURLs(context.Background(), raw)
	if err != nil {
		t.Fatalf("normalize image url: %v", err)
	}
	if !changed {
		t.Fatal("expected image url to be normalized")
	}
	if !strings.Contains(string(normalized), "data:image/png;base64,") {
		t.Fatalf("expected data image url, got %s", string(normalized))
	}
	if strings.Contains(string(normalized), imageServer.URL) {
		t.Fatalf("expected original http url to be replaced, got %s", string(normalized))
	}
}

func TestForwarder_Warmup(t *testing.T) {
	var gotAuth string
	var path string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer upstream.Close()

	r := &router.Router{}
	r.SetPoolForTest("kimi", router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-warmup", Weight: 1, RPMLimit: 60},
		},
	}))
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "sk-warmup", Weight: 1, RPMLimit: 60},
		},
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 0)
	defer fwd.Close()

	fwd.Warmup(context.Background())

	if path != "/models" {
		t.Errorf("warmup path: got %q, want /models", path)
	}
	if gotAuth != "Bearer sk-warmup" {
		t.Errorf("warmup auth: got %q, want Bearer sk-warmup", gotAuth)
	}
}

func TestForwarder_ForwardOpenAIRequestWithInfo(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(models.OpenAIChatCompletionResponse{
			ID:     "test-id",
			Object: "chat.completion",
			Model:  "deepseek-chat",
			Choices: []models.OpenAIChoice{
				{Message: models.OpenAIMessage{Role: "assistant", Content: "OK"}, FinishReason: "stop"},
			},
			Usage: models.OpenAIUsage{PromptTokens: 5, CompletionTokens: 1, TotalTokens: 6},
		})
	}))
	defer upstream.Close()

	r := &router.Router{}
	r.SetPoolForTest("kimi", router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys:    []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
	}))
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	result, err := fwd.ForwardOpenAIRequestWithInfo(context.Background(), req)
	if err != nil {
		t.Fatalf("ForwardOpenAIRequestWithInfo failed: %v", err)
	}
	defer result.Response.Body.Close()

	if result.Info.Provider != "kimi" {
		t.Errorf("expected provider kimi, got %s", result.Info.Provider)
	}
	if result.Info.ProviderKeyID != "key-1" {
		t.Errorf("expected key id key-1, got %s", result.Info.ProviderKeyID)
	}
	if result.Info.UpstreamModel != "deepseek-chat" {
		t.Errorf("expected upstream model deepseek-chat, got %s", result.Info.UpstreamModel)
	}
	if !strings.Contains(result.Info.RouterDecision, "gpt-4o -> deepseek-chat") {
		t.Errorf("unexpected router decision: %s", result.Info.RouterDecision)
	}
}

func TestForwarder_StreamResponseSSE(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"}}]}\n\n"))
		w.Write([]byte("data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":\"stop\"}]}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer upstream.Close()

	r := &router.Router{}
	r.SetPoolForTest("kimi", router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys:    []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
	}))
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "gpt-4"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Stream:   true,
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	resp, err := fwd.ForwardOpenAIRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("ForwardOpenAIRequest failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Hello") || !strings.Contains(string(body), " world") {
		t.Errorf("expected streamed content to contain 'Hello' and ' world', got %s", string(body))
	}
	if !strings.Contains(string(body), "[DONE]") {
		t.Errorf("expected [DONE] terminator, got %s", string(body))
	}
}

func TestForwarder_NormalizeImageURLUnsupportedContentType(t *testing.T) {
	imageServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("not an image"))
	}))
	defer imageServer.Close()

	fwd := NewForwarder(&router.Router{}, 10*time.Second, 0)
	defer fwd.Close()

	raw := []byte(`[{"type":"image_url","image_url":{"url":"` + imageServer.URL + `/test.png"}}]`)
	_, _, err := fwd.normalizeMessageImageURLs(context.Background(), raw)
	if err == nil {
		t.Fatal("expected error for unsupported image content type")
	}
}

func TestForwarder_401DisablesKey(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"error":"invalid key"}`))
	}))
	defer upstream.Close()

	pool := router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys:    []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
	})
	r := &router.Router{}
	r.SetPoolForTest("kimi", pool)
	r.DisableRateLimitForTest()
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL:  upstream.URL,
		ModelMap: map[string]string{"gpt-4o": "deepseek-chat"},
	})

	fwd := NewForwarder(r, 10*time.Second, 1)
	defer fwd.Close()

	req := &models.OpenAIChatCompletionRequest{
		Model:    "gpt-4o",
		Messages: []models.OpenAIMessage{{Role: "user", Content: "Hi"}},
	}

	resp, err := fwd.ForwardOpenAIRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("expected retryable response to be returned, got error: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected 401 response, got %d", resp.StatusCode)
	}

	if pool.HealthyCount() != 0 {
		t.Errorf("expected key to be disabled, healthy count = %d", pool.HealthyCount())
	}
}

func TestForwarder_HeartbeatMarksKeyResult(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			t.Errorf("expected /models path, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"object":"list","data":[]}`))
	}))
	defer upstream.Close()

	pool := router.NewKeyPool("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys:    []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
	})
	r := &router.Router{}
	r.SetPoolForTest("kimi", pool)
	r.SetModelMapForTest("kimi", config.ProviderConfig{
		BaseURL: upstream.URL,
		Keys:    []config.ProviderKey{{ID: "key-1", Key: "sk-test", Weight: 1, RPMLimit: 60}},
	})

	fwd := NewForwarder(r, 10*time.Second, 0)
	defer fwd.Close()

	fwd.StartHeartbeat(context.Background(), 10*time.Millisecond)
	time.Sleep(50 * time.Millisecond)

	keys := pool.AllKeys()
	if len(keys) != 1 {
		t.Fatalf("expected 1 key, got %d", len(keys))
	}
	if keys[0].LastSeenAt.IsZero() {
		t.Error("expected LastSeenAt to be set after heartbeat")
	}
}

func TestForwarder_StreamResponse(t *testing.T) {
	src := io.NopCloser(strings.NewReader("Hello, World!"))

	rec := httptest.NewRecorder()
	rec.Flush() // ensure it implements http.Flusher

	err := StreamResponse(src, rec)
	if err != nil {
		t.Fatalf("StreamResponse failed: %v", err)
	}

	body := rec.Body.String()
	if body != "Hello, World!" {
		t.Errorf("expected 'Hello, World!', got %s", body)
	}
}

func TestForwarder_StreamResponseNonFlusher(t *testing.T) {
	src := io.NopCloser(strings.NewReader("test"))
	// httptest.ResponseRecorder implements Flusher, so use a custom writer
	type nonFlusherWriter struct {
		http.ResponseWriter
	}
	w := &nonFlusherWriter{ResponseWriter: httptest.NewRecorder()}

	err := StreamResponse(src, w)
	if err == nil {
		t.Error("expected error for non-flusher response writer")
	}
}
