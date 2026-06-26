// Package forwarder 实现 HTTP 请求转发到上游 LLM Provider。
// 使用 net/http 直接发送请求，支持 SSE 流式响应透传。
package forwarder

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"llm-gateway/internal/config"
	"llm-gateway/internal/models"
	"llm-gateway/internal/router"
)

// traceIDKey is the context key used to carry the gateway trace_id into the
// forwarder for logging purposes.
type traceIDKey struct{}

// WithTraceID returns a context that carries the gateway trace_id.
func WithTraceID(ctx context.Context, traceID string) context.Context {
	return context.WithValue(ctx, traceIDKey{}, traceID)
}

func traceIDFromContext(ctx context.Context) string {
	if v, ok := ctx.Value(traceIDKey{}).(string); ok {
		return v
	}
	return ""
}

// Forwarder 负责将请求转发到上游 Provider。
type Forwarder struct {
	client       *http.Client
	router       *router.Router
	timeout      time.Duration
	retryCount   int
	logger       Logger
	warmupPath   string
	heartbeatStop chan struct{}
}

// Logger is the subset of the middleware logger used by the forwarder.
type Logger interface {
	Debug(traceID, module, event string, data map[string]interface{})
	Info(traceID, module, event string, data map[string]interface{})
	Error(traceID, module, event string, err error, data map[string]interface{})
}

// ResponseInfo describes the selected upstream route for observability.
type ResponseInfo struct {
	Provider       string
	ProviderKeyID  string
	UpstreamModel  string
	RouterDecision string
	FallbackCount  int
}

// ForwardResponse wraps an upstream response with routing metadata.
type ForwardResponse struct {
	Response *http.Response
	Info     ResponseInfo
}

// NewForwarder 创建 Forwarder 实例。
func NewForwarder(r *router.Router, timeout time.Duration, retryCount int) *Forwarder {
	return &Forwarder{
		client:        newPooledHTTPClient(timeout),
		router:        r,
		timeout:       timeout,
		retryCount:    retryCount,
		warmupPath:    "/models",
		heartbeatStop: make(chan struct{}),
	}
}

// newPooledHTTPClient 创建带连接池的 http.Client，复用到同一上游的 TLS/TCP 连接，
// 降低大 prompt 首次请求的 TTFT。
func newPooledHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			ForceAttemptHTTP2:     true,
			MaxIdleConns:          200,
			MaxIdleConnsPerHost:   100,
			MaxConnsPerHost:       200,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
		},
	}
}

// SetWarmupPath 设置预热时访问的 endpoint，默认为 "/models"。
func (f *Forwarder) SetWarmupPath(path string) {
	f.warmupPath = path
}

// StartHeartbeat 启动后台心跳：每隔 interval 对每个 provider key 探测一次可通性，
// 根据结果更新 key 的 State/LastSeen，让 dashboard 显示真实的在线状态。
func (f *Forwarder) StartHeartbeat(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		return
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-f.heartbeatStop:
				return
			case <-ticker.C:
				f.runHeartbeat(ctx)
			}
		}
	}()
}

func (f *Forwarder) runHeartbeat(ctx context.Context) {
	endpoints := f.router.ProviderEndpoints()
	var wg sync.WaitGroup
	for _, ep := range endpoints {
		if ep.BaseURL == "" {
			continue
		}
		for _, key := range ep.Keys {
			if key.Key == "" {
				continue
			}
			wg.Add(1)
			go func(ep router.ProviderEndpoint, key config.ProviderKey) {
				defer wg.Done()
				f.heartbeatKey(ctx, ep, key)
			}(ep, key)
		}
	}
	wg.Wait()
}

func (f *Forwarder) heartbeatKey(ctx context.Context, ep router.ProviderEndpoint, key config.ProviderKey) {
	url := strings.TrimRight(ep.BaseURL, "/") + f.warmupPath
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, url, nil)
	if err != nil {
		f.router.MarkKeyResult(ep.Provider, key.ID, http.StatusServiceUnavailable)
		return
	}

	for k, v := range ep.Headers {
		if k != "" && v != "" {
			req.Header.Set(k, v)
		}
	}
	req.Header.Set("Authorization", "Bearer "+key.Key)

	resp, err := f.client.Do(req)
	if err != nil {
		if f.logger != nil {
			f.logger.Error("", "forwarder", "heartbeat_failed", err, map[string]interface{}{
				"provider": ep.Provider,
				"key_id":   key.ID,
				"url":      url,
			})
		}
		f.router.MarkKeyResult(ep.Provider, key.ID, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	status := resp.StatusCode
	if status == http.StatusUnauthorized || status == http.StatusForbidden {
		// key 本身有问题，永久禁用，需要人工处理
		f.router.MarkKeyResult(ep.Provider, key.ID, status)
	} else if status >= http.StatusInternalServerError {
		f.router.MarkKeyResult(ep.Provider, key.ID, status)
	} else {
		// 2xx/3xx/429 都说明网络可通；429 表示限流，不表示 key 挂了
		f.router.MarkKeyResult(ep.Provider, key.ID, http.StatusOK)
	}

	if f.logger != nil {
		f.logger.Debug("", "forwarder", "heartbeat_ok", map[string]interface{}{
			"provider": ep.Provider,
			"key_id":   key.ID,
			"url":      url,
			"status":   status,
		})
	}
}

// SetLogger attaches a logger for DEBUG output.
func (f *Forwarder) SetLogger(l Logger) {
	f.logger = l
}

// Warmup 预先与所有配置的上游 Provider 建立 TCP/TLS 连接，
// 把连接放进 http.Transport 的 idle pool，避免第一个真实请求触发 TLS 握手。
// 预热失败只记录日志，不会阻塞网关启动。
func (f *Forwarder) Warmup(ctx context.Context) {
	endpoints := f.router.ProviderEndpoints()
	if len(endpoints) == 0 {
		return
	}

	var wg sync.WaitGroup
	for _, ep := range endpoints {
		wg.Add(1)
		go func(ep router.ProviderEndpoint) {
			defer wg.Done()
			f.warmupEndpoint(ctx, ep)
		}(ep)
	}
	wg.Wait()
}

func (f *Forwarder) warmupEndpoint(ctx context.Context, ep router.ProviderEndpoint) {
	if ep.BaseURL == "" {
		return
	}

	url := strings.TrimRight(ep.BaseURL, "/") + f.warmupPath
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		f.logWarmup(ep.Provider, url, err)
		return
	}

	for k, v := range ep.Headers {
		if k != "" && v != "" {
			req.Header.Set(k, v)
		}
	}
	// 部分 Provider 对 /models 也要求鉴权；使用第一个 key 避免 401 导致连接被关闭。
	if len(ep.Keys) > 0 && ep.Keys[0].Key != "" {
		req.Header.Set("Authorization", "Bearer "+ep.Keys[0].Key)
	}

	resp, err := f.client.Do(req)
	if err != nil {
		f.logWarmup(ep.Provider, url, err)
		return
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	if f.logger != nil {
		f.logger.Debug("", "forwarder", "warmup_ok", map[string]interface{}{
			"provider": ep.Provider,
			"url":      url,
			"status":   resp.StatusCode,
		})
	}
}

func (f *Forwarder) logWarmup(provider, url string, err error) {
	if f.logger == nil {
		return
	}
	f.logger.Debug("", "forwarder", "warmup_failed", map[string]interface{}{
		"provider": provider,
		"url":      url,
		"error":    err.Error(),
	})
}

// ForwardOpenAIRequest 转发 OpenAI 格式的请求到上游 Provider。
func (f *Forwarder) ForwardOpenAIRequest(ctx context.Context, req *models.OpenAIChatCompletionRequest) (*http.Response, error) {
	result, err := f.forwardWithRetry(ctx, req)
	if err != nil {
		return nil, err
	}
	return result.Response, nil
}

// ForwardOpenAIRequestWithInfo forwards a request and returns routing metadata.
func (f *Forwarder) ForwardOpenAIRequestWithInfo(ctx context.Context, req *models.OpenAIChatCompletionRequest) (*ForwardResponse, error) {
	return f.forwardWithRetry(ctx, req)
}

// ForwardOpenAIRequestRaw 转发原始请求体到上游 Provider。
func (f *Forwarder) ForwardOpenAIRequestRaw(ctx context.Context, body []byte, model string) (*http.Response, error) {
	return f.forwardRawWithRetry(ctx, body, model)
}

func (f *Forwarder) forwardWithRetry(ctx context.Context, req *models.OpenAIChatCompletionRequest) (*ForwardResponse, error) {
	var lastErr error
	var retryableResp *http.Response
	var retryableInfo ResponseInfo
	routeOpts := router.RouteOptions{RequireImages: req.HasImageContent()}
	for attempt := 0; attempt <= f.retryCount; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt) * time.Second
			time.Sleep(backoff)
		}

		route, err := f.router.RouteWithOptions(ctx, req.Model, routeOpts)
		if err != nil {
			lastErr = err
			continue
		}

		// 映射模型名，使用局部副本避免污染入参
		mappedModel := route.ModelMap[req.Model]
		if mappedModel == "" {
			mappedModel = req.Model
		}
		info := ResponseInfo{
			Provider:       route.Provider,
			ProviderKeyID:  route.Key.ID,
			UpstreamModel:  mappedModel,
			RouterDecision: fmt.Sprintf("%s/%s: %s -> %s", route.Provider, route.Key.ID, req.Model, mappedModel),
			FallbackCount:  attempt,
		}
		reqCopy := *req
		reqCopy.Messages = append([]models.OpenAIMessage(nil), req.Messages...)
		reqCopy.Model = mappedModel
		if err := f.normalizeImageURLs(ctx, &reqCopy); err != nil {
			return nil, err
		}

		body, err := json.Marshal(&reqCopy)
		if err != nil {
			return nil, fmt.Errorf("marshal request: %w", err)
		}

		if f.logger != nil {
			f.logger.Debug(traceIDFromContext(ctx), "forwarder", "upstream_request", map[string]interface{}{
				"provider":       route.Provider,
				"key_id":         route.Key.ID,
				"upstream_model": mappedModel,
				"attempt":        attempt,
				"body":           truncateForLog(string(body), 16000),
			})
		}

		resp, err := f.doRequest(ctx, route.BaseURL+"/chat/completions", body, route.Key.Key, route.Headers)
		if err != nil {
			lastErr = err
			continue
		}

		// 某些 Provider（如 Kimi）会在 HTTP 200 响应体里返回 JSON 错误，
		// 需要 peek 出来按真实错误码处理，才能正确重试并禁用失效 key。
		effectiveStatus, errBody := effectiveStatusFromResponse(resp)

		// 更新 Key 健康状态（使用识别后的真实状态码）
		f.router.MarkKeyResult(route.Provider, route.Key.ID, effectiveStatus)

		if shouldRetryUpstreamStatus(effectiveStatus) {
			if retryableResp != nil && retryableResp != resp {
				_ = retryableResp.Body.Close()
			}
			retryableResp = resp
			retryableInfo = info
			if len(errBody) > 0 {
				lastErr = fmt.Errorf("provider %s returned %d (body: %s)", route.Provider, effectiveStatus, truncateForLog(string(errBody), 512))
			} else {
				lastErr = fmt.Errorf("provider %s returned %d", route.Provider, effectiveStatus)
			}
			continue
		}

		if retryableResp != nil && retryableResp != resp {
			_ = retryableResp.Body.Close()
		}
		return &ForwardResponse{Response: resp, Info: info}, nil
	}

	if retryableResp != nil {
		return &ForwardResponse{Response: retryableResp, Info: retryableInfo}, nil
	}
	return nil, fmt.Errorf("all retries exhausted: %w", lastErr)
}

const maxDownloadedImageBytes = 15 * 1024 * 1024

func (f *Forwarder) normalizeImageURLs(ctx context.Context, req *models.OpenAIChatCompletionRequest) error {
	for i := range req.Messages {
		if len(req.Messages[i].RawContent) == 0 {
			continue
		}
		normalized, changed, err := f.normalizeMessageImageURLs(ctx, req.Messages[i].RawContent)
		if err != nil {
			return err
		}
		if changed {
			req.Messages[i].RawContent = normalized
		}
	}
	return nil
}

func (f *Forwarder) normalizeMessageImageURLs(ctx context.Context, raw []byte) ([]byte, bool, error) {
	var parts []map[string]interface{}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw, false, nil
	}

	changed := false
	for _, part := range parts {
		if partType, _ := part["type"].(string); partType != "image_url" {
			continue
		}
		imageURL, _ := part["image_url"].(map[string]interface{})
		if imageURL == nil {
			continue
		}
		urlValue, _ := imageURL["url"].(string)
		if !isHTTPImageURL(urlValue) {
			continue
		}
		dataURL, err := f.downloadImageAsDataURL(ctx, urlValue)
		if err != nil {
			return nil, false, err
		}
		imageURL["url"] = dataURL
		changed = true
	}

	if !changed {
		return raw, false, nil
	}
	normalized, err := json.Marshal(parts)
	if err != nil {
		return nil, false, err
	}
	return normalized, true, nil
}

func isHTTPImageURL(value string) bool {
	lower := strings.ToLower(strings.TrimSpace(value))
	return strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "https://")
}

func (f *Forwarder) downloadImageAsDataURL(ctx context.Context, imageURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return "", err
	}
	resp, err := f.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("download image: upstream returned %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxDownloadedImageBytes+1))
	if err != nil {
		return "", err
	}
	if len(body) > maxDownloadedImageBytes {
		return "", fmt.Errorf("download image: image exceeds %d bytes", maxDownloadedImageBytes)
	}
	contentType := strings.TrimSpace(strings.Split(resp.Header.Get("Content-Type"), ";")[0])
	if contentType == "" || !strings.HasPrefix(strings.ToLower(contentType), "image/") {
		contentType = http.DetectContentType(body)
	}
	if !strings.HasPrefix(strings.ToLower(contentType), "image/") {
		return "", fmt.Errorf("download image: unsupported content type %s", contentType)
	}
	return "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(body), nil
}

func (f *Forwarder) forwardRawWithRetry(ctx context.Context, body []byte, model string) (*http.Response, error) {
	var lastErr error
	var retryableResp *http.Response
	for attempt := 0; attempt <= f.retryCount; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt) * time.Second
			time.Sleep(backoff)
		}

		route, err := f.router.Route(ctx, model)
		if err != nil {
			lastErr = err
			continue
		}

		// 替换请求体中的模型名
		var reqMap map[string]interface{}
		if err := json.Unmarshal(body, &reqMap); err == nil {
			if mappedModel := route.ModelMap[model]; mappedModel != "" {
				reqMap["model"] = mappedModel
			}
			body, _ = json.Marshal(reqMap)
		}

		resp, err := f.doRequest(ctx, route.BaseURL+"/chat/completions", body, route.Key.Key, route.Headers)
		if err != nil {
			lastErr = err
			continue
		}

		f.router.MarkKeyResult(route.Provider, route.Key.ID, resp.StatusCode)

		if shouldRetryUpstreamStatus(resp.StatusCode) {
			if retryableResp != nil && retryableResp != resp {
				_ = retryableResp.Body.Close()
			}
			retryableResp = resp
			lastErr = fmt.Errorf("provider %s returned %d", route.Provider, resp.StatusCode)
			continue
		}

		if retryableResp != nil && retryableResp != resp {
			_ = retryableResp.Body.Close()
		}
		return resp, nil
	}

	if retryableResp != nil {
		return retryableResp, nil
	}
	return nil, fmt.Errorf("all retries exhausted: %w", lastErr)
}

func shouldRetryUpstreamStatus(statusCode int) bool {
	return statusCode == http.StatusUnauthorized ||
		statusCode == http.StatusTooManyRequests ||
		statusCode == http.StatusPaymentRequired ||
		statusCode >= http.StatusInternalServerError
}

// effectiveStatusFromResponse peeks at a response body that some providers return
// with HTTP 200 but contains a JSON error envelope. It restores the body so the
// response can still be read downstream, and returns the effective HTTP status
// code that should be used for retries and key health tracking.
func effectiveStatusFromResponse(resp *http.Response) (int, []byte) {
	if resp == nil || resp.Body == nil {
		return http.StatusServiceUnavailable, nil
	}

	// Only 2xx/3xx bodies may hide JSON errors; real 4xx/5xx keep their status.
	if resp.StatusCode >= http.StatusBadRequest {
		return resp.StatusCode, nil
	}

	buf := make([]byte, 4096)
	n, err := resp.Body.Read(buf)
	var readErr error
	var rest bytes.Buffer
	if n > 0 {
		rest.Write(buf[:n])
		if err == nil {
			_, readErr = io.Copy(&rest, resp.Body)
		}
	} else if err != nil && err != io.EOF {
		readErr = err
	}

	body := rest.Bytes()
	trimmed := strings.TrimSpace(string(body))
	if !strings.HasPrefix(trimmed, "{\"error\"") && !strings.HasPrefix(trimmed, "{\"type\"") {
		resp.Body = io.NopCloser(io.MultiReader(bytes.NewReader(body), resp.Body))
		if readErr != nil {
			return http.StatusServiceUnavailable, body
		}
		return resp.StatusCode, nil
	}

	var env struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
		Type string `json:"type"`
	}
	_ = json.Unmarshal(body, &env)

	errType := strings.ToLower(env.Error.Type)
	if errType == "" {
		errType = strings.ToLower(env.Type)
	}
	msg := strings.ToLower(env.Error.Message)

	switch {
	case errType == "invalid_authentication_error" || strings.Contains(msg, "api key") && strings.Contains(msg, "invalid"):
		resp.StatusCode = http.StatusUnauthorized
	case errType == "rate_limit_error" || strings.Contains(msg, "rate limit") || strings.Contains(msg, "too many requests"):
		resp.StatusCode = http.StatusTooManyRequests
	case errType == "invalid_request_error" || strings.Contains(msg, "bad request"):
		resp.StatusCode = http.StatusBadRequest
	case errType == "not_found_error" || strings.Contains(msg, "not found"):
		resp.StatusCode = http.StatusNotFound
	default:
		// Unknown JSON error in a 200 body: treat as retryable upstream error.
		resp.StatusCode = http.StatusServiceUnavailable
	}

	resp.Body = io.NopCloser(bytes.NewReader(body))
	return resp.StatusCode, body
}

func truncateForLog(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "...[truncated]"
}

func (f *Forwarder) doRequest(ctx context.Context, url string, body []byte, apiKey string, headers map[string]string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	for key, value := range headers {
		if key != "" && value != "" {
			req.Header.Set(key, value)
		}
	}

	return f.client.Do(req)
}

// StreamResponse 将上游 SSE 流式响应写入 http.ResponseWriter。
func StreamResponse(src io.ReadCloser, dst http.ResponseWriter) error {
	defer src.Close()

	flusher, ok := dst.(http.Flusher)
	if !ok {
		return fmt.Errorf("response writer does not support flushing")
	}

	buf := make([]byte, 4096)
	for {
		n, err := src.Read(buf)
		if n > 0 {
			_, writeErr := dst.Write(buf[:n])
			if writeErr != nil {
				return writeErr
			}
			flusher.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// Close 关闭 Forwarder 的资源。
func (f *Forwarder) Close() {
	close(f.heartbeatStop)
	f.client.CloseIdleConnections()
}
