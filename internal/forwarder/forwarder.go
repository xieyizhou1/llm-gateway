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
	"net/http"
	"strings"
	"time"

	"llm-gateway/internal/models"
	"llm-gateway/internal/router"
)

// Forwarder 负责将请求转发到上游 Provider。
type Forwarder struct {
	client     *http.Client
	router     *router.Router
	timeout    time.Duration
	retryCount int
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
		client: &http.Client{
			Timeout: timeout,
		},
		router:     r,
		timeout:    timeout,
		retryCount: retryCount,
	}
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

		resp, err := f.doRequest(ctx, route.BaseURL+"/chat/completions", body, route.Key.Key, route.Headers)
		if err != nil {
			lastErr = err
			continue
		}

		// 更新 Key 健康状态
		f.router.MarkKeyResult(route.Provider, route.Key.ID, resp.StatusCode)

		if shouldRetryUpstreamStatus(resp.StatusCode) {
			if retryableResp != nil && retryableResp != resp {
				_ = retryableResp.Body.Close()
			}
			retryableResp = resp
			retryableInfo = info
			lastErr = fmt.Errorf("provider %s returned %d", route.Provider, resp.StatusCode)
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
	return statusCode == http.StatusUnauthorized || statusCode == http.StatusTooManyRequests || statusCode >= http.StatusInternalServerError
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
	f.client.CloseIdleConnections()
}
