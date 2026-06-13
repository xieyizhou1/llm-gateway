// Package provider contains upstream provider adapter contracts.
package provider

import (
	"bytes"
	"context"
	"net/http"
	"strings"
)

type Adapter interface {
	Name() string
	ChatCompletionsURL(baseURL string) string
	NewChatCompletionsRequest(ctx context.Context, url string, body []byte, apiKey string) (*http.Request, error)
}

type OpenAICompatibleAdapter struct{}

func (OpenAICompatibleAdapter) Name() string { return "openai_compatible" }

func (OpenAICompatibleAdapter) ChatCompletionsURL(baseURL string) string {
	return strings.TrimRight(baseURL, "/") + "/chat/completions"
}

func (OpenAICompatibleAdapter) NewChatCompletionsRequest(ctx context.Context, url string, body []byte, apiKey string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	return req, nil
}
