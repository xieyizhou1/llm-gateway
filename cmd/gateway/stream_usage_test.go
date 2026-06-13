package main

import (
	"testing"

	"llm-gateway/internal/models"
	"llm-gateway/internal/usage"
)

func TestEnsureStreamUsageAddsIncludeUsage(t *testing.T) {
	req := &models.OpenAIChatCompletionRequest{Model: "gpt-5.5", Stream: true}
	ensureStreamUsage(req)
	if req.StreamOptions == nil {
		t.Fatal("expected stream_options to be set")
	}
	if got := req.StreamOptions["include_usage"]; got != true {
		t.Fatalf("expected include_usage true, got %v", got)
	}
}

func TestCaptureOpenAIStreamUsageUpdatesRecord(t *testing.T) {
	record := &usage.RequestLog{}
	captureOpenAIStreamUsage([]byte(`data: {"id":"cmpl","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}

data: [DONE]

`), record)
	if record.PromptTokens != 12 || record.OutputTokens != 5 || record.TotalTokens != 17 {
		t.Fatalf("unexpected record usage: %+v", record)
	}
}
