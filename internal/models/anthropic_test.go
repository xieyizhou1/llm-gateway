package models

import (
	"encoding/json"
	"testing"
)

func TestAnthropicMessageRequestUnmarshalSystemString(t *testing.T) {
	input := []byte(`{
		"model": "claude-sonnet-4-6",
		"system": "Be concise",
		"max_tokens": 128,
		"messages": [{"role": "user", "content": "Hello"}]
	}`)

	var req AnthropicMessageRequest
	if err := json.Unmarshal(input, &req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.System != "Be concise" {
		t.Fatalf("expected system string, got %q", req.System)
	}
}

func TestAnthropicMessageRequestUnmarshalSystemBlocks(t *testing.T) {
	input := []byte(`{
		"model": "claude-sonnet-4-6",
		"system": [
			{"type": "text", "text": "First system block"},
			{"type": "text", "text": "Second system block"}
		],
		"max_tokens": 128,
		"messages": [{
			"role": "user",
			"content": [{"type": "text", "text": "Hello"}]
		}],
		"output_config": {
			"format": {"type": "json_schema", "schema": {"type": "object"}}
		}
	}`)

	var req AnthropicMessageRequest
	if err := json.Unmarshal(input, &req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	expected := "First system block\n\nSecond system block"
	if req.System != expected {
		t.Fatalf("expected %q, got %q", expected, req.System)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("expected one message, got %d", len(req.Messages))
	}
}
