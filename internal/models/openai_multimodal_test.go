package models

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestOpenAIMessagePreservesMultimodalContent(t *testing.T) {
	body := []byte(`{
		"role": "user",
		"content": [
			{"type": "text", "text": "What is in this image?"},
			{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
		]
	}`)

	var msg OpenAIMessage
	if err := json.Unmarshal(body, &msg); err != nil {
		t.Fatalf("unmarshal multimodal message: %v", err)
	}
	if msg.Content != "What is in this image?" {
		t.Fatalf("expected text extraction, got %q", msg.Content)
	}
	if len(msg.RawContent) == 0 {
		t.Fatal("expected raw content to be preserved")
	}

	out, err := json.Marshal(msg)
	if err != nil {
		t.Fatalf("marshal multimodal message: %v", err)
	}
	got := string(out)
	if !strings.Contains(got, `"image_url"`) {
		t.Fatalf("expected image_url content to be preserved, got %s", got)
	}
	if strings.Contains(got, `"content":"What is in this image?"`) {
		t.Fatalf("expected content array, got string content: %s", got)
	}
}
