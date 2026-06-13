// Package models 定义 Anthropic Messages API 的请求和响应结构体。
package models

import (
	"encoding/json"
	"strings"
)

// AnthropicMessageRequest 对应 Anthropic /v1/messages 请求体。
type AnthropicMessageRequest struct {
	Model       string             `json:"model"`
	Messages    []AnthropicMessage `json:"messages"`
	System      string             `json:"system,omitempty"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	Tools       []AnthropicTool    `json:"tools,omitempty"`
	ToolChoice  interface{}        `json:"tool_choice,omitempty"`
	Thinking    interface{}        `json:"thinking,omitempty"`
}

// UnmarshalJSON accepts both Anthropic system formats:
// "system": "..." and "system": [{"type":"text","text":"..."}].
func (r *AnthropicMessageRequest) UnmarshalJSON(data []byte) error {
	type alias AnthropicMessageRequest
	aux := struct {
		System json.RawMessage `json:"system"`
		*alias
	}{
		alias: (*alias)(r),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if len(aux.System) == 0 || string(aux.System) == "null" {
		return nil
	}

	system, err := parseAnthropicSystem(aux.System)
	if err != nil {
		return err
	}
	r.System = system
	return nil
}

func parseAnthropicSystem(raw json.RawMessage) (string, error) {
	var system string
	if err := json.Unmarshal(raw, &system); err == nil {
		return system, nil
	}

	var blocks []interface{}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", err
	}

	var parts []string
	for _, block := range blocks {
		switch b := block.(type) {
		case string:
			if b != "" {
				parts = append(parts, b)
			}
		case map[string]interface{}:
			if text, ok := b["text"].(string); ok && text != "" {
				parts = append(parts, text)
			}
		}
	}
	return strings.Join(parts, "\n\n"), nil
}

// AnthropicMessage 是 Anthropic 消息格式。
// Content 可以是字符串或 [{"type":"text","text":"..."}] 数组。
type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

// AnthropicMessageResponse 对应 Anthropic /v1/messages 非流式响应。
type AnthropicMessageResponse struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Model        string             `json:"model"`
	Content      []AnthropicContent `json:"content"`
	StopReason   string             `json:"stop_reason"`
	StopSequence *string            `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage     `json:"usage"`
}

// AnthropicTool 定义 Anthropic 工具声明。
type AnthropicTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema,omitempty"`
}

// AnthropicContent 是响应中的内容块。
type AnthropicContent struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	ID        string      `json:"id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   interface{} `json:"content,omitempty"`
	Thinking  string      `json:"thinking,omitempty"`
	Signature string      `json:"signature,omitempty"`
}

// AnthropicUsage 记录 token 消耗。
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// AnthropicStreamChunk 是 SSE 流式响应中的单个数据块。
type AnthropicStreamChunk struct {
	Type         string                  `json:"type"`
	Index        int                     `json:"index,omitempty"`
	Delta        *AnthropicStreamDelta   `json:"delta,omitempty"`
	Usage        *AnthropicUsage         `json:"usage,omitempty"`
	Message      *AnthropicStreamMessage `json:"message,omitempty"`
	ContentBlock *AnthropicContent       `json:"content_block,omitempty"`
}

// AnthropicStreamDelta 是流式增量内容。
type AnthropicStreamDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
}

// AnthropicStreamMessage 是流式消息开始事件。
type AnthropicStreamMessage struct {
	ID    string `json:"id"`
	Type  string `json:"type"`
	Role  string `json:"role"`
	Model string `json:"model"`
}
