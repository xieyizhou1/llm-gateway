// Package models 定义 OpenAI Chat Completions API 的请求和响应结构体。
package models

import (
	"bytes"
	"encoding/json"
	"strings"
)

// OpenAIChatCompletionRequest 对应 OpenAI /v1/chat/completions 请求体。
type OpenAIChatCompletionRequest struct {
	Model         string                 `json:"model"`
	Messages      []OpenAIMessage        `json:"messages"`
	MaxTokens     int                    `json:"max_tokens,omitempty"`
	Temperature   float64                `json:"temperature,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	StreamOptions map[string]interface{} `json:"stream_options,omitempty"`
	Tools         []OpenAITool           `json:"tools,omitempty"`
	ToolChoice    interface{}            `json:"tool_choice,omitempty"`
}

// OpenAIMessage 是 OpenAI 消息格式。
type OpenAIMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	RawContent       []byte     `json:"-"`
	ReasoningContent string     `json:"reasoning_content"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
}

// UnmarshalJSON accepts both classic text content and OpenAI multi-modal
// content parts. Text is extracted into Content for existing gateway logic,
// while RawContent preserves the original JSON value for upstream forwarding.
func (m *OpenAIMessage) UnmarshalJSON(data []byte) error {
	type alias OpenAIMessage
	var raw struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		ReasoningContent string          `json:"reasoning_content"`
		ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID       string          `json:"tool_call_id,omitempty"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*m = OpenAIMessage{
		Role:             raw.Role,
		ReasoningContent: raw.ReasoningContent,
		ToolCalls:        raw.ToolCalls,
		ToolCallID:       raw.ToolCallID,
	}
	if len(raw.Content) == 0 || bytes.Equal(raw.Content, []byte("null")) {
		return nil
	}
	m.RawContent = append(m.RawContent[:0], raw.Content...)

	var text string
	if err := json.Unmarshal(raw.Content, &text); err == nil {
		m.Content = text
		m.RawContent = nil
		return nil
	}

	m.Content = textFromContentParts(raw.Content)
	_ = alias{}
	return nil
}

// MarshalJSON preserves multi-modal content arrays when they were received.
func (m OpenAIMessage) MarshalJSON() ([]byte, error) {
	type alias OpenAIMessage
	raw := struct {
		Role             string          `json:"role"`
		Content          json.RawMessage `json:"content"`
		ReasoningContent string          `json:"reasoning_content,omitempty"`
		ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
		ToolCallID       string          `json:"tool_call_id,omitempty"`
	}{
		Role:             m.Role,
		ReasoningContent: m.ReasoningContent,
		ToolCalls:        m.ToolCalls,
		ToolCallID:       m.ToolCallID,
	}
	if len(m.RawContent) > 0 && json.Valid(m.RawContent) {
		raw.Content = append(raw.Content[:0], m.RawContent...)
	} else {
		content, err := json.Marshal(m.Content)
		if err != nil {
			return nil, err
		}
		raw.Content = content
	}
	_ = alias{}
	return json.Marshal(raw)
}

func textFromContentParts(raw []byte) string {
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" && strings.TrimSpace(part.Text) != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// HasImageContent reports whether the message contains OpenAI-compatible image
// parts. It is used by the router to avoid sending vision requests to text-only
// providers.
func (m OpenAIMessage) HasImageContent() bool {
	if len(m.RawContent) == 0 {
		return false
	}
	var parts []struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(m.RawContent, &parts); err != nil {
		return false
	}
	for _, part := range parts {
		if part.Type == "image_url" || part.Type == "input_image" {
			return true
		}
	}
	return false
}

// HasImageContent reports whether any chat message contains image parts.
func (r OpenAIChatCompletionRequest) HasImageContent() bool {
	for _, msg := range r.Messages {
		if msg.HasImageContent() {
			return true
		}
	}
	return false
}

// ToolCall 定义工具调用结构。
type ToolCall struct {
	Index    int          `json:"index,omitempty"`
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall 定义函数调用详情。
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// OpenAITool 定义工具声明。
type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

// OpenAIFunction 定义函数声明。
type OpenAIFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// OpenAIChatCompletionResponse 对应 OpenAI /v1/chat/completions 非流式响应。
type OpenAIChatCompletionResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

// OpenAIChoice 是响应中的单个选择。
type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIUsage 记录 token 消耗。
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// OpenAIStreamChunk 是 SSE 流式响应中的单个数据块。
type OpenAIStreamChunk struct {
	ID      string               `json:"id"`
	Object  string               `json:"object"`
	Created int64                `json:"created"`
	Model   string               `json:"model"`
	Choices []OpenAIStreamChoice `json:"choices"`
	Usage   *OpenAIUsage         `json:"usage,omitempty"`
}

// OpenAIStreamChoice 是流式响应中的单个选择。
type OpenAIStreamChoice struct {
	Index        int           `json:"index"`
	Delta        OpenAIMessage `json:"delta"`
	FinishReason *string       `json:"finish_reason"`
}
