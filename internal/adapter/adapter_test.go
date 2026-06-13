package adapter

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"llm-gateway/internal/models"
)

func TestAnthropicToOpenAI(t *testing.T) {
	anthropicReq := &models.AnthropicMessageRequest{
		Model: "claude-sonnet",
		Messages: []models.AnthropicMessage{
			{Role: "user", Content: "Hello"},
		},
		System:      "You are a helpful assistant",
		MaxTokens:   100,
		Temperature: 0.7,
		Stream:      true,
	}

	openAIReq := AnthropicToOpenAI(anthropicReq)

	if openAIReq.Model != "claude-sonnet" {
		t.Errorf("expected model claude-sonnet, got %s", openAIReq.Model)
	}
	if len(openAIReq.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].Role != "system" {
		t.Errorf("expected first message role system, got %s", openAIReq.Messages[0].Role)
	}
	if openAIReq.Messages[0].Content != "You are a helpful assistant" {
		t.Errorf("expected system content, got %s", openAIReq.Messages[0].Content)
	}
	if openAIReq.Messages[1].Role != "user" {
		t.Errorf("expected second message role user, got %s", openAIReq.Messages[1].Role)
	}
	if openAIReq.MaxTokens != 100 {
		t.Errorf("expected max_tokens 100, got %d", openAIReq.MaxTokens)
	}
	if openAIReq.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %f", openAIReq.Temperature)
	}
	if !openAIReq.Stream {
		t.Error("expected stream true")
	}
}

func TestAnthropicToOpenAIWithToolsAndToolBlocks(t *testing.T) {
	anthropicReq := &models.AnthropicMessageRequest{
		Model: "claude-sonnet",
		Tools: []models.AnthropicTool{
			{
				Name:        "shell_command",
				Description: "Run a shell command",
				InputSchema: map[string]interface{}{
					"type": "object",
				},
			},
		},
		ToolChoice: map[string]interface{}{
			"type": "tool",
			"name": "shell_command",
		},
		Messages: []models.AnthropicMessage{
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "text", "text": "List processes"},
				},
			},
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type": "tool_use",
						"id":   "toolu_123",
						"name": "shell_command",
						"input": map[string]interface{}{
							"command": "ps",
						},
					},
				},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "toolu_123",
						"content":     "PID 1",
					},
				},
			},
		},
	}

	openAIReq := AnthropicToOpenAI(anthropicReq)
	if len(openAIReq.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(openAIReq.Tools))
	}
	if openAIReq.Tools[0].Function.Name != "shell_command" {
		t.Fatalf("expected shell_command tool, got %s", openAIReq.Tools[0].Function.Name)
	}
	toolChoice, ok := openAIReq.ToolChoice.(map[string]interface{})
	if !ok {
		t.Fatalf("expected tool_choice map, got %T", openAIReq.ToolChoice)
	}
	if toolChoice["name"] != "shell_command" {
		t.Fatalf("expected tool_choice name shell_command, got %v", toolChoice["name"])
	}
	if len(openAIReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].Role != "user" || openAIReq.Messages[0].Content != "List processes" {
		t.Fatalf("unexpected user message: %+v", openAIReq.Messages[0])
	}
	if openAIReq.Messages[1].Role != "assistant" {
		t.Fatalf("expected assistant tool call message, got %+v", openAIReq.Messages[1])
	}
	if strings.TrimSpace(openAIReq.Messages[1].ReasoningContent) == "" {
		t.Fatalf("expected assistant tool call reasoning_content to be preserved")
	}
	if len(openAIReq.Messages[1].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(openAIReq.Messages[1].ToolCalls))
	}
	if openAIReq.Messages[1].ToolCalls[0].ID != "toolu_123" {
		t.Fatalf("expected tool call id toolu_123, got %s", openAIReq.Messages[1].ToolCalls[0].ID)
	}
	if openAIReq.Messages[1].ToolCalls[0].Function.Arguments != "{\"command\":\"ps\"}" {
		t.Fatalf("unexpected tool arguments: %s", openAIReq.Messages[1].ToolCalls[0].Function.Arguments)
	}
	if openAIReq.Messages[2].Role != "tool" || openAIReq.Messages[2].ToolCallID != "toolu_123" {
		t.Fatalf("unexpected tool result message: %+v", openAIReq.Messages[2])
	}
	if openAIReq.Messages[2].Content != "PID 1" {
		t.Fatalf("unexpected tool result content: %s", openAIReq.Messages[2].Content)
	}
}

func TestOpenAIToAnthropic(t *testing.T) {
	openAIResp := &models.OpenAIChatCompletionResponse{
		ID:      "chatcmpl-123",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "moonshot-v1-8k",
		Choices: []models.OpenAIChoice{
			{
				Index: 0,
				Message: models.OpenAIMessage{
					Role:    "assistant",
					Content: "Hello! How can I help you?",
				},
				FinishReason: "stop",
			},
		},
		Usage: models.OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	anthropicResp := OpenAIToAnthropic(openAIResp)

	if anthropicResp.ID != "chatcmpl-123" {
		t.Errorf("expected id chatcmpl-123, got %s", anthropicResp.ID)
	}
	if anthropicResp.Role != "assistant" {
		t.Errorf("expected role assistant, got %s", anthropicResp.Role)
	}
	if len(anthropicResp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(anthropicResp.Content))
	}
	if anthropicResp.Content[0].Text != "Hello! How can I help you?" {
		t.Errorf("unexpected content: %s", anthropicResp.Content[0].Text)
	}
	if anthropicResp.StopReason != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", anthropicResp.StopReason)
	}
	if anthropicResp.Usage.InputTokens != 10 {
		t.Errorf("expected input_tokens 10, got %d", anthropicResp.Usage.InputTokens)
	}
	if anthropicResp.Usage.OutputTokens != 20 {
		t.Errorf("expected output_tokens 20, got %d", anthropicResp.Usage.OutputTokens)
	}
}

func TestOpenAIToAnthropicWithToolCalls(t *testing.T) {
	openAIResp := &models.OpenAIChatCompletionResponse{
		ID:      "chatcmpl-tool",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "moonshot-v1-8k",
		Choices: []models.OpenAIChoice{
			{
				Index: 0,
				Message: models.OpenAIMessage{
					Role: "assistant",
					ToolCalls: []models.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: models.FunctionCall{
								Name:      "shell_command",
								Arguments: "{\"command\":\"pwd\"}",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
		Usage: models.OpenAIUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}

	anthropicResp := OpenAIToAnthropic(openAIResp)
	if anthropicResp.StopReason != "tool_use" {
		t.Fatalf("expected stop_reason tool_use, got %s", anthropicResp.StopReason)
	}
	if len(anthropicResp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(anthropicResp.Content))
	}
	if anthropicResp.Content[0].Type != "tool_use" {
		t.Fatalf("expected tool_use block, got %s", anthropicResp.Content[0].Type)
	}
	if anthropicResp.Content[0].ID != "call_123" {
		t.Fatalf("expected call id call_123, got %s", anthropicResp.Content[0].ID)
	}
	if anthropicResp.Content[0].Name != "shell_command" {
		t.Fatalf("expected tool name shell_command, got %s", anthropicResp.Content[0].Name)
	}
	input, ok := anthropicResp.Content[0].Input.(map[string]interface{})
	if !ok {
		t.Fatalf("expected input object, got %T", anthropicResp.Content[0].Input)
	}
	if input["command"] != "pwd" {
		t.Fatalf("expected command pwd, got %v", input["command"])
	}
}

func TestOpenAIToAnthropicRepairsMalformedToolArguments(t *testing.T) {
	openAIResp := &models.OpenAIChatCompletionResponse{
		ID:      "chatcmpl-tool",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "moonshot-v1-8k",
		Choices: []models.OpenAIChoice{
			{
				Index: 0,
				Message: models.OpenAIMessage{
					Role: "assistant",
					ToolCalls: []models.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: models.FunctionCall{
								Name:      "PowerShell",
								Arguments: "{\"command\":\"Get-Process | Select-Object Name, Id\",\"description\":\"truncated",
							},
						},
					},
				},
				FinishReason: "tool_calls",
			},
		},
	}

	anthropicResp := OpenAIToAnthropic(openAIResp)
	if len(anthropicResp.Content) != 1 {
		t.Fatalf("expected 1 content block, got %d", len(anthropicResp.Content))
	}
	input, ok := anthropicResp.Content[0].Input.(map[string]interface{})
	if !ok {
		t.Fatalf("expected repaired input object, got %T", anthropicResp.Content[0].Input)
	}
	if input["command"] != "Get-Process | Select-Object Name, Id" {
		t.Fatalf("unexpected repaired command: %v", input["command"])
	}
}

func TestMapModel(t *testing.T) {
	modelMap := map[string]string{
		"gpt-4":         "moonshot-v1-8k",
		"claude-sonnet": "moonshot-v1-32k",
	}

	tests := []struct {
		input    string
		expected string
	}{
		{"gpt-4", "moonshot-v1-8k"},
		{"claude-sonnet", "moonshot-v1-32k"},
		{"unknown-model", "unknown-model"},
	}

	for _, tt := range tests {
		result := MapModel(tt.input, modelMap)
		if result != tt.expected {
			t.Errorf("MapModel(%s) = %s, expected %s", tt.input, result, tt.expected)
		}
	}
}

func TestExtractSystemMessage(t *testing.T) {
	messages := []models.OpenAIMessage{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi"},
	}

	system, filtered := ExtractSystemMessage(messages)
	if system != "You are helpful" {
		t.Errorf("expected system message 'You are helpful', got %s", system)
	}
	if len(filtered) != 2 {
		t.Fatalf("expected 2 filtered messages, got %d", len(filtered))
	}
	if filtered[0].Role != "user" {
		t.Errorf("expected first filtered role user, got %s", filtered[0].Role)
	}
}

func TestOpenAIToAnthropicRequest(t *testing.T) {
	openAIReq := &models.OpenAIChatCompletionRequest{
		Model: "gpt-4",
		Messages: []models.OpenAIMessage{
			{Role: "system", Content: "Be concise"},
			{Role: "user", Content: "What is Go?"},
		},
		MaxTokens:   50,
		Temperature: 0.5,
		Stream:      false,
	}

	anthropicReq := OpenAIToAnthropicRequest(openAIReq)
	if anthropicReq.System != "Be concise" {
		t.Errorf("expected system 'Be concise', got %s", anthropicReq.System)
	}
	if len(anthropicReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(anthropicReq.Messages))
	}
	if anthropicReq.Messages[0].Role != "user" {
		t.Errorf("expected role user, got %s", anthropicReq.Messages[0].Role)
	}
}

func TestOpenAIStreamChunkToAnthropic(t *testing.T) {
	finishReason := "stop"
	chunk := &models.OpenAIStreamChunk{
		ID:      "chatcmpl-456",
		Object:  "chat.completion.chunk",
		Created: 1234567891,
		Model:   "moonshot-v1-8k",
		Choices: []models.OpenAIStreamChoice{
			{
				Index: 0,
				Delta: models.OpenAIMessage{
					Role:    "assistant",
					Content: "Hello",
				},
				FinishReason: &finishReason,
			},
		},
	}

	result := OpenAIStreamChunkToAnthropic(chunk)
	if len(result) != 2 {
		t.Fatalf("expected 2 chunks, got %d", len(result))
	}
	if result[0].Type != "content_block_delta" {
		t.Errorf("expected type content_block_delta, got %s", result[0].Type)
	}
	if result[0].Delta == nil || result[0].Delta.Text != "Hello" {
		t.Errorf("expected delta text 'Hello', got %v", result[0].Delta)
	}
	if result[1].Type != "message_stop" {
		t.Errorf("expected type message_stop, got %s", result[1].Type)
	}
}

func TestOpenAIStreamChunkToAnthropicToolCallsAndReasoning(t *testing.T) {
	finishReason := "tool_calls"
	chunk := &models.OpenAIStreamChunk{
		ID:      "chatcmpl-456",
		Object:  "chat.completion.chunk",
		Created: 1234567891,
		Model:   "moonshot-v1-8k",
		Choices: []models.OpenAIStreamChoice{
			{
				Index: 0,
				Delta: models.OpenAIMessage{
					ReasoningContent: "hidden reasoning",
					ToolCalls: []models.ToolCall{
						{
							Index: 0,
							ID:    "call_123",
							Type:  "function",
							Function: models.FunctionCall{
								Name:      "shell_command",
								Arguments: "{\"command\":\"pwd\"}",
							},
						},
					},
				},
				FinishReason: &finishReason,
			},
		},
	}

	result := OpenAIStreamChunkToAnthropic(chunk)
	if len(result) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(result))
	}
	if result[0].Delta == nil || result[0].Delta.Type != "thinking_delta" {
		t.Fatalf("expected thinking delta, got %+v", result[0])
	}
	if result[1].Delta == nil || result[1].Delta.Type != "input_json_delta" {
		t.Fatalf("expected input_json_delta, got %+v", result[1])
	}
	if result[1].Index != 0 {
		t.Fatalf("expected tool index 0, got %d", result[1].Index)
	}
	if result[2].Type != "message_stop" {
		t.Fatalf("expected message_stop, got %s", result[2].Type)
	}
}

func TestAnthropicStreamStart(t *testing.T) {
	chunk := AnthropicStreamStart("msg-123", "claude-sonnet")
	if chunk.Type != "message_start" {
		t.Errorf("expected type message_start, got %s", chunk.Type)
	}
	if chunk.Message == nil {
		t.Fatal("expected Message to be non-nil")
	}
	if chunk.Message.ID != "msg-123" {
		t.Errorf("expected message id msg-123, got %s", chunk.Message.ID)
	}
	if chunk.Message.Model != "claude-sonnet" {
		t.Errorf("expected model claude-sonnet, got %s", chunk.Message.Model)
	}
	if chunk.Message.Type != "message" {
		t.Errorf("expected type message, got %s", chunk.Message.Type)
	}
	if chunk.Message.Role != "assistant" {
		t.Errorf("expected role assistant, got %s", chunk.Message.Role)
	}
}

func TestTransformOpenAIStreamBody(t *testing.T) {
	body := io.NopCloser(strings.NewReader("hello world"))
	var buf bytes.Buffer
	err := TransformOpenAIStreamBody(body, &buf)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if buf.String() != "hello world" {
		t.Errorf("expected 'hello world', got %s", buf.String())
	}
}

func TestSerializeAnthropicStreamChunk(t *testing.T) {
	chunk := &models.AnthropicStreamChunk{
		Type:  "content_block_delta",
		Index: 0,
		Delta: &models.AnthropicStreamDelta{
			Type: "text_delta",
			Text: "Hello",
		},
	}
	result, err := SerializeAnthropicStreamChunk(chunk)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !strings.HasPrefix(result, "data: ") {
		t.Errorf("expected data: prefix, got %s", result)
	}
	if !strings.HasSuffix(result, "\n\n") {
		t.Errorf("expected \\n\\n suffix, got %s", result)
	}
}

func TestSerializeOpenAIStreamChunk(t *testing.T) {
	chunk := &models.OpenAIStreamChunk{
		ID:      "chatcmpl-789",
		Object:  "chat.completion.chunk",
		Created: 1234567892,
		Model:   "gpt-4",
		Choices: []models.OpenAIStreamChoice{
			{
				Index: 0,
				Delta: models.OpenAIMessage{Role: "assistant", Content: "World"},
			},
		},
	}
	result, err := SerializeOpenAIStreamChunk(chunk)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if !strings.HasPrefix(result, "data: ") {
		t.Errorf("expected data: prefix, got %s", result)
	}
	if !strings.HasSuffix(result, "\n\n") {
		t.Errorf("expected \\n\\n suffix, got %s", result)
	}
}

func TestParseOpenAIStreamChunk(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantNil  bool
		wantErr  bool
		wantText string
	}{
		{
			name:     "valid chunk",
			input:    `data: {"id":"cmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}`,
			wantNil:  false,
			wantErr:  false,
			wantText: "Hi",
		},
		{
			name:    "done signal",
			input:   "data: [DONE]",
			wantNil: true,
			wantErr: false,
		},
		{
			name:    "empty data",
			input:   "data: ",
			wantNil: true,
			wantErr: false,
		},
		{
			name:    "invalid json",
			input:   "data: {invalid",
			wantNil: true,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chunk, err := ParseOpenAIStreamChunk(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error")
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}
			if tt.wantNil {
				if chunk != nil {
					t.Errorf("expected nil chunk, got %+v", chunk)
				}
				return
			}
			if chunk == nil {
				t.Fatal("expected non-nil chunk")
			}
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != tt.wantText {
				t.Errorf("expected text %s, got %s", tt.wantText, chunk.Choices[0].Delta.Content)
			}
		})
	}
}

func TestCopyAndDecodeJSON(t *testing.T) {
	type testStruct struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}
	input := `{"name":"test","value":42}`
	var result testStruct
	err := CopyAndDecodeJSON(strings.NewReader(input), &result)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Name != "test" || result.Value != 42 {
		t.Errorf("unexpected result: %+v", result)
	}
}

func TestEncodeJSONToReader(t *testing.T) {
	type testStruct struct {
		Name string `json:"name"`
	}
	input := testStruct{Name: "hello"}
	reader, err := EncodeJSONToReader(&input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	body, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("unexpected error reading: %v", err)
	}
	expected := `{"name":"hello"}`
	if string(body) != expected {
		t.Errorf("expected %s, got %s", expected, string(body))
	}
}

func TestResponsesToOpenAICodexShape(t *testing.T) {
	req := &models.ResponsesRequest{
		Model:        "gpt-5.5",
		Instructions: "Follow the developer instructions.",
		Input: []interface{}{
			map[string]interface{}{
				"type": "message",
				"role": "developer",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": "Use the local workspace.",
					},
				},
			},
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{
						"type": "input_text",
						"text": "Reply OK.",
					},
				},
			},
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
			{
				Type:        "custom",
				Name:        "apply_patch",
				Description: "Apply a patch",
			},
			{Type: "web_search"},
		},
		Stream:          true,
		MaxOutputTokens: 100,
	}

	openAIReq := ResponsesToOpenAI(req)
	if openAIReq.Model != "gpt-5.5" {
		t.Fatalf("expected model gpt-5.5, got %s", openAIReq.Model)
	}
	if len(openAIReq.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].Role != "system" {
		t.Fatalf("expected instructions to become system, got %s", openAIReq.Messages[0].Role)
	}
	if openAIReq.Messages[1].Role != "system" {
		t.Fatalf("expected developer input to become system, got %s", openAIReq.Messages[1].Role)
	}
	if openAIReq.Messages[2].Role != "user" {
		t.Fatalf("expected user input to stay user, got %s", openAIReq.Messages[2].Role)
	}
	if len(openAIReq.Tools) != 2 {
		t.Fatalf("expected 2 converted tools, got %d", len(openAIReq.Tools))
	}
	if openAIReq.Tools[0].Function.Name != "shell_command" {
		t.Fatalf("expected shell_command tool, got %s", openAIReq.Tools[0].Function.Name)
	}
	if openAIReq.Tools[1].Function.Name != "apply_patch" {
		t.Fatalf("expected apply_patch fallback tool, got %s", openAIReq.Tools[1].Function.Name)
	}
}

func TestResponsesToOpenAISkipsEmptyContentMessages(t *testing.T) {
	req := &models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "message",
				"role":    "assistant",
				"content": []interface{}{},
			},
			map[string]interface{}{
				"type": "message",
				"role": "user",
				"content": []interface{}{
					map[string]interface{}{"type": "input_text", "text": "OK"},
				},
			},
		},
	}

	openAIReq := ResponsesToOpenAI(req)
	if len(openAIReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].Role != "user" || openAIReq.Messages[0].Content != "OK" {
		t.Fatalf("unexpected message: %+v", openAIReq.Messages[0])
	}
}

func TestResponsesToOpenAIToolRoundTripItems(t *testing.T) {
	req := &models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{
				"type":      "function_call",
				"name":      "exec_command",
				"call_id":   "call_123",
				"arguments": `{"cmd":"pwd"}`,
			},
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_123",
				"output":  "Output:\n/home/xsj\n",
			},
		},
	}

	openAIReq := ResponsesToOpenAI(req)
	if len(openAIReq.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(openAIReq.Messages))
	}
	callMsg := openAIReq.Messages[0]
	if callMsg.Role != "assistant" || len(callMsg.ToolCalls) != 1 {
		t.Fatalf("expected assistant tool call message, got %+v", callMsg)
	}
	if strings.TrimSpace(callMsg.ReasoningContent) == "" {
		t.Fatalf("expected assistant tool call reasoning_content to be populated")
	}
	if callMsg.ToolCalls[0].ID != "call_123" || callMsg.ToolCalls[0].Function.Name != "exec_command" {
		t.Fatalf("unexpected tool call: %+v", callMsg.ToolCalls[0])
	}
	outputMsg := openAIReq.Messages[1]
	if outputMsg.Role != "tool" || outputMsg.ToolCallID != "call_123" {
		t.Fatalf("expected tool output message, got %+v", outputMsg)
	}
	if !strings.Contains(outputMsg.Content, "/home/xsj") {
		t.Fatalf("expected tool output content, got %q", outputMsg.Content)
	}
}

func TestEnsureAssistantToolCallReasoningInResponse(t *testing.T) {
	resp := &models.OpenAIChatCompletionResponse{
		Choices: []models.OpenAIChoice{
			{
				Message: models.OpenAIMessage{
					Role: "assistant",
					ToolCalls: []models.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: models.FunctionCall{
								Name:      "bash",
								Arguments: "{}",
							},
						},
					},
				},
			},
		},
	}

	EnsureAssistantToolCallReasoningInResponse(resp)
	if strings.TrimSpace(resp.Choices[0].Message.ReasoningContent) == "" {
		t.Fatalf("expected reasoning_content to be populated in response")
	}
}

func TestRewriteOpenAIStreamBodyWithReasoning(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"bash","arguments":"{}"}}]}}]}`,
		`data: [DONE]`,
		"",
	}, "\n\n")

	var out bytes.Buffer
	err := RewriteOpenAIStreamBodyWithReasoning(io.NopCloser(strings.NewReader(stream)), &out, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	if !strings.Contains(got, "reasoning_content") {
		t.Fatalf("expected stream to contain reasoning_content, got:\n%s", got)
	}
	if !strings.Contains(got, `[DONE]`) {
		t.Fatalf("expected stream to contain DONE marker, got:\n%s", got)
	}
}

func TestResponsesToOpenAIFunctionCallOutputFallsBackToResult(t *testing.T) {
	req := &models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{
				"type":    "function_call_output",
				"call_id": "call_123",
				"result":  "directory listed",
			},
		},
	}

	openAIReq := ResponsesToOpenAI(req)
	if len(openAIReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].Role != "tool" {
		t.Fatalf("expected tool role, got %s", openAIReq.Messages[0].Role)
	}
	if openAIReq.Messages[0].ToolCallID != "call_123" {
		t.Fatalf("expected tool call id call_123, got %s", openAIReq.Messages[0].ToolCallID)
	}
	if openAIReq.Messages[0].Content != "directory listed" {
		t.Fatalf("expected tool output content, got %s", openAIReq.Messages[0].Content)
	}
}

func TestResponsesToOpenAIToolMessageCarriesToolCallID(t *testing.T) {
	req := &models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{
				"type":         "message",
				"role":         "tool",
				"tool_call_id": "call_456",
				"content": []interface{}{
					map[string]interface{}{
						"type": "output_text",
						"text": "done",
					},
				},
			},
		},
	}

	openAIReq := ResponsesToOpenAI(req)
	if len(openAIReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(openAIReq.Messages))
	}
	msg := openAIReq.Messages[0]
	if msg.Role != "tool" {
		t.Fatalf("expected role tool, got %s", msg.Role)
	}
	if msg.ToolCallID != "call_456" {
		t.Fatalf("expected tool_call_id call_456, got %q", msg.ToolCallID)
	}
	if msg.Content != "done" {
		t.Fatalf("expected content done, got %q", msg.Content)
	}
}

func TestResponsesToOpenAIToolMessageDefaultsBlankContentToJSONObject(t *testing.T) {
	req := &models.ResponsesRequest{
		Model: "gpt-5.5",
		Input: []interface{}{
			map[string]interface{}{
				"type":         "message",
				"role":         "tool",
				"tool_call_id": "call_789",
				"content":      []interface{}{},
			},
		},
	}

	openAIReq := ResponsesToOpenAI(req)
	if len(openAIReq.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(openAIReq.Messages))
	}
	msg := openAIReq.Messages[0]
	if msg.Role != "tool" {
		t.Fatalf("expected role tool, got %s", msg.Role)
	}
	if msg.ToolCallID != "call_789" {
		t.Fatalf("expected tool_call_id call_789, got %q", msg.ToolCallID)
	}
	if msg.Content != "{}" {
		t.Fatalf("expected blank tool content to default to {}, got %q", msg.Content)
	}
}

func TestOpenAIStreamToResponsesSSEIncludesEventNames(t *testing.T) {
	finishReason := "stop"
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","content":"OK"}}]}`,
		`data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"` + finishReason + `"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var out bytes.Buffer
	err := OpenAIStreamToResponsesSSE(io.NopCloser(strings.NewReader(stream)), &out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	for _, expected := range []string{
		"event: response.created\n",
		"event: response.output_item.added\n",
		"event: response.content_part.added\n",
		"event: response.output_text.delta\n",
		"event: response.output_text.done\n",
		"event: response.content_part.done\n",
		"event: response.output_item.done\n",
		"event: response.completed\n",
		"data: [DONE]\n\n",
	} {
		if !strings.Contains(got, expected) {
			t.Fatalf("expected stream to contain %q, got:\n%s", expected, got)
		}
	}
	for _, expected := range []string{
		`"item_id":"msg_1"`,
		`"sequence_number":1`,
		`"sequence_number":8`,
		`"status":"completed"`,
	} {
		if !strings.Contains(got, expected) {
			t.Fatalf("expected stream to contain %q, got:\n%s", expected, got)
		}
	}
}

func TestOpenAIStreamToResponsesSSEFunctionCall(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"shell_command","arguments":"{\"command\""}}]}}]}`,
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"pwd\"}"}}]}}]}`,
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var out bytes.Buffer
	err := OpenAIStreamToResponsesSSE(io.NopCloser(strings.NewReader(stream)), &out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	for _, expected := range []string{
		"event: response.output_item.added\n",
		"event: response.function_call_arguments.delta\n",
		"event: response.function_call_arguments.done\n",
		"event: response.output_item.done\n",
		`"type":"function_call"`,
		`"call_id":"call_123"`,
		`"name":"shell_command"`,
		`"arguments":"{\"command\":\"pwd\"}"`,
		"event: response.completed\n",
	} {
		if !strings.Contains(got, expected) {
			t.Fatalf("expected stream to contain %q, got:\n%s", expected, got)
		}
	}
}

func TestOpenAIStreamToResponsesSSEAcceptsDataPrefixWithoutSpace(t *testing.T) {
	stream := strings.Join([]string{
		`data:{"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"shell_command","arguments":"{\"command\""}}]}}]}`,
		`data:{"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"pwd\"}"}}]}}]}`,
		`data:{"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		`data:[DONE]`,
		``,
	}, "\n\n")

	var out bytes.Buffer
	err := OpenAIStreamToResponsesSSE(io.NopCloser(strings.NewReader(stream)), &out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	for _, expected := range []string{
		"event: response.output_item.added\n",
		"event: response.function_call_arguments.delta\n",
		"event: response.function_call_arguments.done\n",
		`"call_id":"call_123"`,
		`"arguments":"{\"command\":\"pwd\"}"`,
		"event: response.completed\n",
	} {
		if !strings.Contains(got, expected) {
			t.Fatalf("expected stream to contain %q, got:\n%s", expected, got)
		}
	}
}

func TestOpenAIStreamToResponsesSSEIgnoresReasoningContent(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"reasoning_content":"hidden reasoning"}}]}`,
		`data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"OK"}}]}`,
		`data: {"id":"chatcmpl-reasoning","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var out bytes.Buffer
	err := OpenAIStreamToResponsesSSE(io.NopCloser(strings.NewReader(stream)), &out)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	got := out.String()
	if strings.Contains(got, "hidden reasoning") {
		t.Fatalf("expected reasoning_content to be hidden, got:\n%s", got)
	}
	if !strings.Contains(got, `"delta":"OK"`) {
		t.Fatalf("expected visible content to be streamed, got:\n%s", got)
	}
}

func TestCompatibilityMatrixResponsesToolCallAndUsageStream(t *testing.T) {
	stream := strings.Join([]string{
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"shell_command","arguments":"{\"command\""}}]}}]}`,
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"ps\"}"}}]}}]}`,
		`data: {"id":"chatcmpl-tools","object":"chat.completion.chunk","created":1,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`,
		`data: [DONE]`,
		``,
	}, "\n\n")

	var out bytes.Buffer
	var gotUsage models.OpenAIUsage
	err := OpenAIStreamToResponsesSSEWithUsage(io.NopCloser(strings.NewReader(stream)), &out, func(u models.OpenAIUsage) {
		gotUsage = u
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	got := out.String()
	for _, expected := range []string{
		"event: response.function_call_arguments.delta\n",
		"event: response.function_call_arguments.done\n",
		`"type":"function_call"`,
		`"call_id":"call_123"`,
		`"name":"shell_command"`,
		`"arguments":"{\"command\":\"ps\"}"`,
		"event: response.completed\n",
	} {
		if !strings.Contains(got, expected) {
			t.Fatalf("expected stream to contain %q, got:\n%s", expected, got)
		}
	}
	if gotUsage.PromptTokens != 11 || gotUsage.CompletionTokens != 7 || gotUsage.TotalTokens != 18 {
		t.Fatalf("unexpected usage: %+v", gotUsage)
	}
}

func TestCompatibilityMatrixAnthropicThinkingToolUseAndResult(t *testing.T) {
	req := &models.AnthropicMessageRequest{
		Model: "gpt-5.5",
		Messages: []models.AnthropicMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{"type": "thinking", "thinking": "need a process list"},
					map[string]interface{}{"type": "tool_use", "id": "toolu_1", "name": "PowerShell", "input": map[string]interface{}{"command": "Get-Process"}},
				},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "tool_result", "tool_use_id": "toolu_1", "content": "PID Name"},
				},
			},
		},
		Tools: []models.AnthropicTool{{
			Name:        "PowerShell",
			Description: "Run a PowerShell command",
			InputSchema: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{"command": map[string]interface{}{"type": "string"}},
			},
		}},
		ToolChoice: map[string]interface{}{"type": "tool", "name": "PowerShell"},
	}

	openAIReq := AnthropicToOpenAI(req)
	if len(openAIReq.Tools) != 1 || openAIReq.Tools[0].Function.Name != "PowerShell" {
		t.Fatalf("tool declaration did not convert: %+v", openAIReq.Tools)
	}
	if len(openAIReq.Messages) != 2 {
		t.Fatalf("expected assistant tool call and tool result messages, got %d", len(openAIReq.Messages))
	}
	if openAIReq.Messages[0].ReasoningContent != "need a process list" {
		t.Fatalf("missing reasoning content: %+v", openAIReq.Messages[0])
	}
	if len(openAIReq.Messages[0].ToolCalls) != 1 {
		t.Fatalf("missing tool call: %+v", openAIReq.Messages[0])
	}
	if openAIReq.Messages[0].ToolCalls[0].Function.Arguments != `{"command":"Get-Process"}` {
		t.Fatalf("bad tool arguments: %s", openAIReq.Messages[0].ToolCalls[0].Function.Arguments)
	}
	if openAIReq.Messages[1].Role != "tool" || openAIReq.Messages[1].ToolCallID != "toolu_1" || openAIReq.Messages[1].Content != "PID Name" {
		t.Fatalf("bad tool result conversion: %+v", openAIReq.Messages[1])
	}
}

func TestAnthropicToOpenAICompactsHistoricalReasoning(t *testing.T) {
	messages := make([]models.AnthropicMessage, 0, 5)
	for i := 0; i < 5; i++ {
		messages = append(messages, models.AnthropicMessage{
			Role: "assistant",
			Content: []interface{}{
				map[string]interface{}{"type": "thinking", "thinking": strings.Repeat("x", maxRecentReasoningChars+100)},
				map[string]interface{}{"type": "tool_use", "id": "toolu_" + string(rune('a'+i)), "name": "shell", "input": map[string]interface{}{"command": "pwd"}},
			},
		})
	}

	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: messages,
	})

	if got := openAIReq.Messages[0].ReasoningContent; got != reasoningPlaceholder {
		t.Fatalf("old reasoning should be compacted, got %q", got)
	}
	if got := len(openAIReq.Messages[4].ReasoningContent); got != maxRecentReasoningChars {
		t.Fatalf("recent reasoning should be truncated to %d chars, got %d", maxRecentReasoningChars, got)
	}
	if len(openAIReq.Messages[4].ToolCalls) != 1 {
		t.Fatalf("tool call should be preserved: %+v", openAIReq.Messages[4])
	}
}

func TestAnthropicToOpenAICompactsHistoricalToolResults(t *testing.T) {
	messages := make([]models.AnthropicMessage, 0, recentToolMessages+2)
	for i := 0; i < recentToolMessages+2; i++ {
		messages = append(messages, models.AnthropicMessage{
			Role: "user",
			Content: []interface{}{
				map[string]interface{}{
					"type":        "tool_result",
					"tool_use_id": "toolu_" + string(rune('a'+i)),
					"content":     strings.Repeat("o", maxRecentToolChars+100),
				},
			},
		})
	}

	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: messages,
	})

	if got := openAIReq.Messages[0].Content; got != toolResultPlaceholder {
		t.Fatalf("old tool result should be compacted, got %q", got)
	}
	if got := len(openAIReq.Messages[len(openAIReq.Messages)-1].Content); got != maxRecentToolChars {
		t.Fatalf("recent tool result should be truncated to %d chars, got %d", maxRecentToolChars, got)
	}
	if openAIReq.Messages[0].ToolCallID == "" {
		t.Fatalf("tool_call_id should be preserved: %+v", openAIReq.Messages[0])
	}
}

func TestAnthropicToOpenAICompactsHistoricalMessageText(t *testing.T) {
	messages := make([]models.AnthropicMessage, 0, recentTextMessages+2)
	for i := 0; i < recentTextMessages+2; i++ {
		messages = append(messages, models.AnthropicMessage{
			Role: "assistant",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": strings.Repeat("m", maxRecentTextChars+100)},
				map[string]interface{}{"type": "tool_use", "id": "toolu_" + string(rune('a'+i)), "name": "shell", "input": map[string]interface{}{"command": "pwd"}},
			},
		})
	}

	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: messages,
	})

	if got := openAIReq.Messages[0].Content; got != messageTextPlaceholder {
		t.Fatalf("old message text should be compacted, got %q", got)
	}
	if got := len(openAIReq.Messages[len(openAIReq.Messages)-1].Content); got != maxRecentTextChars {
		t.Fatalf("recent message text should be truncated to %d chars, got %d", maxRecentTextChars, got)
	}
	if len(openAIReq.Messages[0].ToolCalls) != 1 {
		t.Fatalf("tool call should be preserved when text is compacted: %+v", openAIReq.Messages[0])
	}
}

func TestAnthropicToOpenAIPreservesLatestUserText(t *testing.T) {
	latest := strings.Repeat("u", maxRecentTextChars+1000)
	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model: "claude-sonnet-4-6",
		Messages: []models.AnthropicMessage{
			{Role: "user", Content: strings.Repeat("old", maxRecentTextChars)},
			{Role: "assistant", Content: strings.Repeat("a", maxRecentTextChars+100)},
			{Role: "user", Content: latest},
		},
	})

	if got := openAIReq.Messages[len(openAIReq.Messages)-1].Content; got != latest {
		t.Fatalf("latest user message should be preserved, got len %d want %d", len(got), len(latest))
	}
	if got := len(openAIReq.Messages[1].Content); got != maxRecentTextChars {
		t.Fatalf("historical assistant text should still be truncated to %d chars, got %d", maxRecentTextChars, got)
	}
}

func TestAnthropicToOpenAIRemovesToolsForPlainQuestion(t *testing.T) {
	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []models.AnthropicMessage{{Role: "user", Content: "hello"}},
		Tools: []models.AnthropicTool{
			{Name: "Bash", InputSchema: map[string]interface{}{"type": "object"}},
			{Name: "Read", InputSchema: map[string]interface{}{"type": "object"}},
		},
		ToolChoice: "auto",
	})
	if len(openAIReq.Tools) != 0 {
		t.Fatalf("plain question should not send tools upstream, got %+v", openAIReq.Tools)
	}
	if openAIReq.ToolChoice != nil {
		t.Fatalf("tool_choice should be omitted when tools are omitted, got %+v", openAIReq.ToolChoice)
	}
}

func TestAnthropicToOpenAICompactsAndFiltersTools(t *testing.T) {
	req := &models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []models.AnthropicMessage{{Role: "user", Content: "run command pwd"}},
		Tools: []models.AnthropicTool{
			{
				Name:        "mcp__local__Bash",
				Description: "long tool description should be removed",
				InputSchema: map[string]interface{}{
					"type":        "object",
					"description": "schema description should be removed",
					"properties": map[string]interface{}{
						"command": map[string]interface{}{
							"type":        "string",
							"description": "field description should be removed",
						},
					},
					"required": []interface{}{"command"},
				},
			},
			{
				Name:        "WebFetch",
				Description: "not in the default allowlist",
				InputSchema: map[string]interface{}{
					"type": "object",
				},
			},
		},
	}

	openAIReq := AnthropicToOpenAI(req)
	if len(openAIReq.Tools) != 1 {
		t.Fatalf("expected only allowed tool, got %+v", openAIReq.Tools)
	}
	if openAIReq.Tools[0].Function.Name != "mcp__local__Bash" {
		t.Fatalf("unexpected tool name: %s", openAIReq.Tools[0].Function.Name)
	}
	if openAIReq.Tools[0].Function.Description != "" {
		t.Fatalf("tool description should be removed")
	}
	if _, ok := openAIReq.Tools[0].Function.Parameters["description"]; ok {
		t.Fatalf("schema description should be removed: %+v", openAIReq.Tools[0].Function.Parameters)
	}
	props := openAIReq.Tools[0].Function.Parameters["properties"].(map[string]interface{})
	command := props["command"].(map[string]interface{})
	if _, ok := command["description"]; ok {
		t.Fatalf("field description should be removed: %+v", command)
	}
}

func TestAnthropicToOpenAISelectsEditTools(t *testing.T) {
	openAIReq := AnthropicToOpenAI(&models.AnthropicMessageRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []models.AnthropicMessage{{Role: "user", Content: "modify the file and fix the bug"}},
		Tools: []models.AnthropicTool{
			{Name: "Bash", InputSchema: map[string]interface{}{"type": "object"}},
			{Name: "Read", InputSchema: map[string]interface{}{"type": "object"}},
			{Name: "Edit", InputSchema: map[string]interface{}{"type": "object"}},
			{Name: "Write", InputSchema: map[string]interface{}{"type": "object"}},
			{Name: "WebFetch", InputSchema: map[string]interface{}{"type": "object"}},
		},
	})
	names := map[string]bool{}
	for _, tool := range openAIReq.Tools {
		names[tool.Function.Name] = true
	}
	if !names["Read"] || !names["Edit"] || !names["Write"] {
		t.Fatalf("expected read/edit/write tools, got %+v", names)
	}
	if names["Bash"] || names["WebFetch"] {
		t.Fatalf("unexpected tools selected: %+v", names)
	}
}
