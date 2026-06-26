// Package adapter 负责 Anthropic Messages API 与 OpenAI Chat Completions API 之间的协议转换。
package adapter

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"sort"
	"strings"
	"time"

	"llm-gateway/internal/models"
)

// AnthropicToOpenAI 将 Anthropic 请求转换为 OpenAI 请求格式。
func AnthropicToOpenAI(req *models.AnthropicMessageRequest) *models.OpenAIChatCompletionRequest {
	if req == nil {
		return nil
	}
	messages := make([]models.OpenAIMessage, 0, len(req.Messages)+1)

	// Anthropic 顶层 system → OpenAI messages 中 role: "system"
	if req.System != "" {
		messages = append(messages, models.OpenAIMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, m := range req.Messages {
		messages = append(messages, anthropicMessageToOpenAI(m)...)
	}
	EnsureAssistantToolCallReasoning(messages)
	compactReasoningForUpstream(messages)
	compactToolResultsForUpstream(messages)
	compactMessageTextForUpstream(messages)
	tools := anthropicToolsToOpenAI(req.Tools, latestAnthropicUserText(req.Messages), req.ToolChoice)
	toolChoice := anthropicToolChoiceToOpenAI(req.ToolChoice)
	if len(tools) == 0 {
		toolChoice = nil
	}

	return &models.OpenAIChatCompletionRequest{
		Model:       req.Model,
		Messages:    messages,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      req.Stream,
		Tools:       tools,
		ToolChoice:  toolChoice,
	}
}

const (
	recentReasoningMessages   = 1
	maxRecentReasoningChars   = 1024
	reasoningPlaceholder      = "[previous reasoning omitted]"
	reasoningTruncationNotice = "\n[reasoning truncated]"
	recentToolMessages        = 6
	maxRecentToolChars        = 4096
	toolResultPlaceholder     = "[previous tool result omitted]"
	toolResultTruncationNote  = "\n[tool result truncated]"
	recentTextMessages        = 8
	maxRecentTextChars        = 4096
	messageTextPlaceholder    = "[previous message omitted]"
	messageTextTruncationNote = "\n[message truncated]"
)

// DefaultMaxTokens is applied when a client does not specify max_tokens.
// It must be large enough to accommodate both reasoning and final content.
const DefaultMaxTokens = 4096

// autoCacheMinChars is the minimum total message text size before we inject
// prompt-caching hints. Tiny prompts do not benefit from cache writes.
const autoCacheMinChars = 4096

// ephemeralCacheControl is the OpenAI-compatible cache_control marker.
var ephemeralCacheControl = json.RawMessage(`{"type":"ephemeral"}`)

// NormalizeOpenAIRequest fills in provider-friendly defaults for a chat completions request.
func NormalizeOpenAIRequest(req *models.OpenAIChatCompletionRequest) {
	if req == nil {
		return
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = DefaultMaxTokens
	}
}

// InjectPromptCacheControl adds cache_control markers to all messages except the
// final user message when the provider/model supports prompt caching and the
// prompt is large enough to benefit. Existing client-provided markers are kept.
func InjectPromptCacheControl(req *models.OpenAIChatCompletionRequest, supportsCache bool) {
	if req == nil || !supportsCache || len(req.Messages) == 0 {
		return
	}
	total := 0
	for _, m := range req.Messages {
		total += len(m.Content)
		total += len(m.RawContent)
	}
	if total < autoCacheMinChars {
		return
	}
	lastUser := -1
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == "user" {
			lastUser = i
			break
		}
	}
	for i := range req.Messages {
		if i == lastUser {
			continue
		}
		if len(req.Messages[i].CacheControl) == 0 {
			req.Messages[i].CacheControl = ephemeralCacheControl
		}
	}
}

func compactReasoningForUpstream(messages []models.OpenAIMessage) {
	remainingRecent := recentReasoningMessages
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "assistant" {
			continue
		}
		if !isRealReasoning(messages[i].ReasoningContent) {
			// Drop empty or placeholder reasoning so upstream providers never
			// see "[previous reasoning omitted]" in their context.
			messages[i].ReasoningContent = ""
			continue
		}

		if remainingRecent > 0 {
			messages[i].ReasoningContent = trimLongText(messages[i].ReasoningContent, maxRecentReasoningChars, reasoningTruncationNotice)
			remainingRecent--
			continue
		}

		// Drop historical reasoning_content instead of using a placeholder.
		// Placeholder text pollutes the model's context and gets echoed in the
		// current reasoning. Upstream validation only needs the most recent
		// assistant turn to carry reasoning_content when thinking is enabled.
		messages[i].ReasoningContent = ""
	}
}

// isRealReasoning reports whether a reasoning_content value carries actual
// reasoning rather than an empty or placeholder string. WorkBuddy replays the
// placeholder we emit in responses, so we must treat it as absent when
// forwarding upstream to avoid confusing providers like Kimi.
func isRealReasoning(s string) bool {
	s = strings.TrimSpace(s)
	return s != "" && s != reasoningPlaceholder
}

func compactToolResultsForUpstream(messages []models.OpenAIMessage) {
	remainingRecent := recentToolMessages
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "tool" || strings.TrimSpace(messages[i].Content) == "" {
			continue
		}

		if remainingRecent > 0 {
			messages[i].Content = trimLongText(messages[i].Content, maxRecentToolChars, toolResultTruncationNote)
			remainingRecent--
			continue
		}

		messages[i].Content = toolResultPlaceholder
	}
}

func compactMessageTextForUpstream(messages []models.OpenAIMessage) {
	remainingRecent := recentTextMessages
	latestUserIndex := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" && strings.TrimSpace(messages[i].Content) != "" {
			latestUserIndex = i
			break
		}
	}

	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" && messages[i].Role != "assistant" {
			continue
		}
		if strings.TrimSpace(messages[i].Content) == "" {
			continue
		}
		if i == latestUserIndex {
			continue
		}

		if remainingRecent > 0 {
			messages[i].Content = trimLongText(messages[i].Content, maxRecentTextChars, messageTextTruncationNote)
			remainingRecent--
			continue
		}

		messages[i].Content = messageTextPlaceholder
	}
}

func trimLongText(s string, maxChars int, suffix string) string {
	if maxChars <= 0 || len(s) <= maxChars {
		return s
	}
	limit := maxChars - len(suffix)
	if limit < 0 {
		limit = maxChars
		suffix = ""
	}
	return s[:limit] + suffix
}

// AnthropicToOpenAIResponse 将 Anthropic 响应转换为 OpenAI 响应格式。
func AnthropicToOpenAIResponse(anthropicResp *models.AnthropicMessageResponse) *models.OpenAIChatCompletionResponse {
	if anthropicResp == nil {
		return nil
	}
	message := models.OpenAIMessage{
		Role: "assistant",
	}
	finishReason := mapAnthropicStopReason(anthropicResp.StopReason)

	for _, block := range anthropicResp.Content {
		switch block.Type {
		case "text":
			if message.Content != "" {
				message.Content += "\n"
			}
			message.Content += block.Text
		case "thinking":
			if message.ReasoningContent != "" {
				message.ReasoningContent += "\n"
			}
			message.ReasoningContent += block.Thinking
		case "tool_use":
			callID := block.ID
			if callID == "" {
				callID = "call_" + randomID()
			}
			args := "{}"
			if block.Input != nil {
				if s, ok := block.Input.(string); ok {
					s = strings.TrimSpace(s)
					if s != "" {
						args = s
					}
				} else if b, err := json.Marshal(block.Input); err == nil {
					args = string(b)
				}
			}
			message.ToolCalls = append(message.ToolCalls, models.ToolCall{
				ID:   callID,
				Type: "function",
				Function: models.FunctionCall{
					Name:      block.Name,
					Arguments: args,
				},
			})
			if finishReason == "stop" {
				finishReason = "tool_calls"
			}
		}
	}

	return &models.OpenAIChatCompletionResponse{
		ID:      anthropicResp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   anthropicResp.Model,
		Choices: []models.OpenAIChoice{
			{
				Index:        0,
				Message:      message,
				FinishReason: finishReason,
			},
		},
		Usage: models.OpenAIUsage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens,
		},
	}
}

func mapAnthropicStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// OpenAIToAnthropic 将 OpenAI 响应转换为 Anthropic 响应格式。
func OpenAIToAnthropic(openAIResp *models.OpenAIChatCompletionResponse) *models.AnthropicMessageResponse {
	if openAIResp == nil {
		return nil
	}
	content := make([]models.AnthropicContent, 0)
	for _, c := range openAIResp.Choices {
		// If the model only produced reasoning, surface it as visible text too.
		messageContent := c.Message.Content
		if strings.TrimSpace(messageContent) == "" && strings.TrimSpace(c.Message.ReasoningContent) != "" {
			messageContent = c.Message.ReasoningContent
		}
		if c.Message.ReasoningContent != "" {
			content = append(content, models.AnthropicContent{
				Type:     "thinking",
				Thinking: c.Message.ReasoningContent,
			})
		}
		if messageContent != "" {
			content = append(content, models.AnthropicContent{
				Type: "text",
				Text: messageContent,
			})
		}
		for _, tc := range c.Message.ToolCalls {
			callID := tc.ID
			if callID == "" {
				callID = "toolu_" + randomID()
			}
			content = append(content, models.AnthropicContent{
				Type:  "tool_use",
				ID:    callID,
				Name:  tc.Function.Name,
				Input: parseToolArguments(tc.Function.Arguments),
			})
		}
	}

	return &models.AnthropicMessageResponse{
		ID:         openAIResp.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      openAIResp.Model,
		Content:    content,
		StopReason: mapFinishReason(openAIResp.Choices),
		Usage: models.AnthropicUsage{
			InputTokens:  openAIResp.Usage.PromptTokens,
			OutputTokens: openAIResp.Usage.CompletionTokens,
		},
	}
}

// OpenAIStreamChunkToAnthropic 将 OpenAI SSE chunk 转换为 Anthropic SSE chunk。
func OpenAIStreamChunkToAnthropic(chunk *models.OpenAIStreamChunk) []models.AnthropicStreamChunk {
	result := make([]models.AnthropicStreamChunk, 0, len(chunk.Choices))

	for _, choice := range chunk.Choices {
		if choice.Delta.ReasoningContent != "" {
			result = append(result, models.AnthropicStreamChunk{
				Type:  "content_block_delta",
				Index: choice.Index,
				Delta: &models.AnthropicStreamDelta{
					Type:     "thinking_delta",
					Thinking: choice.Delta.ReasoningContent,
				},
			})
		}
		if choice.Delta.Content != "" {
			result = append(result, models.AnthropicStreamChunk{
				Type:  "content_block_delta",
				Index: choice.Index,
				Delta: &models.AnthropicStreamDelta{
					Type: "text_delta",
					Text: choice.Delta.Content,
				},
			})
		}
		for _, tc := range choice.Delta.ToolCalls {
			result = append(result, models.AnthropicStreamChunk{
				Type:  "content_block_delta",
				Index: tc.Index,
				Delta: &models.AnthropicStreamDelta{
					Type:        "input_json_delta",
					PartialJSON: tc.Function.Arguments,
				},
			})
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			result = append(result, models.AnthropicStreamChunk{
				Type: "message_stop",
			})
		}
	}

	return result
}

// AnthropicStreamStart 生成 Anthropic SSE 流开始事件。
func AnthropicStreamStart(id, model string) *models.AnthropicStreamChunk {
	return &models.AnthropicStreamChunk{
		Type: "message_start",
		Message: &models.AnthropicStreamMessage{
			ID:    id,
			Type:  "message",
			Role:  "assistant",
			Model: model,
		},
	}
}

// TransformOpenAIStreamBody 读取 OpenAI SSE 流，逐行转换为 Anthropic SSE 格式并写入输出。
func TransformOpenAIStreamBody(body io.ReadCloser, w io.Writer) error {
	defer body.Close()
	_, err := io.Copy(w, body)
	return err
}

// MapModel 根据 Provider 的 model_map 将客户端模型名映射为 Provider 模型名。
func MapModel(clientModel string, modelMap map[string]string) string {
	if mapped, ok := modelMap[clientModel]; ok {
		return mapped
	}
	return clientModel
}

func mapFinishReason(choices []models.OpenAIChoice) string {
	if len(choices) == 0 {
		return "end_turn"
	}
	switch choices[0].FinishReason {
	case "stop":
		return "end_turn"
	case "tool_calls":
		return "tool_use"
	case "length":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

// ExtractSystemMessage 从 OpenAI messages 中提取 system 消息作为 Anthropic 顶层 system。
func ExtractSystemMessage(messages []models.OpenAIMessage) (string, []models.OpenAIMessage) {
	var system string
	filtered := make([]models.OpenAIMessage, 0, len(messages))
	for _, m := range messages {
		if m.Role == "system" {
			system = m.Content
		} else {
			filtered = append(filtered, m)
		}
	}
	return system, filtered
}

// OpenAIToAnthropicRequest 将 OpenAI 请求转换为 Anthropic 请求（用于测试或反向场景）。
func OpenAIToAnthropicRequest(req *models.OpenAIChatCompletionRequest) *models.AnthropicMessageRequest {
	if req == nil {
		return nil
	}
	system, filtered := ExtractSystemMessage(req.Messages)

	anthropicMessages := make([]models.AnthropicMessage, 0, len(filtered))
	for _, m := range filtered {
		msg := openAIMessageToAnthropic(m)
		if msg.Role == "" {
			continue
		}
		anthropicMessages = append(anthropicMessages, msg)
	}

	return &models.AnthropicMessageRequest{
		Model:       req.Model,
		Messages:    anthropicMessages,
		System:      system,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      req.Stream,
		Tools:       openAIToolsToAnthropic(req.Tools),
		ToolChoice:  req.ToolChoice,
	}
}

func openAIMessageToAnthropic(m models.OpenAIMessage) models.AnthropicMessage {
	switch m.Role {
	case "user":
		return models.AnthropicMessage{Role: "user", Content: openAIUserContentToAnthropic(m)}
	case "assistant":
		return models.AnthropicMessage{Role: "assistant", Content: openAIAssistantContentToAnthropic(m)}
	case "tool":
		return openAIToolResultToAnthropic(m)
	default:
		return models.AnthropicMessage{Role: m.Role, Content: m.Content}
	}
}

func openAIUserContentToAnthropic(m models.OpenAIMessage) interface{} {
	if len(m.RawContent) > 0 {
		parts, err := parseOpenAIContentParts(m.RawContent)
		if err == nil {
			blocks := make([]interface{}, 0, len(parts))
			for _, part := range parts {
				block := openAIContentPartToAnthropic(part)
				if block != nil {
					blocks = append(blocks, block)
				}
			}
			if len(blocks) > 0 {
				return blocks
			}
		}
	}
	if m.Content != "" {
		return m.Content
	}
	return ""
}

func openAIAssistantContentToAnthropic(m models.OpenAIMessage) interface{} {
	blocks := make([]interface{}, 0)

	if m.ReasoningContent != "" {
		blocks = append(blocks, map[string]interface{}{
			"type":     "thinking",
			"thinking": m.ReasoningContent,
		})
	}

	for _, tc := range m.ToolCalls {
		input := parseToolArguments(tc.Function.Arguments)
		callID := tc.ID
		if callID == "" {
			callID = "toolu_" + randomID()
		}
		blocks = append(blocks, map[string]interface{}{
			"type":  "tool_use",
			"id":    callID,
			"name":  tc.Function.Name,
			"input": input,
		})
	}

	if m.Content != "" {
		blocks = append(blocks, map[string]interface{}{
			"type": "text",
			"text": m.Content,
		})
	}

	if len(blocks) == 0 {
		return ""
	}
	if len(blocks) == 1 {
		if textBlock, ok := blocks[0].(map[string]interface{}); ok && textBlock["type"] == "text" {
			return textBlock["text"]
		}
	}
	return blocks
}

func openAIToolResultToAnthropic(m models.OpenAIMessage) models.AnthropicMessage {
	content := m.Content
	if content == "" {
		content = "{}"
	}
	return models.AnthropicMessage{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{
				"type":        "tool_result",
				"tool_use_id": m.ToolCallID,
				"content":     content,
			},
		},
	}
}

func parseOpenAIContentParts(raw []byte) ([]map[string]interface{}, error) {
	var parts []map[string]interface{}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, err
	}
	return parts, nil
}

func openAIContentPartToAnthropic(part map[string]interface{}) interface{} {
	partType, _ := part["type"].(string)
	switch partType {
	case "text", "input_text":
		text, _ := part["text"].(string)
		if text != "" {
			return map[string]interface{}{
				"type": "text",
				"text": text,
			}
		}
	case "image_url":
		if imageURL, ok := part["image_url"].(map[string]interface{}); ok {
			url, _ := imageURL["url"].(string)
			if url != "" {
				return map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type":      "url",
						"url":       url,
						"media_type": inferMediaTypeFromURL(url),
					},
				}
			}
		}
	case "input_image":
		if dataURL, _ := part["data"].(string); dataURL != "" {
			mediaType, data := splitDataURL(dataURL)
			if data != "" {
				return map[string]interface{}{
					"type": "image",
					"source": map[string]interface{}{
						"type":       "base64",
						"media_type": mediaType,
						"data":       data,
					},
				}
			}
		}
	}
	return nil
}

func inferMediaTypeFromURL(url string) string {
	url = strings.ToLower(url)
	switch {
	case strings.HasSuffix(url, ".png"):
		return "image/png"
	case strings.HasSuffix(url, ".jpg"), strings.HasSuffix(url, ".jpeg"):
		return "image/jpeg"
	case strings.HasSuffix(url, ".gif"):
		return "image/gif"
	case strings.HasSuffix(url, ".webp"):
		return "image/webp"
	default:
		return "image/jpeg"
	}
}

func splitDataURL(dataURL string) (string, string) {
	const prefix = "data:"
	if !strings.HasPrefix(dataURL, prefix) {
		return "image/jpeg", ""
	}
	after := strings.TrimPrefix(dataURL, prefix)
	idx := strings.Index(after, ";base64,")
	if idx < 0 {
		return "image/jpeg", ""
	}
	mediaType := after[:idx]
	if mediaType == "" {
		mediaType = "image/jpeg"
	}
	return mediaType, after[idx+len(";base64,"):]
}

func openAIToolsToAnthropic(tools []models.OpenAITool) []models.AnthropicTool {
	result := make([]models.AnthropicTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Function.Name == "" {
			continue
		}
		result = append(result, models.AnthropicTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
		})
	}
	return result
}

// SerializeAnthropicStreamChunk 将 Anthropic StreamChunk 序列化为 SSE data 行。
func SerializeAnthropicStreamChunk(chunk *models.AnthropicStreamChunk) (string, error) {
	b, err := json.Marshal(chunk)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("data: %s\n\n", b), nil
}

// SerializeOpenAIStreamChunk 将 OpenAI StreamChunk 序列化为 SSE data 行。
func SerializeOpenAIStreamChunk(chunk *models.OpenAIStreamChunk) (string, error) {
	b, err := json.Marshal(chunk)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("data: %s\n\n", b), nil
}

// ParseOpenAIStreamChunk 从 SSE data 行解析 OpenAI StreamChunk。
func ParseOpenAIStreamChunk(line string) (*models.OpenAIStreamChunk, error) {
	line = strings.TrimPrefix(line, "data:")
	line = strings.TrimSpace(line)
	if line == "[DONE]" || line == "" {
		return nil, nil
	}
	var chunk models.OpenAIStreamChunk
	if err := json.Unmarshal([]byte(line), &chunk); err != nil {
		return nil, err
	}
	return &chunk, nil
}

// ResponsesToOpenAI 将 Responses API 请求转换为 OpenAI Chat Completions 请求。
func ResponsesToOpenAI(req *models.ResponsesRequest) *models.OpenAIChatCompletionRequest {
	messages := make([]models.OpenAIMessage, 0)

	// instructions → system message
	if req.Instructions != "" {
		messages = append(messages, models.OpenAIMessage{
			Role:    "system",
			Content: req.Instructions,
		})
	}

	// input → messages
	switch input := req.Input.(type) {
	case string:
		messages = append(messages, models.OpenAIMessage{
			Role:    "user",
			Content: input,
		})
	case []interface{}:
		for _, item := range input {
			if m, ok := item.(map[string]interface{}); ok {
				messages = append(messages, responsesInputItemToOpenAI(m)...)
			}
		}
	case []map[string]interface{}:
		for _, m := range input {
			messages = append(messages, responsesInputItemToOpenAI(m)...)
		}
	}
	EnsureAssistantToolCallReasoning(messages)

	return &models.OpenAIChatCompletionRequest{
		Model:         req.Model,
		Messages:      messages,
		Temperature:   req.Temperature,
		Stream:        req.Stream,
		StreamOptions: map[string]interface{}{"include_usage": true},
		MaxTokens:     defaultResponsesMaxTokens(req.MaxOutputTokens),
		Tools:         ResponsesToolsToOpenAI(req.Tools),
		ToolChoice:    req.ToolChoice,
	}
}

func defaultResponsesMaxTokens(v int) int {
	if v <= 0 {
		return DefaultMaxTokens
	}
	return v
}

// EnsureAssistantToolCallReasoning ensures assistant tool-call turns do not
// carry an empty or placeholder reasoning_content upstream. Some providers
// reject the field when it is blank, and others (like Kimi) echo placeholder
// text in their own reasoning, so we drop it and let omitempty remove it from
// the JSON.
func EnsureAssistantToolCallReasoning(messages []models.OpenAIMessage) {
	for i := range messages {
		if messages[i].Role != "assistant" || len(messages[i].ToolCalls) == 0 {
			continue
		}
		if !isRealReasoning(messages[i].ReasoningContent) {
			messages[i].ReasoningContent = ""
		}
	}
}

// EnsureAssistantToolCallReasoningInResponse normalizes assistant tool-call
// messages in a Chat Completions response before it is returned to the client.
func EnsureAssistantToolCallReasoningInResponse(resp *models.OpenAIChatCompletionResponse) {
	if resp == nil {
		return
	}
	for i := range resp.Choices {
		msg := &resp.Choices[i].Message
		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			if strings.TrimSpace(msg.ReasoningContent) == "" {
				msg.ReasoningContent = reasoningPlaceholder
			}
		}
		// Surface reasoning as visible content when the model produced no answer text.
		// This prevents empty replies for thinking-first providers like Kimi.
		if strings.TrimSpace(msg.Content) == "" && strings.TrimSpace(msg.ReasoningContent) != "" {
			msg.Content = msg.ReasoningContent
		}
	}
}

// RewriteOpenAIStreamBodyWithReasoning rewrites an OpenAI SSE stream so
// assistant tool-call chunks retain a non-empty reasoning_content field.
// If onFinishReason is non-nil, it is called with the first non-nil finish_reason encountered.
func RewriteOpenAIStreamBodyWithReasoning(body io.Reader, w io.Writer, onFinishReason func(string), onUsage func(*models.OpenAIUsage)) error {
	reader := bufio.NewReader(body)

	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}

		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "data:") {
			if strings.EqualFold(strings.TrimSpace(strings.TrimPrefix(trimmed, "data:")), "[DONE]") {
				if _, writeErr := io.WriteString(w, line); writeErr != nil {
					return writeErr
				}
			} else {
				chunk, parseErr := ParseOpenAIStreamChunk(trimmed)
				if parseErr != nil {
					if _, writeErr := io.WriteString(w, line); writeErr != nil {
						return writeErr
					}
				} else if chunk != nil {
					for i := range chunk.Choices {
						if len(chunk.Choices[i].Delta.ToolCalls) > 0 {
							// Keep real reasoning; use a placeholder only when the
							// upstream stream omitted reasoning entirely. Some
							// clients require a non-empty value on tool-call deltas.
							if !isRealReasoning(chunk.Choices[i].Delta.ReasoningContent) {
								chunk.Choices[i].Delta.ReasoningContent = reasoningPlaceholder
							}
						}
						if onFinishReason != nil && chunk.Choices[i].FinishReason != nil && *chunk.Choices[i].FinishReason != "" {
							onFinishReason(*chunk.Choices[i].FinishReason)
							onFinishReason = nil // only call once
						}
					}
					if onUsage != nil && chunk.Usage != nil {
						onUsage(chunk.Usage)
						onUsage = nil // only call once
					}
					serialized, serErr := SerializeOpenAIStreamChunk(chunk)
					if serErr != nil {
						return serErr
					}
					if _, writeErr := io.WriteString(w, serialized); writeErr != nil {
						return writeErr
					}
				}
			}
		} else {
			if _, writeErr := io.WriteString(w, line); writeErr != nil {
				return writeErr
			}
		}

		if err == io.EOF {
			break
		}
	}

	return nil
}

func normalizeResponsesRole(role, content string) string {
	switch role {
	case "developer", "system":
		return "system"
	case "assistant", "user", "tool":
		return role
	case "":
		if strings.Contains(content, "<permissions instructions>") {
			return "system"
		}
		return "user"
	default:
		return "user"
	}
}

func responsesInputItemToOpenAI(m map[string]interface{}) []models.OpenAIMessage {
	itemType, _ := m["type"].(string)
	switch itemType {
	case "function_call":
		name, _ := m["name"].(string)
		arguments, _ := m["arguments"].(string)
		callID := responseCallIDField(m)
		if name == "" || callID == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:    "assistant",
			Content: "",
			ToolCalls: []models.ToolCall{{
				ID:   callID,
				Type: "function",
				Function: models.FunctionCall{
					Name:      name,
					Arguments: arguments,
				},
			}},
		}}
	case "function_call_output":
		callID := responseCallIDField(m)
		content := extractResponsesToolOutput(m)
		if content == "" {
			content = "{}"
		}
		if callID == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:       "tool",
			Content:    content,
			ToolCallID: callID,
		}}
	case "reasoning":
		return nil
	default:
		content := extractResponsesToolOutput(m)
		if content == "" {
			content = extractContent(m["content"])
		}
		if content == "" {
			content = responseStringField(m, "text")
		}
		role, _ := m["role"].(string)
		role = normalizeResponsesRole(role, content)
		msg := models.OpenAIMessage{
			Role:    role,
			Content: content,
		}
		if role == "tool" {
			if content == "" {
				msg.Content = "{}"
			}
			msg.ToolCallID = responseCallIDField(m)
		}
		if msg.Role != "tool" && strings.TrimSpace(msg.Content) == "" {
			return nil
		}
		return []models.OpenAIMessage{msg}
	}
}

func firstString(values ...interface{}) string {
	for _, v := range values {
		switch x := v.(type) {
		case string:
			if strings.TrimSpace(x) != "" {
				return strings.TrimSpace(x)
			}
		}
	}
	return ""
}

func responseStringField(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func parseJSONValue(raw string) interface{} {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]interface{}{}
	}

	var v interface{}
	if err := json.Unmarshal([]byte(raw), &v); err == nil {
		return v
	}
	return raw
}

func parseToolArguments(raw string) interface{} {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]interface{}{}
	}

	parsed := parseJSONValue(raw)
	if _, ok := parsed.(string); !ok {
		return parsed
	}

	repaired := make(map[string]interface{})
	for _, key := range []string{"command", "description"} {
		if value := extractJSONStringField(raw, key); value != "" {
			repaired[key] = value
		}
	}
	if len(repaired) > 0 {
		return repaired
	}
	return parsed
}

func extractJSONStringField(raw, key string) string {
	pattern := `"` + key + `"`
	idx := strings.Index(raw, pattern)
	if idx < 0 {
		return ""
	}
	rest := raw[idx+len(pattern):]
	colon := strings.Index(rest, ":")
	if colon < 0 {
		return ""
	}
	rest = strings.TrimLeft(rest[colon+1:], " \t\r\n")
	if !strings.HasPrefix(rest, `"`) {
		return ""
	}

	dec := json.NewDecoder(strings.NewReader(rest))
	var value string
	if err := dec.Decode(&value); err != nil {
		return ""
	}
	return value
}

func anthropicToolsToOpenAI(tools []models.AnthropicTool, latestUserText string, toolChoice interface{}) []models.OpenAITool {
	sort.SliceStable(tools, func(i, j int) bool {
		return tools[i].Name < tools[j].Name
	})
	filtered := filterAnthropicTools(tools, latestUserText, toolChoice)
	if len(filtered) == 0 && len(tools) > 0 {
		return nil
	}

	result := make([]models.OpenAITool, 0, len(filtered))
	for _, tool := range filtered {
		if tool.Name == "" {
			continue
		}
		parameters := compactToolSchema(tool.InputSchema)
		if parameters == nil {
			parameters = map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"input": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"input"},
			}
		}
		result = append(result, models.OpenAITool{
			Type: "function",
			Function: models.OpenAIFunction{
				Name:       tool.Name,
				Parameters: parameters,
			},
		})
	}
	return result
}

var defaultAnthropicToolAllowlist = map[string]struct{}{
	"bash":         {},
	"read":         {},
	"write":        {},
	"edit":         {},
	"multiedit":    {},
	"grep":         {},
	"glob":         {},
	"ls":           {},
	"todowrite":    {},
	"notebookread": {},
}

func filterAnthropicTools(tools []models.AnthropicTool, latestUserText string, toolChoice interface{}) []models.AnthropicTool {
	if len(tools) == 0 {
		return nil
	}
	mode := inferToolMode(latestUserText)
	forced := forcedToolNames(toolChoice)
	result := make([]models.AnthropicTool, 0, len(tools))
	for _, tool := range tools {
		normalized := normalizeToolName(tool.Name)
		if forced[normalized] || isAllowedToolForMode(normalized, mode) {
			result = append(result, tool)
		}
	}
	return result
}

func isAllowedAnthropicTool(name string) bool {
	normalized := normalizeToolName(name)
	if normalized == "" {
		return false
	}
	_, ok := defaultAnthropicToolAllowlist[normalized]
	return ok
}

type toolMode int

const (
	toolModeNone toolMode = iota
	toolModeRead
	toolModeEdit
	toolModeRun
	toolModeTodo
)

func inferToolMode(text string) toolMode {
	t := strings.ToLower(text)
	if strings.TrimSpace(t) == "" {
		return toolModeNone
	}
	runTerms := []string{"run ", "execute", "command", "shell", "bash", "powershell", "terminal", "test", "npm ", "python ", "go test", "pytest", "执行", "运行", "测试", "命令", "终端"}
	editTerms := []string{"edit", "modify", "change", "fix", "implement", "patch", "write", "create", "delete", "refactor", "apply", "修改", "修复", "实现", "改", "写入", "创建", "删除", "重构", "部署"}
	readTerms := []string{"read", "show", "inspect", "find", "search", "grep", "list", "file", "code", "path", "打开", "读取", "查看", "搜索", "查找", "文件", "代码", "目录"}
	todoTerms := []string{"todo", "plan", "task list", "待办", "计划", "任务"}

	if containsAny(t, runTerms) {
		return toolModeRun
	}
	if containsAny(t, editTerms) {
		return toolModeEdit
	}
	if containsAny(t, readTerms) {
		return toolModeRead
	}
	if containsAny(t, todoTerms) {
		return toolModeTodo
	}
	return toolModeNone
}

func containsAny(text string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(text, term) {
			return true
		}
	}
	return false
}

func isAllowedToolForMode(name string, mode toolMode) bool {
	switch mode {
	case toolModeRead:
		return name == "read" || name == "grep" || name == "glob" || name == "ls" || name == "notebookread"
	case toolModeEdit:
		return name == "read" || name == "grep" || name == "glob" || name == "ls" || name == "edit" || name == "multiedit" || name == "write"
	case toolModeRun:
		return name == "bash" || name == "read" || name == "grep" || name == "glob" || name == "ls"
	case toolModeTodo:
		return name == "todowrite"
	default:
		return false
	}
}

func forcedToolNames(toolChoice interface{}) map[string]bool {
	result := map[string]bool{}
	switch choice := toolChoice.(type) {
	case map[string]interface{}:
		if name := firstString(choice["name"], choice["tool_name"]); name != "" {
			result[normalizeToolName(name)] = true
		}
	}
	return result
}

func latestAnthropicUserText(messages []models.AnthropicMessage) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		if text := extractAnthropicContent(messages[i].Content); strings.TrimSpace(text) != "" {
			return text
		}
	}
	return ""
}

func normalizeToolName(name string) string {
	name = strings.TrimSpace(strings.ToLower(name))
	if name == "" {
		return ""
	}
	parts := strings.Split(name, "__")
	name = parts[len(parts)-1]
	name = strings.ReplaceAll(name, "-", "")
	name = strings.ReplaceAll(name, "_", "")
	name = strings.ReplaceAll(name, " ", "")
	return name
}

func compactToolSchema(schema map[string]interface{}) map[string]interface{} {
	if schema == nil {
		return nil
	}
	if compacted, ok := compactSchemaValue(schema).(map[string]interface{}); ok {
		return compacted
	}
	return schema
}

func compactSchemaValue(v interface{}) interface{} {
	switch x := v.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(x))
		for k, value := range x {
			switch k {
			case "description", "title", "examples", "default", "$comment":
				continue
			case "enum":
				if arr, ok := value.([]interface{}); ok && len(arr) > 12 {
					out[k] = arr[:12]
					continue
				}
			}
			out[k] = compactSchemaValue(value)
		}
		return out
	case []interface{}:
		out := make([]interface{}, 0, len(x))
		for _, item := range x {
			out = append(out, compactSchemaValue(item))
		}
		return out
	default:
		return v
	}
}

func anthropicToolChoiceToOpenAI(v interface{}) interface{} {
	switch choice := v.(type) {
	case nil:
		return nil
	case string:
		return choice
	case map[string]interface{}:
		if name := firstString(choice["name"], choice["tool_name"]); name != "" {
			return map[string]interface{}{
				"type": "function",
				"name": name,
			}
		}
		if typ, _ := choice["type"].(string); typ != "" {
			switch typ {
			case "auto", "none", "required":
				return typ
			}
		}
		return choice
	default:
		return v
	}
}

func anthropicMessageToOpenAI(msg models.AnthropicMessage) []models.OpenAIMessage {
	switch msg.Role {
	case "assistant":
		return anthropicAssistantContentToOpenAI(msg.Content)
	case "user":
		return anthropicUserContentToOpenAI(msg.Content)
	case "tool":
		return anthropicToolContentToOpenAI(msg.Content)
	default:
		content := extractAnthropicContent(msg.Content)
		if strings.TrimSpace(content) == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:    msg.Role,
			Content: content,
		}}
	}
}

func anthropicAssistantContentToOpenAI(content interface{}) []models.OpenAIMessage {
	blocks := anthropicContentBlocks(content)
	if len(blocks) == 0 {
		text := extractAnthropicContent(content)
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:    "assistant",
			Content: text,
		}}
	}

	var textParts []string
	var reasoningParts []string
	toolCalls := make([]models.ToolCall, 0)
	for _, block := range blocks {
		switch strings.TrimSpace(firstString(block["type"])) {
		case "text":
			if text := firstString(block["text"]); text != "" {
				textParts = append(textParts, text)
			}
		case "thinking":
			if thinking := firstString(block["thinking"], block["text"]); thinking != "" {
				reasoningParts = append(reasoningParts, thinking)
			}
		case "tool_use":
			name := firstString(block["name"])
			if name == "" {
				continue
			}
			callID := firstString(block["id"])
			if callID == "" {
				callID = "call_" + randomID()
			}
			toolCalls = append(toolCalls, models.ToolCall{
				ID:   callID,
				Type: "function",
				Function: models.FunctionCall{
					Name:      name,
					Arguments: anthropicToolInputToJSON(block["input"]),
				},
			})
		}
	}

	msg := models.OpenAIMessage{
		Role:             "assistant",
		Content:          strings.Join(textParts, ""),
		ReasoningContent: strings.Join(reasoningParts, ""),
		ToolCalls:        toolCalls,
	}
	if strings.TrimSpace(msg.Content) == "" && strings.TrimSpace(msg.ReasoningContent) == "" && len(msg.ToolCalls) == 0 {
		return nil
	}
	return []models.OpenAIMessage{msg}
}

func anthropicUserContentToOpenAI(content interface{}) []models.OpenAIMessage {
	blocks := anthropicContentBlocks(content)
	if len(blocks) == 0 {
		text := extractAnthropicContent(content)
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:    "user",
			Content: text,
		}}
	}

	result := make([]models.OpenAIMessage, 0, len(blocks))
	var userParts []map[string]interface{}
	flushUserParts := func() {
		if len(userParts) == 0 {
			return
		}
		// Preserve backward compatibility: plain text-only user messages use
		// the simple string Content field rather than a RawContent array.
		if len(userParts) == 1 && userParts[0]["type"] == "text" {
			result = append(result, models.OpenAIMessage{
				Role:    "user",
				Content: fmt.Sprintf("%v", userParts[0]["text"]),
			})
			userParts = nil
			return
		}
		raw, err := json.Marshal(userParts)
		if err != nil {
			return
		}
		result = append(result, models.OpenAIMessage{
			Role:       "user",
			RawContent: raw,
		})
		userParts = nil
	}

	for _, block := range blocks {
		switch strings.TrimSpace(firstString(block["type"])) {
		case "text":
			if text := firstString(block["text"]); text != "" {
				userParts = append(userParts, map[string]interface{}{
					"type": "text",
					"text": text,
				})
			}
		case "image":
			if imageURL := anthropicImageBlockToOpenAI(block); imageURL != "" {
				userParts = append(userParts, map[string]interface{}{
					"type": "image_url",
					"image_url": map[string]interface{}{
						"url": imageURL,
					},
				})
			}
		case "tool_result":
			flushUserParts()
			callID := firstString(block["tool_use_id"], block["id"])
			if callID == "" {
				continue
			}
			contentText := anthropicToolResultContent(block["content"])
			if contentText == "" {
				contentText = anthropicToolResultContent(block["text"])
			}
			if contentText == "" {
				contentText = "{}"
			}
			result = append(result, models.OpenAIMessage{
				Role:       "tool",
				Content:    contentText,
				ToolCallID: callID,
			})
		}
	}

	flushUserParts()
	return result
}

// anthropicImageBlockToOpenAI converts an Anthropic image block to an OpenAI
// image_url data URL. Anthropic format:
//   {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
func anthropicImageBlockToOpenAI(block map[string]interface{}) string {
	source, _ := block["source"].(map[string]interface{})
	if source == nil {
		return ""
	}
	if firstString(source["type"]) != "base64" {
		return ""
	}
	mediaType := firstString(source["media_type"])
	if mediaType == "" {
		mediaType = "image/png"
	}
	data, _ := source["data"].(string)
	if data == "" {
		return ""
	}
	return fmt.Sprintf("data:%s;base64,%s", mediaType, data)
}

func anthropicToolContentToOpenAI(content interface{}) []models.OpenAIMessage {
	blocks := anthropicContentBlocks(content)
	if len(blocks) == 0 {
		text := extractAnthropicContent(content)
		if strings.TrimSpace(text) == "" {
			return nil
		}
		return []models.OpenAIMessage{{
			Role:    "tool",
			Content: text,
		}}
	}

	result := make([]models.OpenAIMessage, 0, len(blocks))
	for _, block := range blocks {
		callID := firstString(block["tool_use_id"], block["id"])
		if callID == "" {
			continue
		}
		contentText := anthropicToolResultContent(block["content"])
		if contentText == "" {
			contentText = anthropicToolResultContent(block["text"])
		}
		if contentText == "" {
			contentText = "{}"
		}
		result = append(result, models.OpenAIMessage{
			Role:       "tool",
			Content:    contentText,
			ToolCallID: callID,
		})
	}
	return result
}

func anthropicContentBlocks(v interface{}) []map[string]interface{} {
	switch c := v.(type) {
	case []interface{}:
		blocks := make([]map[string]interface{}, 0, len(c))
		for _, item := range c {
			if m, ok := item.(map[string]interface{}); ok {
				blocks = append(blocks, m)
			}
		}
		return blocks
	case []map[string]interface{}:
		return c
	default:
		return nil
	}
}

func anthropicToolInputToJSON(v interface{}) string {
	if v == nil {
		return "{}"
	}
	switch x := v.(type) {
	case string:
		if strings.TrimSpace(x) == "" {
			return "{}"
		}
		return x
	default:
		b, err := json.Marshal(x)
		if err != nil {
			return "{}"
		}
		return string(b)
	}
}

func anthropicToolResultContent(v interface{}) string {
	switch c := v.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, item := range c {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					sb.WriteString(text)
				}
			}
		}
		return sb.String()
	case []map[string]interface{}:
		var sb strings.Builder
		for _, m := range c {
			if text, ok := m["text"].(string); ok {
				sb.WriteString(text)
			}
		}
		return sb.String()
	case map[string]interface{}:
		if text, ok := c["text"].(string); ok {
			return text
		}
		b, err := json.Marshal(c)
		if err == nil {
			return string(b)
		}
	}
	return ""
}

func extractResponsesToolOutput(m map[string]interface{}) string {
	if m == nil {
		return ""
	}
	for _, key := range []string{"output", "content", "text", "result"} {
		if s := extractContent(m[key]); strings.TrimSpace(s) != "" {
			return s
		}
		if s, ok := m[key].(string); ok && strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

func responseCallIDField(m map[string]interface{}) string {
	return firstString(m["tool_call_id"], m["call_id"], m["id"])
}

// ResponsesToolsToOpenAI converts Responses API tool declarations to
// Chat Completions function tools for upstream providers such as DeepSeek.
func ResponsesToolsToOpenAI(tools []models.ResponsesTool) []models.OpenAITool {
	result := make([]models.OpenAITool, 0, len(tools))
	for _, tool := range tools {
		fn, ok := responseToolToFunction(tool)
		if !ok {
			continue
		}
		result = append(result, models.OpenAITool{
			Type:     "function",
			Function: fn,
		})
	}
	return result
}

func responseToolToFunction(tool models.ResponsesTool) (models.OpenAIFunction, bool) {
	if tool.Function != nil && tool.Function.Name != "" {
		return *tool.Function, true
	}
	if tool.Name == "" {
		return models.OpenAIFunction{}, false
	}

	parameters := tool.Parameters
	if parameters == nil {
		parameters = map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"input": map[string]interface{}{
					"type": "string",
				},
			},
			"required": []string{"input"},
		}
	}

	return models.OpenAIFunction{
		Name:        tool.Name,
		Description: tool.Description,
		Parameters:  parameters,
	}, true
}

// extractAnthropicContent 从 Anthropic message content 中提取文本。
// content 可能是字符串，也可能是 [{"type":"text","text":"..."}] 数组。
func extractAnthropicContent(v interface{}) string {
	switch c := v.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, item := range c {
			if m, ok := item.(map[string]interface{}); ok {
				if text, ok := m["text"].(string); ok {
					sb.WriteString(text)
				}
			}
		}
		return sb.String()
	case []map[string]interface{}:
		var sb strings.Builder
		for _, m := range c {
			if text, ok := m["text"].(string); ok {
				sb.WriteString(text)
			}
		}
		return sb.String()
	}
	return ""
}

// extractContent 从 Responses API input item 中提取文本内容。
// content 可能是字符串，也可能是 [{"type":"text","text":"..."}] 数组。
func extractContent(v interface{}) string {
	switch c := v.(type) {
	case string:
		return c
	case []interface{}:
		var sb strings.Builder
		for _, item := range c {
			if m, ok := item.(map[string]interface{}); ok {
				// Codex CLI / Responses may use "input_text" for prompt text and
				// "output_text" for tool output or returned content.
				itemType, _ := m["type"].(string)
				if text, ok := m["text"].(string); ok {
					if itemType == "input_text" || itemType == "output_text" || itemType == "text" || itemType == "" {
						sb.WriteString(text)
					}
				}
			}
		}
		return sb.String()
	case []map[string]interface{}:
		var sb strings.Builder
		for _, m := range c {
			itemType, _ := m["type"].(string)
			if text, ok := m["text"].(string); ok {
				if itemType == "input_text" || itemType == "output_text" || itemType == "text" || itemType == "" {
					sb.WriteString(text)
				}
			}
		}
		return sb.String()
	}
	return ""
}

// OpenAIToResponses 将 OpenAI Chat Completions 响应转换为 Responses API 响应。
func OpenAIToResponses(openAIResp *models.OpenAIChatCompletionResponse) *models.ResponsesResponse {
	output := make([]models.ResponsesOutput, 0)
	for _, choice := range openAIResp.Choices {
		// 先处理普通文本内容
		content := make([]models.ResponsesContent, 0)
		if choice.Message.Content != "" {
			content = append(content, models.ResponsesContent{
				Type: "output_text",
				Text: choice.Message.Content,
			})
		}
		output = append(output, models.ResponsesOutput{
			ID:      fmt.Sprintf("msg_%s_%d", responseID(openAIResp.ID), choice.Index),
			Type:    "message",
			Status:  "completed",
			Role:    "assistant",
			Content: content,
		})

		// 再处理 tool_calls，每个转为独立的 function_call 输出项
		for _, tc := range choice.Message.ToolCalls {
			callID := tc.ID
			if callID == "" {
				callID = "call_" + randomID()
			}
			output = append(output, models.ResponsesOutput{
				Type:      "function_call",
				Status:    "completed",
				ID:        callID,
				CallID:    callID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			})
		}
	}

	return &models.ResponsesResponse{
		ID:        openAIResp.ID,
		Object:    "response",
		CreatedAt: openAIResp.Created,
		Model:     openAIResp.Model,
		Status:    "completed",
		Output:    output,
		Usage:     openAIResp.Usage,
	}
}

// CopyAndDecodeJSON 读取 io.Reader 中的 JSON 并解码到目标结构。
func CopyAndDecodeJSON(r io.Reader, v interface{}) error {
	return json.NewDecoder(r).Decode(v)
}

// EncodeJSONToReader 将结构体编码为 io.Reader。
func EncodeJSONToReader(v interface{}) (io.Reader, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(b), nil
}

// OpenAIStreamToResponsesSSE 将上游 Chat Completions SSE 流实时转换为 Responses API SSE 格式。
// It emits typed SSE events close to the OpenAI Responses stream shape that Codex consumes.
func OpenAIStreamToResponsesSSE(body io.ReadCloser, w io.Writer) error {
	return OpenAIStreamToResponsesSSEWithUsage(body, w, nil)
}

// OpenAIStreamToResponsesSSEWithUsage converts Chat Completions SSE to Responses SSE
// and reports the final upstream usage chunk when present.
func OpenAIStreamToResponsesSSEWithUsage(body io.Reader, w io.Writer, onUsage func(models.OpenAIUsage)) error {
	reader := bufio.NewReader(body)

	state := newResponsesStreamState()

	for {
		line, err := reader.ReadString('\n')
		if err != nil && err != io.EOF {
			return err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			if err == io.EOF {
				break
			}
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			if err == io.EOF {
				break
			}
			continue
		}

		data := strings.TrimPrefix(line, "data:")
		data = strings.TrimSpace(data)

		if data == "[DONE]" {
			state.finish(w)
			fmt.Fprintf(w, "data: [DONE]\n\n")
			return nil
		}

		var chunk models.OpenAIStreamChunk
		if unmarshalErr := json.Unmarshal([]byte(data), &chunk); unmarshalErr != nil {
			if err == io.EOF {
				break
			}
			continue
		}
		if chunk.Usage != nil && onUsage != nil {
			onUsage(*chunk.Usage)
		}

		state.start(w, &chunk)

		for _, choice := range chunk.Choices {
			// reasoning_content is not visible assistant output. Codex consumes
			// response.output_text.* as final message text, so only forward content.
			text := choice.Delta.Content
			if text != "" {
				state.emitTextDelta(w, text)
			}
			for _, tc := range choice.Delta.ToolCalls {
				state.emitToolCallDelta(w, tc)
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				state.finishItems(w)
			}
		}

		if err == io.EOF {
			break
		}
	}

	if state.started {
		state.finish(w)
		fmt.Fprintf(w, "data: [DONE]\n\n")
	}

	return nil
}

type responsesStreamState struct {
	seq              int
	started          bool
	respID           string
	model            string
	createdAt        int64
	textItemStarted  bool
	textItemDone     bool
	textItemID       string
	textOutputIndex  int
	accumulatedText  strings.Builder
	nextOutputIndex  int
	toolCalls        map[int]*responsesToolCallState
	completedOutputs []map[string]interface{}
}

type responsesToolCallState struct {
	itemID      string
	callID      string
	name        string
	arguments   strings.Builder
	outputIndex int
	done        bool
}

func newResponsesStreamState() *responsesStreamState {
	return &responsesStreamState{
		textOutputIndex:  -1,
		toolCalls:        make(map[int]*responsesToolCallState),
		completedOutputs: make([]map[string]interface{}, 0),
	}
}

func (s *responsesStreamState) start(w io.Writer, chunk *models.OpenAIStreamChunk) {
	if s.started {
		return
	}
	s.started = true
	s.respID = chunk.ID
	if s.respID == "" {
		s.respID = "resp_" + randomID()
	}
	s.model = chunk.Model
	s.createdAt = chunk.Created
	sendSSEEvent(w, s.nextEvent(map[string]interface{}{
		"type": "response.created",
		"response": map[string]interface{}{
			"id":         s.respID,
			"object":     "response",
			"created_at": s.createdAt,
			"status":     "in_progress",
			"model":      s.model,
			"output":     []interface{}{},
		},
	}))
}

func (s *responsesStreamState) emitTextDelta(w io.Writer, text string) {
	s.ensureTextItem(w)
	s.accumulatedText.WriteString(text)
	sendSSEEvent(w, s.nextEvent(map[string]interface{}{
		"type":          "response.output_text.delta",
		"item_id":       s.textItemID,
		"output_index":  s.textOutputIndex,
		"content_index": 0,
		"delta":         text,
	}))
}

func (s *responsesStreamState) ensureTextItem(w io.Writer) {
	if s.textItemStarted {
		return
	}
	s.textItemStarted = true
	s.textOutputIndex = s.nextOutputIndex
	s.nextOutputIndex++
	s.textItemID = "msg_" + responseID(s.respID)
	item := map[string]interface{}{
		"id":      s.textItemID,
		"type":    "message",
		"status":  "in_progress",
		"role":    "assistant",
		"content": []interface{}{},
	}
	sendSSEEvent(w, s.nextEvent(map[string]interface{}{
		"type":         "response.output_item.added",
		"output_index": s.textOutputIndex,
		"item":         item,
	}))
	sendSSEEvent(w, s.nextEvent(map[string]interface{}{
		"type":          "response.content_part.added",
		"item_id":       s.textItemID,
		"output_index":  s.textOutputIndex,
		"content_index": 0,
		"part": map[string]interface{}{
			"type": "output_text",
			"text": "",
		},
	}))
}

func (s *responsesStreamState) emitToolCallDelta(w io.Writer, tc models.ToolCall) {
	key := tc.Index
	if key == 0 && tc.ID != "" {
		for existingKey, existing := range s.toolCalls {
			if existing.callID == tc.ID {
				key = existingKey
				break
			}
		}
	}

	tool := s.toolCalls[key]
	if tool == nil {
		callID := tc.ID
		if callID == "" {
			callID = "call_" + randomID()
		}
		tool = &responsesToolCallState{
			itemID:      callID,
			callID:      callID,
			name:        tc.Function.Name,
			outputIndex: s.nextOutputIndex,
		}
		s.nextOutputIndex++
		s.toolCalls[key] = tool
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":         "response.output_item.added",
			"output_index": tool.outputIndex,
			"item": map[string]interface{}{
				"id":        tool.itemID,
				"type":      "function_call",
				"status":    "in_progress",
				"call_id":   tool.callID,
				"name":      tool.name,
				"arguments": "",
			},
		}))
	}
	if tc.ID != "" {
		tool.callID = tc.ID
		tool.itemID = tc.ID
	}
	if tc.Function.Name != "" {
		tool.name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		tool.arguments.WriteString(tc.Function.Arguments)
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":         "response.function_call_arguments.delta",
			"item_id":      tool.itemID,
			"output_index": tool.outputIndex,
			"delta":        tc.Function.Arguments,
		}))
	}
}

func (s *responsesStreamState) finishItems(w io.Writer) {
	if s.textItemStarted && !s.textItemDone {
		text := s.accumulatedText.String()
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":          "response.output_text.done",
			"item_id":       s.textItemID,
			"output_index":  s.textOutputIndex,
			"content_index": 0,
			"text":          text,
		}))
		part := map[string]interface{}{"type": "output_text", "text": text}
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":          "response.content_part.done",
			"item_id":       s.textItemID,
			"output_index":  s.textOutputIndex,
			"content_index": 0,
			"part":          part,
		}))
		item := map[string]interface{}{
			"id":      s.textItemID,
			"type":    "message",
			"status":  "completed",
			"role":    "assistant",
			"content": []map[string]interface{}{part},
		}
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": s.textOutputIndex,
			"item":         item,
		}))
		s.completedOutputs = append(s.completedOutputs, item)
		s.textItemDone = true
	}

	keys := make([]int, 0, len(s.toolCalls))
	for key := range s.toolCalls {
		keys = append(keys, key)
	}
	sort.Ints(keys)
	for _, key := range keys {
		tool := s.toolCalls[key]
		if tool.done {
			continue
		}
		args := tool.arguments.String()
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":         "response.function_call_arguments.done",
			"item_id":      tool.itemID,
			"output_index": tool.outputIndex,
			"arguments":    args,
		}))
		item := map[string]interface{}{
			"id":        tool.itemID,
			"type":      "function_call",
			"status":    "completed",
			"call_id":   tool.callID,
			"name":      tool.name,
			"arguments": args,
		}
		sendSSEEvent(w, s.nextEvent(map[string]interface{}{
			"type":         "response.output_item.done",
			"output_index": tool.outputIndex,
			"item":         item,
		}))
		s.completedOutputs = append(s.completedOutputs, item)
		tool.done = true
	}
}

func (s *responsesStreamState) finish(w io.Writer) {
	s.finishItems(w)
	sendSSEEvent(w, s.nextEvent(map[string]interface{}{
		"type": "response.completed",
		"response": map[string]interface{}{
			"id":         s.respID,
			"object":     "response",
			"status":     "completed",
			"model":      s.model,
			"output":     s.completedOutputs,
			"created_at": s.createdAt,
		},
	}))
}

func (s *responsesStreamState) nextEvent(event map[string]interface{}) map[string]interface{} {
	s.seq++
	event["sequence_number"] = s.seq
	return event
}

func responseID(id string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		return randomID()
	}
	id = strings.NewReplacer("resp_", "", "chatcmpl-", "", " ", "_").Replace(id)
	return id
}

// sendSSEEvent 将事件对象序列化为 SSE data 行并写入 w，并立即 Flush。
func sendSSEEvent(w io.Writer, event map[string]interface{}) {
	b, _ := json.Marshal(event)
	if eventType, ok := event["type"].(string); ok && eventType != "" {
		fmt.Fprintf(w, "event: %s\n", eventType)
	}
	fmt.Fprintf(w, "data: %s\n\n", b)
	if fw, ok := w.(interface{ Flush() error }); ok {
		_ = fw.Flush()
	}
}

// randomID 生成一个短随机 ID 字符串。
func randomID() string {
	const charset = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 12)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}
