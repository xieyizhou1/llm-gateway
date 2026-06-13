// Package models 定义 OpenAI Responses API 的请求和响应结构体。
package models

// ResponsesInputItem 是 Responses API 的输入项。
type ResponsesInputItem struct {
	Type    string `json:"type"`
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
	Text    string `json:"text,omitempty"`
}

// ResponsesTool is the Responses API tool declaration shape.
// Codex sends function tools as {"type":"function","name":...,"parameters":...}
// while older tests may send Chat Completions style {"type":"function","function":...}.
type ResponsesTool struct {
	Type        string                 `json:"type"`
	Name        string                 `json:"name,omitempty"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Strict      interface{}            `json:"strict,omitempty"`
	Function    *OpenAIFunction        `json:"function,omitempty"`
	Format      map[string]interface{} `json:"format,omitempty"`
}

// ResponsesRequest 对应 OpenAI /v1/responses 请求体。
type ResponsesRequest struct {
	Model              string          `json:"model"`
	Input              interface{}     `json:"input"` // string or []ResponsesInputItem
	Instructions       string          `json:"instructions,omitempty"`
	Temperature        float64         `json:"temperature,omitempty"`
	MaxOutputTokens    int             `json:"max_output_tokens,omitempty"`
	Stream             bool            `json:"stream,omitempty"`
	Tools              []ResponsesTool `json:"tools,omitempty"`
	ToolChoice         interface{}     `json:"tool_choice,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	ParallelToolCalls  bool            `json:"parallel_tool_calls,omitempty"`
}

// ResponsesResponse 对应 OpenAI /v1/responses 非流式响应。
type ResponsesResponse struct {
	ID                string            `json:"id"`
	Object            string            `json:"object"`
	CreatedAt         int64             `json:"created_at"`
	Model             string            `json:"model"`
	Status            string            `json:"status"`
	Output            []ResponsesOutput `json:"output"`
	Usage             OpenAIUsage       `json:"usage"`
	ParallelToolCalls bool              `json:"parallel_tool_calls,omitempty"`
}

// ResponsesOutput 是响应中的单个输出项。
type ResponsesOutput struct {
	Type      string             `json:"type"`
	Status    string             `json:"status,omitempty"`
	Role      string             `json:"role,omitempty"`
	Content   []ResponsesContent `json:"content,omitempty"`
	ID        string             `json:"id,omitempty"`
	CallID    string             `json:"call_id,omitempty"`
	Name      string             `json:"name,omitempty"`
	Arguments string             `json:"arguments,omitempty"`
}

// ResponsesContent 是输出项中的内容块。
type ResponsesContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// ResponsesStreamEvent 是 SSE 流式响应中的事件。
type ResponsesStreamEvent struct {
	Type     string             `json:"type"`
	Response *ResponsesResponse `json:"response,omitempty"`
	Item     *ResponsesOutput   `json:"item,omitempty"`
	Part     *ResponsesContent  `json:"part,omitempty"`
	Delta    *ResponsesContent  `json:"delta,omitempty"`
}
