package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"

	mw "llm-gateway/internal/middleware"
	"llm-gateway/internal/models"
	"llm-gateway/internal/usage"
)

func ensureStreamUsage(req *models.OpenAIChatCompletionRequest) {
	if req == nil || !req.Stream {
		return
	}
	if req.StreamOptions == nil {
		req.StreamOptions = map[string]interface{}{}
	}
	req.StreamOptions["include_usage"] = true
}

func captureOpenAIStreamUsage(data []byte, record *usage.RequestLog) {
	if record == nil || len(data) == 0 {
		return
	}
	parts := bytes.Split(data, []byte("\n"))
	for _, part := range parts {
		part = bytes.TrimSpace(part)
		if !bytes.HasPrefix(part, []byte("data:")) {
			continue
		}
		raw := bytes.TrimSpace(bytes.TrimPrefix(part, []byte("data:")))
		if len(raw) == 0 || bytes.Equal(raw, []byte("[DONE]")) {
			continue
		}
		var chunk models.OpenAIStreamChunk
		if err := json.Unmarshal(raw, &chunk); err == nil && chunk.Usage != nil {
			applyOpenAIUsage(record, *chunk.Usage)
		}
	}
}

func applyOpenAIUsage(record *usage.RequestLog, u models.OpenAIUsage) {
	if record == nil {
		return
	}
	if u.PromptTokens != 0 {
		record.PromptTokens = u.PromptTokens
	}
	if u.CompletionTokens != 0 {
		record.OutputTokens = u.CompletionTokens
	}
	if u.TotalTokens != 0 {
		record.TotalTokens = u.TotalTokens
	} else {
		record.TotalTokens = record.PromptTokens + record.OutputTokens
	}
}

func applyPerformanceFields(record *usage.RequestLog) {
	if record == nil {
		return
	}
	record.PromptBucket = promptBucket(record.PromptTokens)
	if record.TTFTMS > 0 && record.LatencyMS > record.TTFTMS {
		record.GenerationMS = record.LatencyMS - record.TTFTMS
	}
	if record.GenerationMS > 0 && record.OutputTokens > 0 {
		record.OutputTokensPerSecond = float64(record.OutputTokens) / (float64(record.GenerationMS) / 1000)
	}
	record.SlowReason = slowReason(record)
}

func promptBucket(tokens int) string {
	switch {
	case tokens >= 120000:
		return "120K+"
	case tokens >= 80000:
		return "80K-120K"
	case tokens >= 50000:
		return "50K-80K"
	case tokens >= 10000:
		return "10K-50K"
	default:
		return "<10K"
	}
}

func slowReason(record *usage.RequestLog) string {
	if record == nil {
		return ""
	}
	reasons := make([]string, 0, 3)
	if record.PromptTokens >= 120000 {
		reasons = append(reasons, "large_prompt")
	}
	if record.TTFTMS >= 10000 {
		reasons = append(reasons, "slow_first_token")
	}
	if record.GenerationMS >= 30000 {
		reasons = append(reasons, "slow_generation")
	}
	if record.FallbackCount > 0 || record.StatusCode >= fiber.StatusInternalServerError {
		reasons = append(reasons, "upstream_error_retry")
	}
	return strings.Join(reasons, ",")
}

func shouldLogRequestShape(traceID string, record *usage.RequestLog) bool {
	if record == nil {
		return false
	}
	if record.StatusCode >= fiber.StatusBadRequest {
		return true
	}
	if isSlowUsageRecord(record) {
		return true
	}
	return isShapeSample(traceID)
}

func isSlowUsageRecord(record *usage.RequestLog) bool {
	if record == nil {
		return false
	}
	return record.LatencyMS >= 10000 || record.TTFTMS >= 5000 || record.PromptTokens >= 50000
}

func isShapeSample(traceID string) bool {
	sum := sha256.Sum256([]byte(traceID))
	return sum[0] < 26
}

func anthropicRequestShape(req *models.AnthropicMessageRequest) map[string]interface{} {
	shape := map[string]interface{}{
		"tool_schema_chars": 0,
	}
	if req == nil {
		return shape
	}
	for _, tool := range req.Tools {
		if b, err := json.Marshal(tool.InputSchema); err == nil {
			shape["tool_schema_chars"] = shape["tool_schema_chars"].(int) + len(b)
		}
	}
	return shape
}

func openAIRequestShape(req *models.OpenAIChatCompletionRequest) map[string]interface{} {
	shape := map[string]interface{}{
		"tool_schema_chars": 0,
	}
	if req == nil {
		return shape
	}
	for _, tool := range req.Tools {
		if b, err := json.Marshal(tool.Function.Parameters); err == nil {
			shape["tool_schema_chars"] = shape["tool_schema_chars"].(int) + len(b)
		}
	}
	return shape
}

func logSampledRequestShapes(logger *mw.Logger, traceID string, record *usage.RequestLog, anthropicShape, openAIShape map[string]interface{}) {
	if logger == nil || record == nil || anthropicShape == nil || openAIShape == nil {
		return
	}
	if !shouldLogRequestShape(traceID, record) {
		return
	}
	logger.Info(traceID, "handler", "request_shape", map[string]interface{}{
		"status":          record.StatusCode,
		"latency_ms":      record.LatencyMS,
		"ttft_ms":         record.TTFTMS,
		"prompt_tokens":   record.PromptTokens,
		"output_tokens":   record.OutputTokens,
		"cache_hit_rate":  record.CacheHitRate,
		"sampled":         isShapeSample(traceID),
		"slow":            isSlowUsageRecord(record),
		"anthropic_shape": anthropicShape,
		"openai_shape":    openAIShape,
	})
}

func markTTFT(record *usage.RequestLog, start time.Time) {
	if record == nil || !record.Stream || record.TTFTMS != 0 || start.IsZero() {
		return
	}
	record.TTFTMS = time.Since(start).Milliseconds()
}
