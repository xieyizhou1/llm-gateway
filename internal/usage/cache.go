// Package usage contains cache usage normalization for provider responses.
package usage

import (
	"encoding/json"
	"strconv"

	"llm-gateway/internal/models"
)

// CacheUsage stores the first-stage cache metrics that we keep in logs.
type CacheUsage struct {
	CacheReadTokens  int
	CacheWriteTokens int
	CacheHitRate     float64
}

// ParseOpenAICacheUsage extracts the small cache subset we care about now.
// If the provider does not return any cache-related fields, everything stays 0.
func ParseOpenAICacheUsage(raw map[string]json.RawMessage, promptTokens int) CacheUsage {
	out := CacheUsage{}
	if len(raw) == 0 {
		return out
	}

	if n, ok := parseOptionalIntField(raw, "cache_read_tokens"); ok {
		out.CacheReadTokens = n
	}
	if n, ok := parseOptionalIntField(raw, "cache_write_tokens"); ok {
		out.CacheWriteTokens = n
	}
	if n, ok := parseNestedIntField(raw, "prompt_tokens_details", "cached_tokens"); ok {
		if out.CacheReadTokens == 0 {
			out.CacheReadTokens = n
		}
	}
	hitTokens, hitOK := parseOptionalIntField(raw, "prompt_cache_hit_tokens")
	missTokens, missOK := parseOptionalIntField(raw, "prompt_cache_miss_tokens")
	if out.CacheReadTokens == 0 && hitOK {
		out.CacheReadTokens = hitTokens
	}
	if rate, ok := parseOptionalFloatField(raw, "cache_hit_rate"); ok {
		out.CacheHitRate = rate
		return out
	}
	if hitOK || missOK {
		total := hitTokens + missTokens
		if total > 0 {
			out.CacheHitRate = float64(hitTokens) / float64(total)
		}
		return out
	}
	if out.CacheReadTokens > 0 {
		// prompt token total comes from the caller, not necessarily from raw usage.
		if promptTokens > 0 {
			out.CacheHitRate = float64(out.CacheReadTokens) / float64(promptTokens)
		}
	}
	return out
}

func parseNestedIntField(raw map[string]json.RawMessage, parent string, child string) (int, bool) {
	data, ok := raw[parent]
	if !ok {
		return 0, false
	}
	var nested map[string]json.RawMessage
	if err := json.Unmarshal(data, &nested); err != nil {
		return 0, false
	}
	return parseOptionalIntField(nested, child)
}

func parseOptionalIntField(raw map[string]json.RawMessage, key string) (int, bool) {
	data, ok := raw[key]
	if !ok {
		return 0, false
	}
	var n int
	if err := json.Unmarshal(data, &n); err == nil {
		return n, true
	}
	var f float64
	if err := json.Unmarshal(data, &f); err == nil {
		return int(f), true
	}
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		if parsed, err := strconv.Atoi(s); err == nil {
			return parsed, true
		}
	}
	return 0, false
}

func parseOptionalFloatField(raw map[string]json.RawMessage, key string) (float64, bool) {
	data, ok := raw[key]
	if !ok {
		return 0, false
	}
	var n float64
	if err := json.Unmarshal(data, &n); err == nil {
		return n, true
	}
	return 0, false
}

// CacheUsageFromOpenAIUsage extracts cache metrics from a parsed OpenAI usage
// object. It normalises the different field names used by providers (OpenAI,
// DeepSeek, Kimi) so streaming callbacks can record cache hits the same way
// non-streaming responses do.
func CacheUsageFromOpenAIUsage(u models.OpenAIUsage) CacheUsage {
	out := CacheUsage{}

	cacheRead := u.CacheReadTokens
	if cacheRead == 0 && u.PromptTokensDetails.CachedTokens > 0 {
		cacheRead = u.PromptTokensDetails.CachedTokens
	}
	if cacheRead == 0 && u.PromptCacheHitTokens > 0 {
		cacheRead = u.PromptCacheHitTokens
	}
	out.CacheReadTokens = cacheRead
	out.CacheWriteTokens = u.CacheWriteTokens

	if u.PromptCacheHitTokens > 0 || u.PromptCacheMissTokens > 0 {
		total := u.PromptCacheHitTokens + u.PromptCacheMissTokens
		if total > 0 {
			out.CacheHitRate = float64(u.PromptCacheHitTokens) / float64(total)
		}
		return out
	}
	if out.CacheReadTokens > 0 && u.PromptTokens > 0 {
		out.CacheHitRate = float64(out.CacheReadTokens) / float64(u.PromptTokens)
	}
	return out
}
