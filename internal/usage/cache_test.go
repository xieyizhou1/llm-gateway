package usage

import (
	"encoding/json"
	"testing"
)

func TestParseOpenAICacheUsagePromptTokenDetails(t *testing.T) {
	raw := map[string]json.RawMessage{
		"prompt_tokens_details": json.RawMessage(`{"cached_tokens":750}`),
	}
	got := ParseOpenAICacheUsage(raw, 1000)
	if got.CacheReadTokens != 750 {
		t.Fatalf("expected cached tokens 750, got %d", got.CacheReadTokens)
	}
	if got.CacheHitRate != 0.75 {
		t.Fatalf("expected hit rate 0.75, got %f", got.CacheHitRate)
	}
}

func TestParseOpenAICacheUsagePromptCacheHitMiss(t *testing.T) {
	raw := map[string]json.RawMessage{
		"prompt_cache_hit_tokens":  json.RawMessage(`30`),
		"prompt_cache_miss_tokens": json.RawMessage(`70`),
	}
	got := ParseOpenAICacheUsage(raw, 100)
	if got.CacheReadTokens != 30 {
		t.Fatalf("expected read tokens 30, got %d", got.CacheReadTokens)
	}
	if got.CacheHitRate != 0.3 {
		t.Fatalf("expected hit rate 0.3, got %f", got.CacheHitRate)
	}
}
