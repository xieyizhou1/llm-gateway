package usage

import (
	"encoding/json"
	"testing"

	"llm-gateway/internal/models"
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

func TestCacheUsageFromOpenAIUsage(t *testing.T) {
	t.Run("prompt_tokens_details", func(t *testing.T) {
		got := CacheUsageFromOpenAIUsage(models.OpenAIUsage{
			PromptTokens:        1000,
			PromptTokensDetails: models.PromptTokensDetails{CachedTokens: 800},
		})
		if got.CacheReadTokens != 800 {
			t.Fatalf("expected read tokens 800, got %d", got.CacheReadTokens)
		}
		if got.CacheHitRate != 0.8 {
			t.Fatalf("expected hit rate 0.8, got %f", got.CacheHitRate)
		}
	})
	t.Run("prompt_cache_hit_miss", func(t *testing.T) {
		got := CacheUsageFromOpenAIUsage(models.OpenAIUsage{
			PromptCacheHitTokens:  20,
			PromptCacheMissTokens: 80,
		})
		if got.CacheReadTokens != 20 {
			t.Fatalf("expected read tokens 20, got %d", got.CacheReadTokens)
		}
		if got.CacheHitRate != 0.2 {
			t.Fatalf("expected hit rate 0.2, got %f", got.CacheHitRate)
		}
	})
	t.Run("cache_read_tokens_fallback", func(t *testing.T) {
		got := CacheUsageFromOpenAIUsage(models.OpenAIUsage{
			PromptTokens:    500,
			CacheReadTokens: 250,
		})
		if got.CacheReadTokens != 250 {
			t.Fatalf("expected read tokens 250, got %d", got.CacheReadTokens)
		}
		if got.CacheHitRate != 0.5 {
			t.Fatalf("expected hit rate 0.5, got %f", got.CacheHitRate)
		}
	})
}
