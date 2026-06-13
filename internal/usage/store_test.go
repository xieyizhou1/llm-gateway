package usage

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

func TestRecordPersistsRequestLog(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	rec := RequestLog{
		TraceID:               "trace-test",
		CreatedAt:             time.Now().UTC(),
		Method:                "POST",
		Path:                  "/v1/responses",
		Protocol:              "responses",
		VirtualKeyHash:        "hash",
		User:                  "default",
		APIKey:                "sk-...test",
		ClientIP:              "203.0.113.8",
		Model:                 "gpt-5.5",
		Provider:              "kimi",
		ProviderKeyID:         "primary",
		UpstreamModel:         "kimi-for-coding",
		RouterDecision:        "kimi/primary",
		Stream:                true,
		HasImage:              true,
		StatusCode:            200,
		LatencyMS:             123,
		TTFTMS:                45,
		GenerationMS:          78,
		OutputTokensPerSecond: 64.1,
		PromptBucket:          "<10K",
		SlowReason:            "",
		PromptTokens:          12,
		OutputTokens:          5,
		TotalTokens:           17,
		RequestBody:           `{"model":"gpt-5.5"}`,
		ResponseBody:          `{"id":"resp"}`,
	}
	if err := store.Record(context.Background(), rec); err != nil {
		t.Fatalf("record: %v", err)
	}
	rows, err := store.Requests(context.Background(), RequestFilter{Limit: 1})
	if err != nil {
		t.Fatalf("requests: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	got := rows[0]
	if got.TraceID != "trace-test" || got.ClientIP != "203.0.113.8" || !got.HasImage || got.TTFTMS != 45 || got.GenerationMS != 78 || got.PromptBucket != "<10K" || got.PromptTokens != 12 || got.OutputTokens != 5 || got.TotalTokens != 17 || got.RequestBody != `{"model":"gpt-5.5"}` || got.ResponseBody != `{"id":"resp"}` {
		t.Fatalf("unexpected row: %+v", got)
	}

	byTrace, err := store.RequestByTraceID(context.Background(), "trace-test")
	if err != nil {
		t.Fatalf("request by trace: %v", err)
	}
	if byTrace == nil || byTrace.TraceID != "trace-test" || byTrace.RequestBody != `{"model":"gpt-5.5"}` || byTrace.ResponseBody != `{"id":"resp"}` {
		t.Fatalf("unexpected by trace: %+v", byTrace)
	}
}

func TestDashboardReadModelsReturnConcreteData(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer store.Close()

	now := time.Now().UTC()
	records := []RequestLog{
		{
			TraceID:         "ok",
			CreatedAt:       now,
			Method:          "POST",
			Path:            "/v1/responses",
			Protocol:        "responses",
			VirtualKeyHash:  "hash",
			User:            "team-a",
			APIKey:          "sk-...team",
			ClientIP:        "198.51.100.20",
			Model:           "gpt-5.5",
			Provider:        "kimi",
			ProviderKeyID:   "kimi-1",
			UpstreamModel:   "kimi-for-coding",
			RouterDecision:  "kimi/kimi-1",
			StatusCode:      200,
			LatencyMS:       150,
			TTFTMS:          30,
			GenerationMS:    120,
			PromptTokens:    100,
			OutputTokens:    40,
			TotalTokens:     140,
			CacheReadTokens: 20,
			HasImage:        true,
			RequestBody:     `{"model":"gpt-5.5"}`,
			ResponseBody:    `{"id":"ok"}`,
		},
		{
			TraceID:        "err",
			CreatedAt:      now,
			Method:         "POST",
			Path:           "/v1/messages",
			Protocol:       "anthropic_messages",
			VirtualKeyHash: "hash",
			User:           "team-a",
			APIKey:         "sk-...team",
			ClientIP:       "198.51.100.20",
			Model:          "claude-sonnet-4-6",
			Provider:       "kimi",
			ProviderKeyID:  "kimi-1",
			UpstreamModel:  "kimi-for-coding",
			RouterDecision: "kimi/kimi-1",
			StatusCode:     502,
			LatencyMS:      2300,
			SlowReason:     "slow_request",
			PromptTokens:   50,
			OutputTokens:   0,
			TotalTokens:    50,
			ErrorCode:      "gateway_error",
			ErrorMessage:   "provider failed",
			RequestBody:    `{"model":"claude-sonnet-4-6"}`,
			ResponseBody:   `{"error":"provider failed"}`,
		},
	}
	for _, rec := range records {
		if err := store.Record(context.Background(), rec); err != nil {
			t.Fatalf("record %s: %v", rec.TraceID, err)
		}
	}

	summary, err := store.Summary(context.Background(), 24)
	if err != nil {
		t.Fatalf("summary: %v", err)
	}
	if summary.Requests != 2 || summary.Errors != 1 || summary.AvgLatencyMS == 0 {
		t.Fatalf("unexpected summary: %+v", summary)
	}

	usageData, err := store.UsageDashboard(context.Background(), 14, 10)
	if err != nil {
		t.Fatalf("usage dashboard: %v", err)
	}
	if len(usageData.Daily) == 0 || usageData.TokenMix.PromptTokens != 150 || usageData.TokenMix.OutputTokens != 40 || len(usageData.Recent) != 2 {
		t.Fatalf("unexpected usage data: %+v", usageData)
	}

	stats, err := store.KeyStats(context.Background(), 24)
	if err != nil {
		t.Fatalf("key stats: %v", err)
	}
	if len(stats) != 1 || stats[0].Provider != "kimi" || stats[0].Requests != 2 || stats[0].Errors != 1 || stats[0].SuccessRate != 0.5 {
		t.Fatalf("unexpected provider stats: %+v", stats)
	}

	errorData, err := store.ErrorDashboard(context.Background(), 24, 10)
	if err != nil {
		t.Fatalf("error dashboard: %v", err)
	}
	if len(errorData.ErrorRequests) != 1 || len(errorData.SlowRequests) != 1 {
		t.Fatalf("unexpected error data: %+v", errorData)
	}
}
