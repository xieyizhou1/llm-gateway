// Package usage persists request-level accounting and dashboard metadata.
package usage

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	_ "modernc.org/sqlite"

	"llm-gateway/internal/auth"
	"llm-gateway/internal/config"
)

type Store struct {
	db *sql.DB
}

type RequestLog struct {
	TraceID               string    `json:"trace_id"`
	CreatedAt             time.Time `json:"created_at"`
	Method                string    `json:"method"`
	Path                  string    `json:"path"`
	Protocol              string    `json:"protocol"`
	VirtualKeyHash        string    `json:"virtual_key_hash"`
	User                  string    `json:"user"`
	APIKey                string    `json:"api_key"`
	ClientIP              string    `json:"client_ip"`
	Model                 string    `json:"model"`
	Provider              string    `json:"provider"`
	ProviderKeyID         string    `json:"provider_key_id"`
	UpstreamModel         string    `json:"upstream_model"`
	RouterDecision        string    `json:"router_decision"`
	FallbackCount         int       `json:"fallback_count"`
	Stream                bool      `json:"stream"`
	HasImage              bool      `json:"has_image"`
	Recorded              bool      `json:"-"` // transient: true if already saved to DB
	StreamProcessing      bool      `json:"-"` // transient: true if stream is being processed asynchronously
	StreamCompleted       bool      `json:"-"` // transient: true if stream processing finished
	StatusCode            int       `json:"status_code"`
	LatencyMS             int64     `json:"latency_ms"`
	TTFTMS                int64     `json:"ttft_ms"`
	GenerationMS          int64     `json:"generation_ms"`
	OutputTokensPerSecond float64   `json:"output_tokens_per_second"`
	PromptBucket          string    `json:"prompt_bucket"`
	SlowReason            string    `json:"slow_reason"`
	PromptTokens          int       `json:"prompt_tokens"`
	OutputTokens          int       `json:"output_tokens"`
	TotalTokens           int       `json:"total_tokens"`
	CostCents             int64     `json:"cost_cents"`
	CacheReadTokens       int       `json:"cache_read_tokens"`
	CacheWriteTokens      int       `json:"cache_write_tokens"`
	CacheHitRate          float64   `json:"cache_hit_rate"`
	InputHash             string    `json:"input_hash"`
	OutputHash            string    `json:"output_hash"`
	FinishReason          string    `json:"finish_reason"`
	ErrorCode             string    `json:"error_code"`
	ErrorMessage          string    `json:"error_message"`
	RequestBody           string    `json:"request_body"`
	ResponseBody          string    `json:"response_body"`
}

type Summary struct {
	Requests        int64            `json:"requests"`
	Errors          int64            `json:"errors"`
	PromptTokens    int64            `json:"prompt_tokens"`
	OutputTokens    int64            `json:"output_tokens"`
	TotalTokens     int64            `json:"total_tokens"`
	CostCents       int64            `json:"cost_cents"`
	AvgLatencyMS    int64            `json:"avg_latency_ms"`
	AvgTTFTMS       int64            `json:"avg_ttft_ms"`
	AvgGenerationMS int64            `json:"avg_generation_ms"`
	ByProvider      map[string]int64 `json:"by_provider"`
	ByModel         map[string]int64 `json:"by_model"`
	WindowStartsAt  time.Time        `json:"window_starts_at"`
}

type DailyUsage struct {
	Date         string `json:"date"`
	Requests     int64  `json:"requests"`
	Errors       int64  `json:"errors"`
	PromptTokens int64  `json:"prompt_tokens"`
	OutputTokens int64  `json:"output_tokens"`
	CostCents    int64  `json:"cost_cents"`
}

type ErrorBreakdownItem struct {
	StatusCode int    `json:"status_code"`
	Label      string `json:"label"`
	Count      int64  `json:"count"`
}

type TokenMix struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	OutputTokens     int64 `json:"output_tokens"`
	CacheReadTokens  int64 `json:"cache_read_tokens"`
	CacheWriteTokens int64 `json:"cache_write_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type UsageDashboard struct {
	Daily    []DailyUsage `json:"daily"`
	TokenMix TokenMix     `json:"token_mix"`
	Recent   []RequestLog `json:"recent"`
}

type ErrorDashboard struct {
	Breakdown     []ErrorBreakdownItem `json:"breakdown"`
	ErrorRequests []RequestLog         `json:"error_requests"`
	SlowRequests  []RequestLog         `json:"slow_requests"`
}

type KeyStat struct {
	Provider      string  `json:"provider"`
	ProviderKeyID string  `json:"provider_key_id"`
	Requests      int64   `json:"requests"`
	Errors        int64   `json:"errors"`
	SuccessRate   float64 `json:"success_rate"`
	TotalTokens   int64   `json:"total_tokens"`
}

type RequestFilter struct {
	Provider      string
	ProviderKeyID string
	Model         string
	User          string
	Status        string
	Limit         int
}

type VirtualKey struct {
	ID                   int64      `json:"id"`
	KeyHash              string     `json:"key_hash"`
	APIKey               string     `json:"api_key"`
	User                 string     `json:"user"`
	AllowedModels        []string   `json:"allowed_models"`
	RPMLimit             int        `json:"rpm_limit"`
	TPMLimit             int        `json:"tpm_limit"`
	ConcurrencyLimit     int        `json:"concurrency_limit"`
	DailySpendLimitCents int64      `json:"daily_spend_limit_cents"`
	Enabled              bool       `json:"enabled"`
	State                string     `json:"state"`
	CreatedAt            time.Time  `json:"created_at"`
	UpdatedAt            time.Time  `json:"updated_at"`
	LastUsedAt           *time.Time `json:"last_used_at,omitempty"`
	RevokedAt            *time.Time `json:"revoked_at,omitempty"`
}

type CreateVirtualKeyRequest struct {
	User                 string   `json:"user"`
	AllowedModels        []string `json:"allowed_models"`
	RPMLimit             int      `json:"rpm_limit"`
	TPMLimit             int      `json:"tpm_limit"`
	ConcurrencyLimit     int      `json:"concurrency_limit"`
	DailySpendLimitCents int64    `json:"daily_spend_limit_cents"`
}

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	store := &Store{db: db}
	if err := store.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return store, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) migrate() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS request_logs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			trace_id TEXT NOT NULL,
			created_at TEXT NOT NULL,
			date_key TEXT NOT NULL,
			method TEXT NOT NULL,
			path TEXT NOT NULL,
			protocol TEXT NOT NULL,
			virtual_key_hash TEXT NOT NULL,
			model TEXT NOT NULL,
			provider TEXT NOT NULL,
			provider_key_id TEXT NOT NULL,
			upstream_model TEXT NOT NULL,
			stream INTEGER NOT NULL,
			has_image INTEGER NOT NULL DEFAULT 0,
			status_code INTEGER NOT NULL,
			latency_ms INTEGER NOT NULL,
			ttft_ms INTEGER NOT NULL DEFAULT 0,
			generation_ms INTEGER NOT NULL DEFAULT 0,
			output_tokens_per_second REAL NOT NULL DEFAULT 0,
			prompt_bucket TEXT NOT NULL DEFAULT '',
			slow_reason TEXT NOT NULL DEFAULT '',
			prompt_tokens INTEGER NOT NULL,
			output_tokens INTEGER NOT NULL,
			total_tokens INTEGER NOT NULL,
			cost_cents INTEGER NOT NULL,
			cache_read_tokens INTEGER NOT NULL DEFAULT 0,
			cache_write_tokens INTEGER NOT NULL DEFAULT 0,
			cache_hit_rate REAL NOT NULL DEFAULT 0,
			input_hash TEXT NOT NULL,
			output_hash TEXT NOT NULL,
			finish_reason TEXT NOT NULL,
			error_code TEXT NOT NULL,
			error_message TEXT NOT NULL,
			user TEXT NOT NULL DEFAULT '',
			api_key TEXT NOT NULL DEFAULT '',
			client_ip TEXT NOT NULL DEFAULT '',
			router_decision TEXT NOT NULL DEFAULT '',
			fallback_count INTEGER NOT NULL DEFAULT 0,
			request_body TEXT NOT NULL DEFAULT '',
			response_body TEXT NOT NULL DEFAULT '',
			revoked_at TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS virtual_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			key_hash TEXT NOT NULL UNIQUE,
			api_key TEXT NOT NULL,
			user TEXT NOT NULL,
			allowed_models TEXT NOT NULL,
			rpm_limit INTEGER NOT NULL,
			tpm_limit INTEGER NOT NULL,
			concurrency_limit INTEGER NOT NULL,
			daily_spend_limit_cents INTEGER NOT NULL,
			enabled INTEGER NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			last_used_at TEXT,
			revoked_at TEXT
		)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_date_key_key ON request_logs(date_key, virtual_key_hash)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_model ON request_logs(model)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_provider ON request_logs(provider)`,
		`CREATE INDEX IF NOT EXISTS idx_request_logs_provider_key ON request_logs(provider, provider_key_id)`,
		`CREATE INDEX IF NOT EXISTS idx_virtual_keys_enabled ON virtual_keys(enabled)`,
	}
	for _, stmt := range stmts {
		if _, err := s.db.Exec(stmt); err != nil {
			return err
		}
	}
	for _, stmt := range []string{
		`ALTER TABLE request_logs ADD COLUMN user TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN api_key TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN client_ip TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN router_decision TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN fallback_count INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN has_image INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN ttft_ms INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN generation_ms INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN output_tokens_per_second REAL NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN prompt_bucket TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN slow_reason TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN cache_read_tokens INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN cache_write_tokens INTEGER NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN cache_hit_rate REAL NOT NULL DEFAULT 0`,
		`ALTER TABLE request_logs ADD COLUMN request_body TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN response_body TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE request_logs ADD COLUMN revoked_at TEXT`,
		`ALTER TABLE virtual_keys ADD COLUMN revoked_at TEXT`,
	} {
		_, _ = s.db.Exec(stmt)
	}
	return nil
}

func (s *Store) Record(ctx context.Context, r RequestLog) error {
	if s == nil {
		return nil
	}
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	if r.TotalTokens == 0 {
		r.TotalTokens = r.PromptTokens + r.OutputTokens
	}
	_, err := s.db.ExecContext(ctx, `INSERT INTO request_logs (
		trace_id, created_at, date_key, method, path, protocol, virtual_key_hash, model, provider,
		provider_key_id, upstream_model, stream, status_code, latency_ms, prompt_tokens, output_tokens,
		ttft_ms, generation_ms, output_tokens_per_second, prompt_bucket, slow_reason,
		total_tokens, cost_cents, cache_read_tokens, cache_write_tokens, cache_hit_rate,
		input_hash, output_hash, finish_reason, error_code, error_message,
		user, api_key, client_ip, router_decision, fallback_count, has_image, request_body, response_body
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		r.TraceID, r.CreatedAt.Format(time.RFC3339Nano), r.CreatedAt.Format("2006-01-02"),
		r.Method, r.Path, r.Protocol, r.VirtualKeyHash, r.Model, r.Provider, r.ProviderKeyID,
		r.UpstreamModel, boolInt(r.Stream), r.StatusCode, r.LatencyMS, r.PromptTokens, r.OutputTokens,
		r.TTFTMS, r.GenerationMS, r.OutputTokensPerSecond, r.PromptBucket, r.SlowReason,
		r.TotalTokens, r.CostCents, r.CacheReadTokens, r.CacheWriteTokens, r.CacheHitRate,
		r.InputHash, r.OutputHash, r.FinishReason, r.ErrorCode, truncate(r.ErrorMessage, 2000),
		r.User, r.APIKey, r.ClientIP, r.RouterDecision, r.FallbackCount, boolInt(r.HasImage),
		truncate(r.RequestBody, 16000), truncate(r.ResponseBody, 16000))
	return err
}

func (s *Store) Summary(ctx context.Context, hours int) (Summary, error) {
	if hours <= 0 {
		hours = 24
	}
	start := time.Now().UTC().Add(-time.Duration(hours) * time.Hour)
	out := Summary{ByProvider: map[string]int64{}, ByModel: map[string]int64{}, WindowStartsAt: start}
	var avgLatency, avgTTFT, avgGeneration float64
	row := s.db.QueryRowContext(ctx, `SELECT COUNT(*), SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END),
		COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(total_tokens),0),
		COALESCE(SUM(cost_cents),0), COALESCE(AVG(latency_ms),0), COALESCE(AVG(ttft_ms),0),
		COALESCE(AVG(generation_ms),0)
		FROM request_logs WHERE created_at >= ?`, start.Format(time.RFC3339Nano))
	if err := row.Scan(&out.Requests, &out.Errors, &out.PromptTokens, &out.OutputTokens, &out.TotalTokens, &out.CostCents, &avgLatency, &avgTTFT, &avgGeneration); err != nil {
		return out, err
	}
	out.AvgLatencyMS = int64(avgLatency)
	out.AvgTTFTMS = int64(avgTTFT)
	out.AvgGenerationMS = int64(avgGeneration)
	rows, err := s.db.QueryContext(ctx, `SELECT provider, COUNT(*) FROM request_logs WHERE created_at >= ? GROUP BY provider`, start.Format(time.RFC3339Nano))
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var k string
		var v int64
		if err := rows.Scan(&k, &v); err == nil {
			out.ByProvider[defaultLabel(k)] = v
		}
	}
	rows, err = s.db.QueryContext(ctx, `SELECT model, COUNT(*) FROM request_logs WHERE created_at >= ? GROUP BY model`, start.Format(time.RFC3339Nano))
	if err != nil {
		return out, err
	}
	defer rows.Close()
	for rows.Next() {
		var k string
		var v int64
		if err := rows.Scan(&k, &v); err == nil {
			out.ByModel[defaultLabel(k)] = v
		}
	}
	return out, nil
}

func (s *Store) UsageDashboard(ctx context.Context, days int, recentLimit int) (UsageDashboard, error) {
	if days <= 0 {
		days = 14
	}
	if recentLimit <= 0 {
		recentLimit = 20
	}
	daily, err := s.Daily(ctx, days)
	if err != nil {
		return UsageDashboard{}, err
	}
	start := time.Now().UTC().AddDate(0, 0, -days+1).Format("2006-01-02")
	var mix TokenMix
	row := s.db.QueryRowContext(ctx, `SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(output_tokens),0),
		COALESCE(SUM(cache_read_tokens),0), COALESCE(SUM(cache_write_tokens),0), COALESCE(SUM(total_tokens),0)
		FROM request_logs WHERE date_key >= ?`, start)
	if err := row.Scan(&mix.PromptTokens, &mix.OutputTokens, &mix.CacheReadTokens, &mix.CacheWriteTokens, &mix.TotalTokens); err != nil {
		return UsageDashboard{}, err
	}
	recent, err := s.Requests(ctx, RequestFilter{Limit: recentLimit})
	if err != nil {
		return UsageDashboard{}, err
	}
	return UsageDashboard{Daily: daily, TokenMix: mix, Recent: recent}, nil
}

func (s *Store) Daily(ctx context.Context, days int) ([]DailyUsage, error) {
	if days <= 0 {
		days = 14
	}
	start := time.Now().UTC().AddDate(0, 0, -days+1).Format("2006-01-02")
	rows, err := s.db.QueryContext(ctx, `SELECT date_key, COUNT(*), SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END),
		COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(cost_cents),0)
		FROM request_logs WHERE date_key >= ? GROUP BY date_key ORDER BY date_key DESC`, start)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []DailyUsage
	for rows.Next() {
		var item DailyUsage
		if err := rows.Scan(&item.Date, &item.Requests, &item.Errors, &item.PromptTokens, &item.OutputTokens, &item.CostCents); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *Store) ErrorBreakdown(ctx context.Context, hours int) ([]ErrorBreakdownItem, error) {
	if hours <= 0 {
		hours = 24
	}
	start := time.Now().UTC().Add(-time.Duration(hours) * time.Hour)
	counts := map[int]int64{}
	rows, err := s.db.QueryContext(ctx, `SELECT status_code, COUNT(*) FROM request_logs WHERE created_at >= ? AND status_code >= 400 GROUP BY status_code`, start.Format(time.RFC3339Nano))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var status int
		var count int64
		if err := rows.Scan(&status, &count); err != nil {
			return nil, err
		}
		counts[status] = count
	}
	order := []int{401, 403, 404, 429, 500, 502}
	out := make([]ErrorBreakdownItem, 0, len(order))
	for _, status := range order {
		out = append(out, ErrorBreakdownItem{StatusCode: status, Label: errorLabel(status), Count: counts[status]})
		delete(counts, status)
	}
	for status, count := range counts {
		out = append(out, ErrorBreakdownItem{StatusCode: status, Label: errorLabel(status), Count: count})
	}
	return out, rows.Err()
}

func (s *Store) ErrorDashboard(ctx context.Context, hours int, limit int) (ErrorDashboard, error) {
	if hours <= 0 {
		hours = 24
	}
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	breakdown, err := s.ErrorBreakdown(ctx, hours)
	if err != nil {
		return ErrorDashboard{}, err
	}
	errorRequests, err := s.RequestsSince(ctx, RequestFilter{Status: "error", Limit: limit}, time.Now().UTC().Add(-time.Duration(hours)*time.Hour))
	if err != nil {
		return ErrorDashboard{}, err
	}
	slowRequests, err := s.SlowRequestsSince(ctx, time.Now().UTC().Add(-time.Duration(hours)*time.Hour), limit)
	if err != nil {
		return ErrorDashboard{}, err
	}
	return ErrorDashboard{Breakdown: breakdown, ErrorRequests: errorRequests, SlowRequests: slowRequests}, nil
}

func (s *Store) KeyStats(ctx context.Context, hours int) ([]KeyStat, error) {
	if hours <= 0 {
		hours = 24
	}
	start := time.Now().UTC().Add(-time.Duration(hours) * time.Hour)
	rows, err := s.db.QueryContext(ctx, `SELECT provider, provider_key_id, COUNT(*),
		SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), COALESCE(SUM(total_tokens),0)
		FROM request_logs WHERE created_at >= ? AND provider != '' AND provider_key_id != ''
		GROUP BY provider, provider_key_id`, start.Format(time.RFC3339Nano))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []KeyStat
	for rows.Next() {
		var item KeyStat
		if err := rows.Scan(&item.Provider, &item.ProviderKeyID, &item.Requests, &item.Errors, &item.TotalTokens); err != nil {
			return nil, err
		}
		if item.Requests > 0 {
			item.SuccessRate = float64(item.Requests-item.Errors) / float64(item.Requests)
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *Store) Requests(ctx context.Context, f RequestFilter) ([]RequestLog, error) {
	return s.RequestsSince(ctx, f, time.Time{})
}

func (s *Store) RequestByTraceID(ctx context.Context, traceID string) (*RequestLog, error) {
	traceID = strings.TrimSpace(traceID)
	if traceID == "" {
		return nil, sql.ErrNoRows
	}
	rows, err := s.scanRequestLogs(ctx, `SELECT trace_id, created_at, method, path, protocol, virtual_key_hash, user, api_key, client_ip, model,
		provider, provider_key_id, upstream_model, router_decision, fallback_count, stream, status_code,
		latency_ms, ttft_ms, generation_ms, output_tokens_per_second, prompt_bucket, slow_reason,
		prompt_tokens, output_tokens, total_tokens, cost_cents, cache_read_tokens,
		cache_write_tokens, cache_hit_rate, input_hash, output_hash,
		finish_reason, error_code, error_message, has_image, request_body, response_body FROM request_logs WHERE trace_id = ? LIMIT 1`, traceID)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, sql.ErrNoRows
	}
	return &rows[0], nil
}

func (s *Store) RequestsSince(ctx context.Context, f RequestFilter, start time.Time) ([]RequestLog, error) {
	if f.Limit <= 0 || f.Limit > 500 {
		f.Limit = 100
	}
	where := []string{"1=1"}
	args := []interface{}{}
	add := func(column, value string) {
		if strings.TrimSpace(value) != "" {
			where = append(where, column+" = ?")
			args = append(args, value)
		}
	}
	add("provider", f.Provider)
	add("provider_key_id", f.ProviderKeyID)
	add("model", f.Model)
	add("user", f.User)
	if !start.IsZero() {
		where = append(where, "created_at >= ?")
		args = append(args, start.Format(time.RFC3339Nano))
	}
	switch f.Status {
	case "success":
		where = append(where, "status_code < 400")
	case "error":
		where = append(where, "status_code >= 400")
	}
	args = append(args, f.Limit)
	query := `SELECT trace_id, created_at, method, path, protocol, virtual_key_hash, user, api_key, client_ip, model,
		provider, provider_key_id, upstream_model, router_decision, fallback_count, stream, status_code,
		latency_ms, ttft_ms, generation_ms, output_tokens_per_second, prompt_bucket, slow_reason,
		prompt_tokens, output_tokens, total_tokens, cost_cents, cache_read_tokens,
		cache_write_tokens, cache_hit_rate, input_hash, output_hash,
		finish_reason, error_code, error_message, has_image, request_body, response_body FROM request_logs WHERE ` + strings.Join(where, " AND ") + ` ORDER BY created_at DESC LIMIT ?`
	return s.scanRequestLogs(ctx, query, args...)
}

func (s *Store) scanRequestLogs(ctx context.Context, query string, args ...interface{}) ([]RequestLog, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []RequestLog
	for rows.Next() {
		var item RequestLog
		var created string
		var stream int
		var hasImage int
		if err := rows.Scan(&item.TraceID, &created, &item.Method, &item.Path, &item.Protocol, &item.VirtualKeyHash,
			&item.User, &item.APIKey, &item.ClientIP, &item.Model, &item.Provider, &item.ProviderKeyID, &item.UpstreamModel,
			&item.RouterDecision, &item.FallbackCount, &stream, &item.StatusCode, &item.LatencyMS, &item.TTFTMS,
			&item.GenerationMS, &item.OutputTokensPerSecond, &item.PromptBucket, &item.SlowReason, &item.PromptTokens,
			&item.OutputTokens, &item.TotalTokens, &item.CostCents, &item.CacheReadTokens,
			&item.CacheWriteTokens, &item.CacheHitRate, &item.InputHash, &item.OutputHash,
			&item.FinishReason, &item.ErrorCode, &item.ErrorMessage, &hasImage, &item.RequestBody, &item.ResponseBody); err != nil {
			return nil, err
		}
		item.Stream = stream == 1
		item.HasImage = hasImage == 1
		item.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *Store) SlowRequestsSince(ctx context.Context, start time.Time, limit int) ([]RequestLog, error) {
	if limit <= 0 || limit > 500 {
		limit = 100
	}
	where := "slow_reason != ''"
	args := []interface{}{}
	if !start.IsZero() {
		where = "created_at >= ? AND (" + where + ")"
		args = append(args, start.Format(time.RFC3339Nano))
	}
	args = append(args, limit)
	query := `SELECT trace_id, created_at, method, path, protocol, virtual_key_hash, user, api_key, client_ip, model,
		provider, provider_key_id, upstream_model, router_decision, fallback_count, stream, status_code,
		latency_ms, ttft_ms, generation_ms, output_tokens_per_second, prompt_bucket, slow_reason,
		prompt_tokens, output_tokens, total_tokens, cost_cents, cache_read_tokens,
		cache_write_tokens, cache_hit_rate, input_hash, output_hash,
		finish_reason, error_code, error_message, has_image, request_body, response_body FROM request_logs WHERE ` + where + ` ORDER BY latency_ms DESC, created_at DESC LIMIT ?`
	return s.scanRequestLogs(ctx, query, args...)
}

func (s *Store) SeedConfigVirtualKeys(ctx context.Context, keys []config.VirtualKey) error {
	for _, key := range keys {
		if strings.TrimSpace(key.Key) == "" {
			continue
		}
		now := time.Now().UTC().Format(time.RFC3339Nano)
		models, _ := json.Marshal(key.AllowedModels)
		_, err := s.db.ExecContext(ctx, `INSERT INTO virtual_keys
			(key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit, concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
			ON CONFLICT(key_hash) DO UPDATE SET user=excluded.user, allowed_models=excluded.allowed_models,
			rpm_limit=excluded.rpm_limit, tpm_limit=excluded.tpm_limit, concurrency_limit=excluded.concurrency_limit,
			daily_spend_limit_cents=excluded.daily_spend_limit_cents, api_key=excluded.api_key, updated_at=excluded.updated_at`,
			auth.HashKey(key.Key), auth.MaskKey(key.Key), defaultUser(key.User), string(models), key.RPMLimit,
			key.TPMLimit, key.ConcurrencyLimit, key.DailySpendLimitCents, now, now)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) ListVirtualKeys(ctx context.Context) ([]VirtualKey, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT id, key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit,
		concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at, last_used_at, revoked_at
		FROM virtual_keys ORDER BY id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []VirtualKey
	for rows.Next() {
		item, err := scanVirtualKey(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (s *Store) ActiveVirtualKeys(ctx context.Context) ([]auth.VirtualKeyInfo, error) {
	rows, err := s.db.QueryContext(ctx, `SELECT key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit,
		concurrency_limit, daily_spend_limit_cents FROM virtual_keys WHERE enabled = 1 AND revoked_at IS NULL`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []auth.VirtualKeyInfo
	for rows.Next() {
		var info auth.VirtualKeyInfo
		var modelsJSON string
		if err := rows.Scan(&info.KeyHash, &info.APIKey, &info.User, &modelsJSON, &info.RPMLimit,
			&info.TPMLimit, &info.ConcurrencyLimit, &info.DailySpendLimitCents); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(modelsJSON), &info.AllowedModels)
		out = append(out, info)
	}
	return out, rows.Err()
}

func (s *Store) TouchVirtualKey(ctx context.Context, keyHash string) {
	if s == nil || keyHash == "" {
		return
	}
	_, _ = s.db.ExecContext(ctx, `UPDATE virtual_keys SET last_used_at = ? WHERE key_hash = ?`, time.Now().UTC().Format(time.RFC3339Nano), keyHash)
}

func (s *Store) CreateVirtualKey(ctx context.Context, req CreateVirtualKeyRequest) (VirtualKey, string, error) {
	apiKey, err := generateAPIKey()
	if err != nil {
		return VirtualKey{}, "", err
	}
	if req.User == "" {
		req.User = "default"
	}
	if req.RPMLimit == 0 {
		req.RPMLimit = 100
	}
	if req.TPMLimit == 0 {
		req.TPMLimit = 200000
	}
	if req.ConcurrencyLimit == 0 {
		req.ConcurrencyLimit = 10
	}
	modelsJSON, _ := json.Marshal(req.AllowedModels)
	now := time.Now().UTC().Format(time.RFC3339Nano)
	res, err := s.db.ExecContext(ctx, `INSERT INTO virtual_keys
		(key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit, concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)`,
		auth.HashKey(apiKey), auth.MaskKey(apiKey), req.User, string(modelsJSON), req.RPMLimit, req.TPMLimit,
		req.ConcurrencyLimit, req.DailySpendLimitCents, now, now)
	if err != nil {
		return VirtualKey{}, "", err
	}
	id, _ := res.LastInsertId()
	keys, err := s.ListVirtualKeys(ctx)
	if err != nil {
		return VirtualKey{}, "", err
	}
	for _, key := range keys {
		if key.ID == id {
			return key, apiKey, nil
		}
	}
	return VirtualKey{}, "", fmt.Errorf("created key not found")
}

func (s *Store) SetVirtualKeyState(ctx context.Context, id int64, action string) (*VirtualKey, error) {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	switch action {
	case "enable":
		_, err := s.db.ExecContext(ctx, `UPDATE virtual_keys SET enabled=1, updated_at=? WHERE id=? AND revoked_at IS NULL`, now, id)
		if err != nil {
			return nil, err
		}
	case "disable":
		_, err := s.db.ExecContext(ctx, `UPDATE virtual_keys SET enabled=0, updated_at=? WHERE id=? AND revoked_at IS NULL`, now, id)
		if err != nil {
			return nil, err
		}
	case "revoke":
		_, err := s.db.ExecContext(ctx, `UPDATE virtual_keys SET enabled=0, revoked_at=?, updated_at=? WHERE id=?`, now, now, id)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown action %s", action)
	}
	row := s.db.QueryRowContext(ctx, `SELECT id, key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit,
		concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at, last_used_at, revoked_at
		FROM virtual_keys WHERE id=?`, id)
	key, err := scanVirtualKey(row)
	if err != nil {
		return nil, err
	}
	return &key, nil
}

type scanner interface {
	Scan(dest ...interface{}) error
}

func scanVirtualKey(row scanner) (VirtualKey, error) {
	var item VirtualKey
	var modelsJSON, created, updated string
	var enabled int
	var lastUsed, revoked sql.NullString
	if err := row.Scan(&item.ID, &item.KeyHash, &item.APIKey, &item.User, &modelsJSON, &item.RPMLimit,
		&item.TPMLimit, &item.ConcurrencyLimit, &item.DailySpendLimitCents, &enabled, &created, &updated,
		&lastUsed, &revoked); err != nil {
		return item, err
	}
	_ = json.Unmarshal([]byte(modelsJSON), &item.AllowedModels)
	item.Enabled = enabled == 1
	item.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
	item.UpdatedAt, _ = time.Parse(time.RFC3339Nano, updated)
	if lastUsed.Valid {
		t, _ := time.Parse(time.RFC3339Nano, lastUsed.String)
		item.LastUsedAt = &t
	}
	if revoked.Valid {
		t, _ := time.Parse(time.RFC3339Nano, revoked.String)
		item.RevokedAt = &t
	}
	switch {
	case item.RevokedAt != nil:
		item.State = "revoked"
	case item.Enabled:
		item.State = "active"
	default:
		item.State = "disabled"
	}
	return item, nil
}

func generateAPIKey() (string, error) {
	b := make([]byte, 24)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return "sk-gw-" + base64.RawURLEncoding.EncodeToString(b), nil
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func defaultLabel(s string) string {
	if strings.TrimSpace(s) == "" {
		return "unknown"
	}
	return s
}

func defaultUser(s string) string {
	if strings.TrimSpace(s) == "" {
		return "default"
	}
	return s
}

func errorLabel(status int) string {
	switch status {
	case 401:
		return "Unauthorized"
	case 403:
		return "Forbidden"
	case 404:
		return "Model Not Found"
	case 429:
		return "Rate Limited"
	case 500:
		return "Provider Error"
	case 502:
		return "Gateway Error"
	default:
		if text := http.StatusText(status); text != "" {
			return text
		}
		return "Error"
	}
}
