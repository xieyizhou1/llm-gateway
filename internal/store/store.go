// Package store persists request usage and exposes read models for the dashboard.
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type Store struct{ db *sql.DB }

type RequestRecord struct {
	TraceID        string    `json:"trace_id"`
	CreatedAt      time.Time `json:"created_at"`
	Method         string    `json:"method"`
	Path           string    `json:"path"`
	Protocol       string    `json:"protocol"`
	VirtualKeyHash string    `json:"virtual_key_hash"`
	User           string    `json:"user"`
	APIKey         string    `json:"api_key"`
	Model          string    `json:"model"`
	Provider       string    `json:"provider"`
	ProviderKeyID  string    `json:"provider_key_id"`
	UpstreamModel  string    `json:"upstream_model"`
	RouterDecision string    `json:"router_decision"`
	FallbackCount  int       `json:"fallback_count"`
	Stream         bool      `json:"stream"`
	StatusCode     int       `json:"status_code"`
	LatencyMs      int64     `json:"latency_ms"`
	PromptTokens   int       `json:"prompt_tokens"`
	OutputTokens   int       `json:"output_tokens"`
	TotalTokens    int       `json:"total_tokens"`
	CostCents      int64     `json:"cost_cents"`
	InputHash      string    `json:"input_hash"`
	OutputHash     string    `json:"output_hash"`
	FinishReason   string    `json:"finish_reason"`
	ErrorCode      string    `json:"error_code"`
	ErrorMessage   string    `json:"error_message"`
}

type Summary struct {
	Requests       int64            `json:"requests"`
	Errors         int64            `json:"errors"`
	PromptTokens   int64            `json:"prompt_tokens"`
	OutputTokens   int64            `json:"output_tokens"`
	TotalTokens    int64            `json:"total_tokens"`
	CostCents      int64            `json:"cost_cents"`
	AvgLatencyMs   int64            `json:"avg_latency_ms"`
	ByProvider     map[string]int64 `json:"by_provider"`
	ByModel        map[string]int64 `json:"by_model"`
	WindowStartsAt time.Time        `json:"window_starts_at"`
}

type RequestQuery struct {
	Limit         int
	Provider      string
	ProviderKeyID string
	Model         string
	User          string
	APIKey        string
	Status        string
	From          time.Time
	To            time.Time
}

type KeyStats struct {
	Provider      string  `json:"provider"`
	ProviderKeyID string  `json:"provider_key_id"`
	Requests      int64   `json:"requests"`
	Successes     int64   `json:"successes"`
	Errors        int64   `json:"errors"`
	SuccessRate   float64 `json:"success_rate"`
	PromptTokens  int64   `json:"prompt_tokens"`
	OutputTokens  int64   `json:"output_tokens"`
	TotalTokens   int64   `json:"total_tokens"`
	CostCents     int64   `json:"cost_cents"`
	AvgLatencyMs  int64   `json:"avg_latency_ms"`
}

type DailyUsage struct {
	Date         string `json:"date"`
	Requests     int64  `json:"requests"`
	Errors       int64  `json:"errors"`
	PromptTokens int64  `json:"prompt_tokens"`
	OutputTokens int64  `json:"output_tokens"`
	TotalTokens  int64  `json:"total_tokens"`
	CostCents    int64  `json:"cost_cents"`
}

type VirtualKeyRecord struct {
	ID                   int64      `json:"id"`
	KeyHash              string     `json:"-"`
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

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	s := &Store{db: db}
	if err := s.migrate(context.Background()); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *Store) migrate(ctx context.Context) error {
	_, err := s.db.ExecContext(ctx, `
CREATE TABLE IF NOT EXISTS request_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  trace_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  date_key TEXT NOT NULL,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  protocol TEXT NOT NULL,
  virtual_key_hash TEXT NOT NULL,
  user TEXT NOT NULL DEFAULT '',
  api_key TEXT NOT NULL DEFAULT '',
  model TEXT NOT NULL,
  provider TEXT NOT NULL,
  provider_key_id TEXT NOT NULL,
  upstream_model TEXT NOT NULL,
  router_decision TEXT NOT NULL DEFAULT '',
  fallback_count INTEGER NOT NULL DEFAULT 0,
  stream INTEGER NOT NULL,
  status_code INTEGER NOT NULL,
  latency_ms INTEGER NOT NULL,
  prompt_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  total_tokens INTEGER NOT NULL,
  cost_cents INTEGER NOT NULL,
  input_hash TEXT NOT NULL,
  output_hash TEXT NOT NULL,
  finish_reason TEXT NOT NULL,
  error_code TEXT NOT NULL,
  error_message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_date_key_key ON request_logs(date_key, virtual_key_hash);
CREATE INDEX IF NOT EXISTS idx_request_logs_provider ON request_logs(provider);
CREATE INDEX IF NOT EXISTS idx_request_logs_provider_key ON request_logs(provider, provider_key_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_model ON request_logs(model);
CREATE TABLE IF NOT EXISTS virtual_keys (
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
);
CREATE INDEX IF NOT EXISTS idx_virtual_keys_enabled ON virtual_keys(enabled);
`)
	if err != nil {
		return err
	}
	columns := map[string]string{
		"user":            "TEXT NOT NULL DEFAULT ''",
		"api_key":         "TEXT NOT NULL DEFAULT ''",
		"router_decision": "TEXT NOT NULL DEFAULT ''",
		"fallback_count":  "INTEGER NOT NULL DEFAULT 0",
		"revoked_at":      "TEXT",
	}
	for name, definition := range columns {
		if err := s.ensureColumn(ctx, "request_logs", name, definition); err != nil {
			return err
		}
	}
	if err := s.ensureColumn(ctx, "virtual_keys", "revoked_at", "TEXT"); err != nil {
		return err
	}
	return nil
}

func (s *Store) ensureColumn(ctx context.Context, table, name, definition string) error {
	rows, err := s.db.QueryContext(ctx, "PRAGMA table_info("+table+")")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var colName, colType string
		var notNull int
		var defaultValue interface{}
		var pk int
		if err := rows.Scan(&cid, &colName, &colType, &notNull, &defaultValue, &pk); err != nil {
			return err
		}
		if colName == name {
			return nil
		}
	}
	if err := rows.Err(); err != nil {
		return err
	}
	_, err = s.db.ExecContext(ctx, "ALTER TABLE "+table+" ADD COLUMN "+name+" "+definition)
	return err
}

func (s *Store) InsertRequest(ctx context.Context, r RequestRecord) error {
	if s == nil || s.db == nil {
		return nil
	}
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	stream := 0
	if r.Stream {
		stream = 1
	}
	_, err := s.db.ExecContext(ctx, `
INSERT INTO request_logs (
  trace_id, created_at, date_key, method, path, protocol, virtual_key_hash, user, api_key,
  model, provider, provider_key_id, upstream_model, router_decision, fallback_count, stream, status_code, latency_ms,
  prompt_tokens, output_tokens, total_tokens, cost_cents, input_hash, output_hash,
  finish_reason, error_code, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		r.TraceID, r.CreatedAt.Format(time.RFC3339Nano), r.CreatedAt.Format("2006-01-02"), r.Method, r.Path, r.Protocol, r.VirtualKeyHash, r.User, r.APIKey,
		r.Model, r.Provider, r.ProviderKeyID, r.UpstreamModel, r.RouterDecision, r.FallbackCount, stream, r.StatusCode, r.LatencyMs,
		r.PromptTokens, r.OutputTokens, r.TotalTokens, r.CostCents, r.InputHash, r.OutputHash, r.FinishReason, r.ErrorCode, r.ErrorMessage)
	return err
}

func (s *Store) DailySpendCents(ctx context.Context, virtualKeyHash string, day time.Time) (int64, error) {
	if s == nil || s.db == nil || virtualKeyHash == "" {
		return 0, nil
	}
	var cents sql.NullInt64
	err := s.db.QueryRowContext(ctx, `SELECT COALESCE(SUM(cost_cents), 0) FROM request_logs WHERE date_key = ? AND virtual_key_hash = ?`, day.UTC().Format("2006-01-02"), virtualKeyHash).Scan(&cents)
	if err != nil {
		return 0, err
	}
	return cents.Int64, nil
}

func (s *Store) Summary(ctx context.Context, since time.Time) (Summary, error) {
	out := Summary{ByProvider: map[string]int64{}, ByModel: map[string]int64{}, WindowStartsAt: since.UTC()}
	if s == nil || s.db == nil {
		return out, nil
	}
	row := s.db.QueryRowContext(ctx, `
SELECT COUNT(*), COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0),
       COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(total_tokens), 0),
       COALESCE(SUM(cost_cents), 0), COALESCE(AVG(latency_ms), 0)
FROM request_logs WHERE created_at >= ?`, since.UTC().Format(time.RFC3339Nano))
	var avg float64
	if err := row.Scan(&out.Requests, &out.Errors, &out.PromptTokens, &out.OutputTokens, &out.TotalTokens, &out.CostCents, &avg); err != nil {
		return out, err
	}
	out.AvgLatencyMs = int64(avg + 0.5)
	if err := scanCounts(ctx, s.db, `SELECT provider, COUNT(*) FROM request_logs WHERE created_at >= ? GROUP BY provider`, since, out.ByProvider); err != nil {
		return out, err
	}
	if err := scanCounts(ctx, s.db, `SELECT model, COUNT(*) FROM request_logs WHERE created_at >= ? GROUP BY model`, since, out.ByModel); err != nil {
		return out, err
	}
	return out, nil
}

func scanCounts(ctx context.Context, db *sql.DB, query string, since time.Time, dest map[string]int64) error {
	rows, err := db.QueryContext(ctx, query, since.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var key string
		var value int64
		if err := rows.Scan(&key, &value); err != nil {
			return err
		}
		if key == "" {
			key = "unknown"
		}
		dest[key] = value
	}
	return rows.Err()
}

func (s *Store) RecentRequests(ctx context.Context, limit int) ([]RequestRecord, error) {
	return s.QueryRequests(ctx, RequestQuery{Limit: limit})
}

func (s *Store) QueryRequests(ctx context.Context, q RequestQuery) ([]RequestRecord, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	if q.Limit <= 0 || q.Limit > 1000 {
		q.Limit = 100
	}
	where, args := buildWhere(q)
	args = append(args, q.Limit)
	rows, err := s.db.QueryContext(ctx, `
SELECT trace_id, created_at, method, path, protocol, virtual_key_hash, user, api_key, model, provider, provider_key_id,
       upstream_model, router_decision, fallback_count, stream, status_code, latency_ms, prompt_tokens, output_tokens, total_tokens,
       cost_cents, input_hash, output_hash, finish_reason, error_code, error_message
FROM request_logs `+where+` ORDER BY id DESC LIMIT ?`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := make([]RequestRecord, 0, q.Limit)
	for rows.Next() {
		var r RequestRecord
		var created string
		var stream int
		if err := rows.Scan(&r.TraceID, &created, &r.Method, &r.Path, &r.Protocol, &r.VirtualKeyHash, &r.User, &r.APIKey, &r.Model, &r.Provider, &r.ProviderKeyID,
			&r.UpstreamModel, &r.RouterDecision, &r.FallbackCount, &stream, &r.StatusCode, &r.LatencyMs, &r.PromptTokens, &r.OutputTokens, &r.TotalTokens,
			&r.CostCents, &r.InputHash, &r.OutputHash, &r.FinishReason, &r.ErrorCode, &r.ErrorMessage); err != nil {
			return nil, err
		}
		r.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
		r.Stream = stream == 1
		items = append(items, r)
	}
	return items, rows.Err()
}

func buildWhere(q RequestQuery) (string, []interface{}) {
	parts := make([]string, 0)
	args := make([]interface{}, 0)
	add := func(cond string, value interface{}) { parts = append(parts, cond); args = append(args, value) }
	if q.Provider != "" {
		add("provider = ?", q.Provider)
	}
	if q.ProviderKeyID != "" {
		add("provider_key_id = ?", q.ProviderKeyID)
	}
	if q.Model != "" {
		add("model = ?", q.Model)
	}
	if q.User != "" {
		add("user = ?", q.User)
	}
	if q.APIKey != "" {
		add("api_key = ?", q.APIKey)
	}
	if q.Status == "error" {
		parts = append(parts, "status_code >= 400")
	} else if q.Status == "success" {
		parts = append(parts, "status_code < 400")
	}
	if !q.From.IsZero() {
		add("created_at >= ?", q.From.UTC().Format(time.RFC3339Nano))
	}
	if !q.To.IsZero() {
		add("created_at <= ?", q.To.UTC().Format(time.RFC3339Nano))
	}
	if len(parts) == 0 {
		return "", args
	}
	return "WHERE " + strings.Join(parts, " AND "), args
}

func (s *Store) KeyStats(ctx context.Context, since time.Time) ([]KeyStats, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `
SELECT provider, provider_key_id, COUNT(*),
       COALESCE(SUM(CASE WHEN status_code < 400 THEN 1 ELSE 0 END), 0),
       COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END), 0),
       COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(total_tokens), 0),
       COALESCE(SUM(cost_cents), 0), COALESCE(AVG(latency_ms), 0)
FROM request_logs WHERE created_at >= ? AND provider != '' AND provider_key_id != ''
GROUP BY provider, provider_key_id ORDER BY provider, provider_key_id`, since.UTC().Format(time.RFC3339Nano))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]KeyStats, 0)
	for rows.Next() {
		var st KeyStats
		var avg float64
		if err := rows.Scan(&st.Provider, &st.ProviderKeyID, &st.Requests, &st.Successes, &st.Errors, &st.PromptTokens, &st.OutputTokens, &st.TotalTokens, &st.CostCents, &avg); err != nil {
			return nil, err
		}
		if st.Requests > 0 {
			st.SuccessRate = float64(st.Successes) / float64(st.Requests)
		}
		st.AvgLatencyMs = int64(avg + 0.5)
		out = append(out, st)
	}
	return out, rows.Err()
}

func (s *Store) DailyUsage(ctx context.Context, days int) ([]DailyUsage, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	if days <= 0 || days > 180 {
		days = 14
	}
	since := time.Now().UTC().AddDate(0, 0, -days+1).Format("2006-01-02")
	rows, err := s.db.QueryContext(ctx, `
SELECT date_key, COUNT(*), COALESCE(SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END),0),
       COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(output_tokens),0), COALESCE(SUM(total_tokens),0), COALESCE(SUM(cost_cents),0)
FROM request_logs WHERE date_key >= ? GROUP BY date_key ORDER BY date_key DESC`, since)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]DailyUsage, 0)
	for rows.Next() {
		var d DailyUsage
		if err := rows.Scan(&d.Date, &d.Requests, &d.Errors, &d.PromptTokens, &d.OutputTokens, &d.TotalTokens, &d.CostCents); err != nil {
			return nil, err
		}
		out = append(out, d)
	}
	return out, rows.Err()
}

func ParseDateTime(value string) (time.Time, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, nil
	}
	layouts := []string{time.RFC3339, "2006-01-02", "2006-01-02 15:04:05"}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, value); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid time %q", value)
}

func (s *Store) AddVirtualKey(ctx context.Context, rec VirtualKeyRecord) (VirtualKeyRecord, error) {
	if s == nil || s.db == nil {
		return rec, nil
	}
	now := time.Now().UTC()
	models, _ := json.Marshal(rec.AllowedModels)
	if rec.CreatedAt.IsZero() {
		rec.CreatedAt = now
	}
	if rec.UpdatedAt.IsZero() {
		rec.UpdatedAt = now
	}
	enabled := 0
	if rec.Enabled {
		enabled = 1
	}
	res, err := s.db.ExecContext(ctx, `INSERT INTO virtual_keys (key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit, concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, rec.KeyHash, rec.APIKey, rec.User, string(models), rec.RPMLimit, rec.TPMLimit, rec.ConcurrencyLimit, rec.DailySpendLimitCents, enabled, rec.CreatedAt.Format(time.RFC3339Nano), rec.UpdatedAt.Format(time.RFC3339Nano))
	if err != nil {
		return rec, err
	}
	rec.ID, _ = res.LastInsertId()
	rec.State = virtualKeyState(rec.Enabled, rec.RevokedAt)
	return rec, nil
}

func (s *Store) ListVirtualKeys(ctx context.Context) ([]VirtualKeyRecord, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	rows, err := s.db.QueryContext(ctx, `SELECT id, key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit, concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at, last_used_at, revoked_at FROM virtual_keys ORDER BY id DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]VirtualKeyRecord, 0)
	for rows.Next() {
		rec, err := scanVirtualKey(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, rows.Err()
}

func (s *Store) GetVirtualKeyByHash(ctx context.Context, hash string) (VirtualKeyRecord, bool, error) {
	if s == nil || s.db == nil || hash == "" {
		return VirtualKeyRecord{}, false, nil
	}
	row := s.db.QueryRowContext(ctx, `SELECT id, key_hash, api_key, user, allowed_models, rpm_limit, tpm_limit, concurrency_limit, daily_spend_limit_cents, enabled, created_at, updated_at, last_used_at, revoked_at FROM virtual_keys WHERE key_hash = ? AND enabled = 1 AND revoked_at IS NULL`, hash)
	rec, err := scanVirtualKey(row)
	if err == sql.ErrNoRows {
		return VirtualKeyRecord{}, false, nil
	}
	if err != nil {
		return VirtualKeyRecord{}, false, err
	}
	_, _ = s.db.ExecContext(ctx, `UPDATE virtual_keys SET last_used_at = ? WHERE id = ?`, time.Now().UTC().Format(time.RFC3339Nano), rec.ID)
	return rec, true, nil
}

func (s *Store) SetVirtualKeyEnabled(ctx context.Context, id int64, enabled bool) error {
	if s == nil || s.db == nil {
		return nil
	}
	value := 0
	if enabled {
		value = 1
	}
	res, err := s.db.ExecContext(ctx, `UPDATE virtual_keys SET enabled = ?, updated_at = ? WHERE id = ? AND revoked_at IS NULL`, value, time.Now().UTC().Format(time.RFC3339Nano), id)
	if err != nil {
		return err
	}
	if n, err := res.RowsAffected(); err != nil {
		return err
	} else if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *Store) SetVirtualKeyRevoked(ctx context.Context, id int64) error {
	if s == nil || s.db == nil {
		return nil
	}
	now := time.Now().UTC().Format(time.RFC3339Nano)
	res, err := s.db.ExecContext(ctx, `UPDATE virtual_keys SET enabled = 0, revoked_at = ?, updated_at = ? WHERE id = ? AND revoked_at IS NULL`, now, now, id)
	if err != nil {
		return err
	}
	if n, err := res.RowsAffected(); err != nil {
		return err
	} else if n == 0 {
		return sql.ErrNoRows
	}
	return nil
}

type virtualKeyScanner interface {
	Scan(dest ...interface{}) error
}

func scanVirtualKey(row virtualKeyScanner) (VirtualKeyRecord, error) {
	var rec VirtualKeyRecord
	var modelsRaw, created, updated string
	var enabled int
	var last, revoked sql.NullString
	if err := row.Scan(&rec.ID, &rec.KeyHash, &rec.APIKey, &rec.User, &modelsRaw, &rec.RPMLimit, &rec.TPMLimit, &rec.ConcurrencyLimit, &rec.DailySpendLimitCents, &enabled, &created, &updated, &last, &revoked); err != nil {
		return rec, err
	}
	_ = json.Unmarshal([]byte(modelsRaw), &rec.AllowedModels)
	rec.Enabled = enabled == 1
	rec.CreatedAt, _ = time.Parse(time.RFC3339Nano, created)
	rec.UpdatedAt, _ = time.Parse(time.RFC3339Nano, updated)
	if last.Valid {
		if t, err := time.Parse(time.RFC3339Nano, last.String); err == nil {
			rec.LastUsedAt = &t
		}
	}
	if revoked.Valid {
		if t, err := time.Parse(time.RFC3339Nano, revoked.String); err == nil {
			rec.RevokedAt = &t
		}
	}
	rec.State = virtualKeyState(rec.Enabled, rec.RevokedAt)
	return rec, nil
}

func virtualKeyState(enabled bool, revokedAt *time.Time) string {
	if revokedAt != nil && !revokedAt.IsZero() {
		return "revoked"
	}
	if !enabled {
		return "disabled"
	}
	return "active"
}
