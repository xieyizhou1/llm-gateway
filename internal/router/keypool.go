// Package router 实现负载均衡、Key Pool 管理和限流。
package router

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"
	"llm-gateway/internal/config"
)

// KeyState 表示 Provider Key 的健康状态。
type KeyState int

const (
	KeyStateHealthy KeyState = iota
	KeyStateCooldown
	KeyStateDisabled
)

// ProviderKey 是带状态的 Provider Key 封装。
type ProviderKey struct {
	config.ProviderKey
	State       KeyState
	CooldownEnd time.Time
	FailCount   int32
	LastSeenAt  time.Time
}

// KeyPool 管理单个 Provider 的 Key 池，支持原子轮询和健康检查。
type KeyPool struct {
	providerName string
	keys         []*ProviderKey
	index        uint32
	mu           sync.RWMutex
}

func (s KeyState) String() string {
	switch s {
	case KeyStateHealthy:
		return "healthy"
	case KeyStateCooldown:
		return "cooldown"
	case KeyStateDisabled:
		return "disabled"
	default:
		return "unknown"
	}
}

// NewKeyPool 从配置创建 KeyPool。
func NewKeyPool(providerName string, cfg config.ProviderConfig) *KeyPool {
	keys := make([]*ProviderKey, len(cfg.Keys))
	for i, k := range cfg.Keys {
		keys[i] = &ProviderKey{
			ProviderKey: k,
			State:       KeyStateHealthy,
		}
	}
	return &KeyPool{
		providerName: providerName,
		keys:         keys,
	}
}

// Next 返回下一个健康的 Key（轮询策略）。
func (p *KeyPool) Next() (*ProviderKey, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.keys) == 0 {
		return nil, fmt.Errorf("no keys available for provider %s", p.providerName)
	}

	// 尝试最多 len(keys) 次
	for i := 0; i < len(p.keys); i++ {
		idx := int(atomic.AddUint32(&p.index, 1)-1) % len(p.keys)
		key := p.keys[idx]

		if key.State == KeyStateDisabled {
			continue
		}
		if key.State == KeyStateCooldown && time.Now().Before(key.CooldownEnd) {
			continue
		}
		if key.State == KeyStateCooldown && time.Now().After(key.CooldownEnd) {
			// 冷却结束，自动恢复
			key.State = KeyStateHealthy
			key.FailCount = 0
		}

		return key, nil
	}

	return nil, fmt.Errorf("no healthy keys available for provider %s", p.providerName)
}

// MarkCooldown 将 Key 标记为冷却状态（如收到 429）。
func (p *KeyPool) MarkCooldown(keyID string, duration time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			k.State = KeyStateCooldown
			k.CooldownEnd = time.Now().Add(duration)
			break
		}
	}
}

// MarkDisabled 将 Key 标记为失效（如收到 401）。
func (p *KeyPool) MarkDisabled(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			k.State = KeyStateDisabled
			break
		}
	}
}

// MarkFailure 记录 Key 失败，超过阈值后自动冷却。
func (p *KeyPool) MarkFailure(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			count := atomic.AddInt32(&k.FailCount, 1)
			if count >= 3 {
				k.State = KeyStateCooldown
				k.CooldownEnd = time.Now().Add(60 * time.Second)
			}
			break
		}
	}
}

// ResetFailure 重置 Key 的失败计数和健康状态。
func (p *KeyPool) ResetFailure(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			atomic.StoreInt32(&k.FailCount, 0)
			k.State = KeyStateHealthy
			break
		}
	}
}

// MarkSuccess records a successful upstream response. It clears any failure
// count and recovers a cooldown key, but leaves manually disabled keys alone.
func (p *KeyPool) MarkSuccess(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			atomic.StoreInt32(&k.FailCount, 0)
			k.LastSeenAt = time.Now()
			if k.State == KeyStateCooldown {
				k.State = KeyStateHealthy
				k.CooldownEnd = time.Time{}
			}
			break
		}
	}
}

// MarkHeartbeat records a key as seen by the heartbeat probe.
func (p *KeyPool) MarkHeartbeat(keyID string, ok bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			k.LastSeenAt = time.Now()
			if !ok {
				count := atomic.AddInt32(&k.FailCount, 1)
				if count >= 3 && k.State != KeyStateDisabled {
					k.State = KeyStateCooldown
					k.CooldownEnd = time.Now().Add(60 * time.Second)
				}
			} else {
				atomic.StoreInt32(&k.FailCount, 0)
				if k.State == KeyStateCooldown {
					k.State = KeyStateHealthy
					k.CooldownEnd = time.Time{}
				}
			}
			break
		}
	}
}

// MarkHealthy enables a key and clears any cooldown/failure state.
func (p *KeyPool) MarkHealthy(keyID string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, k := range p.keys {
		if k.ID == keyID {
			atomic.StoreInt32(&k.FailCount, 0)
			k.State = KeyStateHealthy
			k.CooldownEnd = time.Time{}
			break
		}
	}
}

// HealthyCount 返回健康 Key 数量。
func (p *KeyPool) HealthyCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	count := 0
	now := time.Now()
	for _, k := range p.keys {
		if k.State == KeyStateHealthy {
			count++
		} else if k.State == KeyStateCooldown && now.After(k.CooldownEnd) {
			count++
		}
	}
	return count
}

// AllKeys 返回所有 Key 的副本（用于监控）。
func (p *KeyPool) AllKeys() []ProviderKey {
	p.mu.RLock()
	defer p.mu.RUnlock()
	result := make([]ProviderKey, len(p.keys))
	for i, k := range p.keys {
		result[i] = *k
	}
	return result
}

// RateLimiter 基于 Redis 的滑动窗口限流器。
type RateLimiter struct {
	client   *redis.Client
	window   time.Duration
	sequence uint64
}

// NewRateLimiter 创建限流器。
func NewRateLimiter(client *redis.Client, window time.Duration) *RateLimiter {
	return &RateLimiter{
		client: client,
		window: window,
	}
}

// Allow 检查指定 key 在滑动窗口内是否允许通过。
func (r *RateLimiter) Allow(ctx context.Context, key string, limit int) (bool, error) {
	now := time.Now().UnixMilli()
	windowStart := now - r.window.Milliseconds()

	pipe := r.client.Pipeline()
	zremRangeByScore := pipe.ZRemRangeByScore(ctx, key, "0", fmt.Sprintf("%d", windowStart))
	zcard := pipe.ZCard(ctx, key)
	member := fmt.Sprintf("%d-%d", now, atomic.AddUint64(&r.sequence, 1))
	zadd := pipe.ZAdd(ctx, key, redis.Z{Score: float64(now), Member: member})
	expire := pipe.Expire(ctx, key, r.window)

	_, err := pipe.Exec(ctx)
	if err != nil {
		return false, err
	}

	_ = zremRangeByScore
	_ = zadd
	_ = expire

	count := zcard.Val()
	return int(count) < limit, nil
}

// Router 负责选择 Provider 和 Key，并执行限流检查。
type Router struct {
	pools           map[string]*KeyPool
	limiter         *RateLimiter
	strategy        string
	providerOrder   []string
	modelMap        map[string]config.ProviderConfig
	capabilities    map[string]config.ModelCapability
	cooldownSeconds int
}

// NewRouter 创建 Router 实例。
func NewRouter(cfg *config.Config, redisClient *redis.Client) *Router {
	pools := make(map[string]*KeyPool)
	pools["kimi"] = NewKeyPool("kimi", cfg.Providers.Kimi)
	pools["deepseek"] = NewKeyPool("deepseek", cfg.Providers.DeepSeek)

	modelMap := make(map[string]config.ProviderConfig)
	modelMap["kimi"] = cfg.Providers.Kimi
	modelMap["deepseek"] = cfg.Providers.DeepSeek

	var limiter *RateLimiter
	if redisClient != nil {
		limiter = NewRateLimiter(redisClient, time.Minute)
	}

	return &Router{
		pools:           pools,
		limiter:         limiter,
		strategy:        cfg.Router.Strategy,
		providerOrder:   cfg.Router.ProviderOrder,
		modelMap:        modelMap,
		capabilities:    cfg.ModelCapabilities,
		cooldownSeconds: cfg.Router.CooldownSeconds,
	}
}

// SetPoolForTest 仅用于测试，替换指定 Provider 的 KeyPool。
func (r *Router) SetPoolForTest(provider string, pool *KeyPool) {
	if r.pools == nil {
		r.pools = make(map[string]*KeyPool)
	}
	r.pools[provider] = pool
}

// DisableRateLimitForTest 仅用于测试，禁用限流检查。
func (r *Router) DisableRateLimitForTest() {
	r.limiter = nil
}

// SetModelMapForTest 仅用于测试，设置指定 Provider 的配置。
func (r *Router) SetModelMapForTest(provider string, cfg config.ProviderConfig) {
	if r.modelMap == nil {
		r.modelMap = make(map[string]config.ProviderConfig)
	}
	r.modelMap[provider] = cfg
}

// ProviderEndpoint describes an upstream provider endpoint for connection warmup.
type ProviderEndpoint struct {
	Provider string
	BaseURL  string
	Headers  map[string]string
	Keys     []config.ProviderKey
}

// ProviderEndpoints returns all configured provider endpoints.
func (r *Router) ProviderEndpoints() []ProviderEndpoint {
	result := make([]ProviderEndpoint, 0, len(r.modelMap))
	for name, cfg := range r.modelMap {
		result = append(result, ProviderEndpoint{
			Provider: name,
			BaseURL:  cfg.BaseURL,
			Headers:  cfg.Headers,
			Keys:     cfg.Keys,
		})
	}
	return result
}

// Capability returns the configured capability profile for a client-facing model alias.
func (r *Router) Capability(model string) (config.ModelCapability, bool) {
	if r == nil {
		return config.ModelCapability{}, false
	}
	cap, ok := r.capabilities[model]
	return cap, ok
}

// ProviderKeyStatus is a display-safe snapshot of a provider key.
type ProviderKeyStatus struct {
	Provider    string    `json:"provider"`
	ID          string    `json:"id"`
	State       string    `json:"state"`
	CooldownEnd string    `json:"cooldown_end,omitempty"`
	FailCount   int32     `json:"fail_count"`
	LastSeenAt  time.Time `json:"last_seen_at,omitempty"`
}

// RouteResult 包含路由选择的结果。
type RouteResult struct {
	Provider string
	BaseURL  string
	Key      *ProviderKey
	Headers  map[string]string
	ModelMap map[string]string
}

// Route 根据模型名选择 Provider 和 Key。
func (r *Router) Route(ctx context.Context, model string) (*RouteResult, error) {
	return r.RouteWithOptions(ctx, model, RouteOptions{})
}

// RouteOptions describes required request capabilities.
type RouteOptions struct {
	RequireImages bool
}

// RouteWithOptions 根据模型名和请求特性选择 Provider 和 Key。
func (r *Router) RouteWithOptions(ctx context.Context, model string, opts RouteOptions) (*RouteResult, error) {
	providers := r.priorityProvidersForModel(model)
	if len(providers) == 0 {
		providers = r.providerOrder
	}
	if len(providers) == 0 {
		providers = []string{"kimi", "deepseek"}
	}

	for _, providerName := range providers {
		pool := r.pools[providerName]
		if pool == nil {
			continue
		}
		cfg := r.modelMap[providerName]
		if opts.RequireImages && !r.providerSupportsImages(cfg, model) {
			continue
		}
		key, err := pool.Next()
		if err != nil {
			continue
		}

		// 限流检查
		if r.limiter != nil {
			limitKey := fmt.Sprintf("rate:%s:%s", providerName, key.ID)
			allowed, err := r.limiter.Allow(ctx, limitKey, key.RPMLimit)
			if err != nil {
				// Redis 是限流依赖，不应在异常时阻断核心转发路径。
				allowed = true
			}
			if !allowed {
				continue
			}
		}

		return &RouteResult{
			Provider: providerName,
			BaseURL:  cfg.BaseURL,
			Key:      key,
			Headers:  cfg.Headers,
			ModelMap: cfg.ModelMap,
		}, nil
	}

	return nil, fmt.Errorf("no available provider for model %s", model)
}

func (r *Router) priorityProvidersForModel(model string) []string {
	if model == "" {
		return nil
	}
	seen := make(map[string]bool, len(r.pools))
	for _, providerName := range r.providerOrder {
		cfg, ok := r.modelMap[providerName]
		if !ok || cfg.ModelMap == nil {
			continue
		}
		if upstream, ok := cfg.ModelMap[model]; !ok || upstream == "" {
			continue
		}
		if seen[providerName] {
			continue
		}
		seen[providerName] = true
	}
	ordered := make([]string, 0, len(seen))
	for _, providerName := range r.providerOrder {
		if seen[providerName] {
			ordered = append(ordered, providerName)
			delete(seen, providerName)
		}
	}
	return ordered
}

func (r *Router) providerSupportsImages(cfg config.ProviderConfig, model string) bool {
	if !cfg.SupportsImages {
		return false
	}
	mapped, ok := cfg.ModelMap[model]
	if !ok || mapped == "" {
		return false
	}
	capability, ok := r.capabilities[model]
	if !ok {
		return false
	}
	return capability.SupportsImages
}

// MarkKeyResult 根据上游响应状态更新 Key 健康状态。
func (r *Router) MarkKeyResult(provider, keyID string, statusCode int) {
	pool, ok := r.pools[provider]
	if !ok {
		return
	}

	switch {
	case statusCode == http.StatusUnauthorized:
		pool.MarkDisabled(keyID)
	case statusCode == http.StatusTooManyRequests || statusCode == http.StatusPaymentRequired:
		pool.MarkCooldown(keyID, time.Duration(r.cooldownSeconds)*time.Second)
	case statusCode == http.StatusServiceUnavailable:
		pool.MarkHeartbeat(keyID, false)
	case statusCode >= 500:
		pool.MarkFailure(keyID)
	default:
		pool.MarkSuccess(keyID)
	}
}

// HealthStatus 返回所有 Provider 的健康状态概览。
func (r *Router) HealthStatus() map[string]int {
	result := make(map[string]int)
	for name, pool := range r.pools {
		result[name] = pool.HealthyCount()
	}
	return result
}

// AllKeyStatuses returns all provider key states.
func (r *Router) AllKeyStatuses() []ProviderKeyStatus {
	result := []ProviderKeyStatus{}
	for provider, pool := range r.pools {
		for _, key := range pool.AllKeys() {
			item := ProviderKeyStatus{
				Provider:   provider,
				ID:         key.ID,
				State:      key.State.String(),
				FailCount:  key.FailCount,
				LastSeenAt: key.LastSeenAt,
			}
			if !key.CooldownEnd.IsZero() {
				item.CooldownEnd = key.CooldownEnd.Format(time.RFC3339Nano)
			}
			result = append(result, item)
		}
	}
	return result
}

// SetKeyState changes a provider key state from dashboard controls.
func (r *Router) SetKeyState(provider, keyID, action string) bool {
	pool, ok := r.pools[provider]
	if !ok {
		return false
	}
	switch action {
	case "enable":
		pool.MarkHealthy(keyID)
	case "cooldown":
		pool.MarkCooldown(keyID, 60*time.Second)
	case "disable":
		pool.MarkDisabled(keyID)
	default:
		return false
	}
	return true
}
