package router

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"llm-gateway/internal/config"
)

func TestNewKeyPool(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
			{ID: "key-2", Key: "test-key-2", Weight: 2, RPMLimit: 120},
		},
	}

	pool := NewKeyPool("test", cfg)
	if pool == nil {
		t.Fatal("expected non-nil KeyPool")
	}

	keys := pool.AllKeys()
	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}
}

func TestKeyPoolNext(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)

	key, err := pool.Next()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if key.ID != "key-1" {
		t.Errorf("expected key-1, got %s", key.ID)
	}
}

func TestKeyPoolNextEmpty(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys:    []config.ProviderKey{},
	}

	pool := NewKeyPool("test", cfg)
	_, err := pool.Next()
	if err == nil {
		t.Error("expected error for empty key pool")
	}
}

func TestKeyPoolMarkCooldown(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)
	pool.MarkCooldown("key-1", 5*time.Second)

	// 冷却期间应该无法获取
	_, err := pool.Next()
	if err == nil {
		t.Error("expected error when key is in cooldown")
	}
}

func TestKeyPoolMarkDisabled(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
			{ID: "key-2", Key: "test-key-2", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)
	pool.MarkDisabled("key-1")

	// 应该能获取到 key-2
	key, err := pool.Next()
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if key.ID != "key-2" {
		t.Errorf("expected key-2, got %s", key.ID)
	}
}

func TestKeyPoolHealthyCount(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
			{ID: "key-2", Key: "test-key-2", Weight: 1, RPMLimit: 60},
			{ID: "key-3", Key: "test-key-3", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)
	if pool.HealthyCount() != 3 {
		t.Errorf("expected 3 healthy keys, got %d", pool.HealthyCount())
	}

	pool.MarkDisabled("key-1")
	if pool.HealthyCount() != 2 {
		t.Errorf("expected 2 healthy keys, got %d", pool.HealthyCount())
	}
}

func TestKeyPoolMarkFailure(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)

	// 失败 3 次后应该进入冷却
	pool.MarkFailure("key-1")
	pool.MarkFailure("key-1")
	pool.MarkFailure("key-1")

	_, err := pool.Next()
	if err == nil {
		t.Error("expected error after 3 failures")
	}
}

func TestKeyPoolRoundRobin(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
			{ID: "key-2", Key: "test-key-2", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)

	// 轮询应该交替返回 key-1 和 key-2
	key1, _ := pool.Next()
	key2, _ := pool.Next()

	if key1.ID == key2.ID {
		t.Error("expected round-robin to alternate keys")
	}
}

func TestKeyPoolResetFailure(t *testing.T) {
	cfg := config.ProviderConfig{
		BaseURL: "https://api.test.com",
		Keys: []config.ProviderKey{
			{ID: "key-1", Key: "test-key-1", Weight: 1, RPMLimit: 60},
		},
	}

	pool := NewKeyPool("test", cfg)

	// 先让 key-1 失败 3 次进入冷却
	pool.MarkFailure("key-1")
	pool.MarkFailure("key-1")
	pool.MarkFailure("key-1")

	_, err := pool.Next()
	if err == nil {
		t.Fatal("expected error after 3 failures")
	}

	// 重置失败计数
	pool.ResetFailure("key-1")

	key, err := pool.Next()
	if err != nil {
		t.Fatalf("expected no error after reset, got %v", err)
	}
	if key.ID != "key-1" {
		t.Errorf("expected key-1, got %s", key.ID)
	}
}

func TestRouteWithOptionsRequiresImageCapableProvider(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://kimi.test/v1",
				Keys: []config.ProviderKey{
					{ID: "kimi-1", Key: "kimi-key", RPMLimit: 60},
				},
				ModelMap: map[string]string{"kimi-for-coding": "kimi-for-coding"},
			},
			DeepSeek: config.ProviderConfig{
				BaseURL: "https://deepseek.test/v1",
				Keys: []config.ProviderKey{
					{ID: "deepseek-1", Key: "deepseek-key", RPMLimit: 60},
				},
				ModelMap: map[string]string{},
			},
		},
		Router: config.RouterConfig{
			ProviderOrder: []string{"deepseek", "kimi"},
		},
		ModelCapabilities: map[string]config.ModelCapability{
			"kimi-for-coding": {SupportsImages: true},
		},
	}

	r := NewRouter(cfg, nil)
	route, err := r.RouteWithOptions(context.Background(), "kimi-for-coding", RouteOptions{RequireImages: true})
	if err != nil {
		t.Fatalf("expected image-capable route, got %v", err)
	}
	if route.Provider != "kimi" {
		t.Fatalf("expected kimi route for image request, got %s", route.Provider)
	}
}

func TestRoutePrefersMappedProviderForModel(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://kimi.test/v1",
				Keys: []config.ProviderKey{
					{ID: "kimi-1", Key: "kimi-key", RPMLimit: 60},
				},
				ModelMap: map[string]string{"kimi-for-coding": "kimi-for-coding"},
			},
			DeepSeek: config.ProviderConfig{
				BaseURL: "https://deepseek.test/v1",
				Keys: []config.ProviderKey{
					{ID: "deepseek-1", Key: "deepseek-key", RPMLimit: 60},
				},
				ModelMap: map[string]string{},
			},
		},
		Router: config.RouterConfig{
			ProviderOrder: []string{"deepseek", "kimi"},
		},
	}

	r := NewRouter(cfg, nil)
	route, err := r.Route(context.Background(), "kimi-for-coding")
	if err != nil {
		t.Fatalf("expected route, got %v", err)
	}
	if route.Provider != "kimi" {
		t.Fatalf("expected kimi route for kimi-for-coding, got %s", route.Provider)
	}
}

func TestRateLimiterAllow(t *testing.T) {
	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	limiter := NewRateLimiter(client, time.Minute)

	ctx := context.Background()

	// 前 3 次应该允许
	for i := 0; i < 3; i++ {
		allowed, err := limiter.Allow(ctx, "test-key", 3)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !allowed {
			t.Errorf("expected allowed on attempt %d", i+1)
		}
	}

	// 第 4 次应该拒绝
	allowed, err := limiter.Allow(ctx, "test-key", 3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allowed {
		t.Error("expected not allowed after limit exceeded")
	}
}

func TestNewRouter(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
			DeepSeek: config.ProviderConfig{
				BaseURL: "https://api.deepseek.com",
				Keys: []config.ProviderKey{
					{ID: "d1", Key: "sk-d1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis: config.RedisConfig{
			Host: "localhost",
			Port: 6379,
			DB:   0,
		},
		Logging: config.LoggingConfig{
			Level:  "INFO",
			Format: "json",
		},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)
	if r == nil {
		t.Fatal("expected non-nil Router")
	}

	status := r.HealthStatus()
	if status["kimi"] != 1 {
		t.Errorf("expected kimi healthy count 1, got %d", status["kimi"])
	}
	if status["deepseek"] != 1 {
		t.Errorf("expected deepseek healthy count 1, got %d", status["deepseek"])
	}
}

func TestRouterRoute(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis: config.RedisConfig{
			Host: "localhost",
			Port: 6379,
			DB:   0,
		},
		Logging: config.LoggingConfig{
			Level:  "INFO",
			Format: "json",
		},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)
	ctx := context.Background()

	route, err := r.Route(ctx, "gpt-4")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.Provider != "kimi" {
		t.Errorf("expected provider kimi, got %s", route.Provider)
	}
	if route.Key == nil {
		t.Fatal("expected non-nil key")
	}
	if route.Key.ID != "k1" {
		t.Errorf("expected key id k1, got %s", route.Key.ID)
	}
	if route.BaseURL != "https://api.kimi.com" {
		t.Errorf("expected base url https://api.kimi.com, got %s", route.BaseURL)
	}
}

func TestRouterMarkKeyResult(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis: config.RedisConfig{
			Host: "localhost",
			Port: 6379,
			DB:   0,
		},
		Logging: config.LoggingConfig{
			Level:  "INFO",
			Format: "json",
		},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	// 429 -> cooldown
	r.MarkKeyResult("kimi", "k1", 429)
	ctx := context.Background()
	_, err := r.Route(ctx, "gpt-4")
	if err == nil {
		t.Error("expected error after marking cooldown")
	}

	// 401 -> disabled
	r.MarkKeyResult("kimi", "k1", 401)
	status := r.HealthStatus()
	if status["kimi"] != 0 {
		t.Errorf("expected 0 healthy keys, got %d", status["kimi"])
	}
}

func TestRouterHealthStatus(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
					{ID: "k2", Key: "sk-k2", Weight: 1, RPMLimit: 60},
				},
			},
			DeepSeek: config.ProviderConfig{
				BaseURL: "https://api.deepseek.com",
				Keys: []config.ProviderKey{
					{ID: "d1", Key: "sk-d1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis: config.RedisConfig{
			Host: "localhost",
			Port: 6379,
			DB:   0,
		},
		Logging: config.LoggingConfig{
			Level:  "INFO",
			Format: "json",
		},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	status := r.HealthStatus()
	if status["kimi"] != 2 {
		t.Errorf("expected kimi count 2, got %d", status["kimi"])
	}
	if status["deepseek"] != 1 {
		t.Errorf("expected deepseek count 1, got %d", status["deepseek"])
	}

	// disable one key
	r.MarkKeyResult("kimi", "k1", 401)
	status = r.HealthStatus()
	if status["kimi"] != 1 {
		t.Errorf("expected kimi count 1 after disable, got %d", status["kimi"])
	}
}

func TestRouterRouteRateLimitExceeded(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 0},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)
	ctx := context.Background()

	// RPM limit = 0，应该触发限流拒绝
	_, err := r.Route(ctx, "gpt-4")
	if err == nil {
		t.Error("expected error when rate limit exceeded")
	}
}

func TestRouterRouteWithoutRedisClient(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	r := NewRouter(cfg, nil)
	route, err := r.Route(context.Background(), "gpt-4")
	if err != nil {
		t.Fatalf("unexpected error without redis client: %v", err)
	}
	if route.Provider != "kimi" {
		t.Errorf("expected provider kimi, got %s", route.Provider)
	}
}

func TestRouterMarkKeyResult500(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	// 500 应该触发 MarkFailure
	r.MarkKeyResult("kimi", "k1", 500)
	// 第一次失败不应立即冷却
	ctx := context.Background()
	_, err := r.Route(ctx, "gpt-4")
	if err != nil {
		t.Fatalf("unexpected error after first 500: %v", err)
	}

	// 再触发两次失败
	r.MarkKeyResult("kimi", "k1", 500)
	r.MarkKeyResult("kimi", "k1", 500)

	_, err = r.Route(ctx, "gpt-4")
	if err == nil {
		t.Error("expected error after 3 failures (cooldown)")
	}
}

func TestRouterMarkKeyResultUnknownProvider(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	// 对不存在的 provider 调用 MarkKeyResult，不应 panic
	r.MarkKeyResult("unknown", "key1", 429)
}

func TestRouterSetPoolForTest(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	newPool := NewKeyPool("new", config.ProviderConfig{
		BaseURL: "https://api.new.com",
		Keys:    []config.ProviderKey{{ID: "n1", Key: "sk-n1", Weight: 1, RPMLimit: 60}},
	})
	r.SetPoolForTest("newprovider", newPool)

	status := r.HealthStatus()
	if status["newprovider"] != 1 {
		t.Errorf("expected newprovider count 1, got %d", status["newprovider"])
	}
}

func TestRouterDisableRateLimitForTest(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)
	r.DisableRateLimitForTest()

	ctx := context.Background()
	route, err := r.Route(ctx, "gpt-4")
	if err != nil {
		t.Fatalf("unexpected error after disabling rate limit: %v", err)
	}
	if route.Provider != "kimi" {
		t.Errorf("expected provider kimi, got %s", route.Provider)
	}
}

func TestRouterSetModelMapForTest(t *testing.T) {
	cfg := &config.Config{
		Providers: config.ProvidersConfig{
			Kimi: config.ProviderConfig{
				BaseURL: "https://api.kimi.com",
				Keys: []config.ProviderKey{
					{ID: "k1", Key: "sk-k1", Weight: 1, RPMLimit: 60},
				},
			},
		},
		Router: config.RouterConfig{
			Strategy:       "round_robin",
			RetryCount:     2,
			TimeoutSeconds: 30,
		},
		Redis:   config.RedisConfig{Host: "localhost", Port: 6379, DB: 0},
		Logging: config.LoggingConfig{Level: "INFO", Format: "json"},
	}

	mr := miniredis.RunT(t)
	defer mr.Close()

	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	defer client.Close()

	r := NewRouter(cfg, client)

	newCfg := config.ProviderConfig{
		BaseURL:  "https://api.new.com",
		ModelMap: map[string]string{"gpt-5": "new-model"},
	}
	r.SetModelMapForTest("kimi", newCfg)

	ctx := context.Background()
	route, err := r.Route(ctx, "gpt-5")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if route.BaseURL != "https://api.new.com" {
		t.Errorf("expected new base url, got %s", route.BaseURL)
	}
}
