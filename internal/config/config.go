// Package config 负责加载和管理 LLM Gateway 的运行时配置。
// 支持从 YAML 文件和环境变量读取，虚拟 Key 的 API Key 通过 ${ENV_VAR} 语法从环境变量注入。
package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/spf13/viper"
)

// Config 是网关的全局配置根节点。
type Config struct {
	Server            ServerConfig               `mapstructure:"server"`
	Auth              AuthConfig                 `mapstructure:"auth"`
	Providers         ProvidersConfig            `mapstructure:"providers"`
	Redis             RedisConfig                `mapstructure:"redis"`
	Router            RouterConfig               `mapstructure:"router"`
	Usage             UsageConfig                `mapstructure:"usage"`
	Billing           BillingConfig              `mapstructure:"billing"`
	Dashboard         DashboardConfig            `mapstructure:"dashboard"`
	ModelCapabilities map[string]ModelCapability `mapstructure:"model_capabilities"`
	Logging           LoggingConfig              `mapstructure:"logging"`
}

// ServerConfig 定义 HTTP 服务监听参数。
type ServerConfig struct {
	Host string `mapstructure:"host"`
	Port int    `mapstructure:"port"`
}

// AuthConfig 包含虚拟 Key 列表，用于客户端鉴权。
type AuthConfig struct {
	Disabled    bool         `mapstructure:"disabled"`
	VirtualKeys []VirtualKey `mapstructure:"virtual_keys"`
}

// VirtualKey 定义一个客户端虚拟 Key 的权限和配额。
type VirtualKey struct {
	Key                  string   `mapstructure:"key"`
	User                 string   `mapstructure:"user"`
	AllowedModels        []string `mapstructure:"allowed_models"`
	RPMLimit             int      `mapstructure:"rpm_limit"`
	TPMLimit             int      `mapstructure:"tpm_limit"`
	ConcurrencyLimit     int      `mapstructure:"concurrency_limit"`
	DailySpendLimitCents int64    `mapstructure:"daily_spend_limit_cents"`
}

// ProvidersConfig 包含所有上游 LLM Provider 的配置。
type ProvidersConfig struct {
	Kimi     ProviderConfig `mapstructure:"kimi"`
	DeepSeek ProviderConfig `mapstructure:"deepseek"`
}

// ProviderConfig 定义单个 Provider 的 BaseURL、Key 池和模型映射。
type ProviderConfig struct {
	BaseURL  string            `mapstructure:"base_url"`
	Headers  map[string]string `mapstructure:"headers"`
	Keys     []ProviderKey     `mapstructure:"keys"`
	ModelMap map[string]string // 手动加载，避免 viper 解析带点键出错
}

// ProviderKey 是 Provider 的真实 API Key 配置。
type ProviderKey struct {
	ID               string `mapstructure:"id"`
	Key              string `mapstructure:"key"`
	Weight           int    `mapstructure:"weight"`
	RPMLimit         int    `mapstructure:"rpm_limit"`
	ConcurrencyLimit int    `mapstructure:"concurrency_limit"`
}

// RedisConfig 定义 Redis 连接参数，用于限流和健康检查。
type RedisConfig struct {
	Host     string `mapstructure:"host"`
	Port     int    `mapstructure:"port"`
	Password string `mapstructure:"password"`
	DB       int    `mapstructure:"db"`
}

// RouterConfig 定义负载均衡和重试策略。
type RouterConfig struct {
	Strategy         string   `mapstructure:"strategy"`
	RetryCount       int      `mapstructure:"retry_count"`
	TimeoutSeconds   int      `mapstructure:"timeout_seconds"`
	ProviderOrder    []string `mapstructure:"provider_order"`
	CooldownSeconds  int      `mapstructure:"cooldown_seconds"`
	FailureThreshold int32    `mapstructure:"failure_threshold"`
}

// UsageConfig controls request persistence.
type UsageConfig struct {
	Enabled    bool   `mapstructure:"enabled"`
	SQLitePath string `mapstructure:"sqlite_path"`
}

// BillingConfig controls simple spend accounting.
type BillingConfig struct {
	Enabled                 bool  `mapstructure:"enabled"`
	DefaultInputCentsPer1M  int64 `mapstructure:"default_input_cents_per_1m"`
	DefaultOutputCentsPer1M int64 `mapstructure:"default_output_cents_per_1m"`
	DailySpendLimitCents    int64 `mapstructure:"daily_spend_limit_cents"`
}

// DashboardConfig controls the built-in read-only dashboard.
type DashboardConfig struct {
	Enabled bool   `mapstructure:"enabled"`
	Token   string `mapstructure:"token"`
}

// ModelCapability describes the externally advertised capability of a client-facing model alias.
type ModelCapability struct {
	UpstreamModel     string `mapstructure:"upstream_model" json:"upstream_model"`
	MaxContextTokens  int    `mapstructure:"max_context_tokens" json:"max_context_tokens"`
	MaxOutputTokens   int    `mapstructure:"max_output_tokens" json:"max_output_tokens"`
	SupportsTools     bool   `mapstructure:"supports_tools" json:"supports_tools"`
	SupportsStreaming bool   `mapstructure:"supports_streaming" json:"supports_streaming"`
	SupportsThinking  bool   `mapstructure:"supports_thinking" json:"supports_thinking"`
	SupportsCache     bool   `mapstructure:"supports_prompt_cache" json:"supports_prompt_cache"`
	SupportsImages    bool   `mapstructure:"supports_images" json:"supports_images"`
	Notes             string `mapstructure:"notes" json:"notes,omitempty"`
}

// LoggingConfig 定义日志输出行为。
type LoggingConfig struct {
	Level  string `mapstructure:"level"`
	Format string `mapstructure:"format"`
}

var envVarPattern = regexp.MustCompile(`\$\{([^}]+)\}`)

// Load 从指定路径加载配置文件，并解析环境变量占位符。
func Load(path string) (*Config, error) {
	viper.SetConfigFile(path)
	viper.SetEnvPrefix("LG")
	viper.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	viper.AutomaticEnv()

	if err := viper.ReadInConfig(); err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	var cfg Config
	if err := viper.Unmarshal(&cfg); err != nil {
		return nil, fmt.Errorf("unmarshal config: %w", err)
	}

	// 手动修正带点的键（如 gpt-5.4）
	cfg.Providers.Kimi.ModelMap = loadModelMap("providers.kimi.model_map")
	cfg.Providers.DeepSeek.ModelMap = loadModelMap("providers.deepseek.model_map")
	cfg.ModelCapabilities = loadModelCapabilities("model_capabilities")

	// 解析 Provider Key 中的 ${ENV_VAR} 占位符
	resolveProviderKeys(&cfg.Providers.Kimi)
	resolveProviderKeys(&cfg.Providers.DeepSeek)

	// 解析 Auth 虚拟 Key 中的 ${ENV_VAR} 占位符
	resolveAuthKeys(&cfg.Auth)
	cfg.Redis.Password = resolveEnvVars(cfg.Redis.Password)
	cfg.Dashboard.Token = resolveEnvVars(cfg.Dashboard.Token)

	if cfg.Router.CooldownSeconds == 0 {
		cfg.Router.CooldownSeconds = 60
	}
	if cfg.Router.FailureThreshold == 0 {
		cfg.Router.FailureThreshold = 3
	}
	if cfg.Usage.SQLitePath == "" {
		cfg.Usage.SQLitePath = "llm-gateway.db"
	}
	applyDefaultModelCapabilities(&cfg)

	return &cfg, nil
}

// loadModelMap 从 viper 读取 model_map，处理带点的键（如 gpt-5.4）被解析为嵌套 map 的情况。
func loadModelMap(key string) map[string]string {
	result := make(map[string]string)
	raw := viper.Get(key)
	if raw == nil {
		return result
	}

	// viper 可能把 gpt-5.4 解析为 map[gpt-5:map[4:deepseek-v4-pro]]
	// 需要递归 flatten 这种嵌套
	flattenMap("", raw, result)
	return result
}

func flattenMap(prefix string, v interface{}, out map[string]string) {
	switch m := v.(type) {
	case map[string]interface{}:
		for k, val := range m {
			newKey := k
			if prefix != "" {
				newKey = prefix + "." + k
			}
			flattenMap(newKey, val, out)
		}
	case map[interface{}]interface{}:
		for k, val := range m {
			ks, _ := k.(string)
			newKey := ks
			if prefix != "" {
				newKey = prefix + "." + ks
			}
			flattenMap(newKey, val, out)
		}
	case string:
		if prefix != "" {
			out[prefix] = m
		}
	}
}

func loadModelCapabilities(key string) map[string]ModelCapability {
	result := make(map[string]ModelCapability)
	raw := viper.Get(key)
	if raw == nil {
		return result
	}
	flattenModelCapabilities("", raw, result)
	return result
}

func flattenModelCapabilities(prefix string, v interface{}, out map[string]ModelCapability) {
	switch m := v.(type) {
	case map[string]interface{}:
		if cap, ok := decodeModelCapability(m); ok && prefix != "" {
			out[prefix] = cap
			return
		}
		for k, val := range m {
			newKey := k
			if prefix != "" {
				newKey = prefix + "." + k
			}
			flattenModelCapabilities(newKey, val, out)
		}
	case map[interface{}]interface{}:
		converted := make(map[string]interface{}, len(m))
		for k, val := range m {
			ks, _ := k.(string)
			converted[ks] = val
		}
		flattenModelCapabilities(prefix, converted, out)
	}
}

func decodeModelCapability(m map[string]interface{}) (ModelCapability, bool) {
	_, hasContext := m["max_context_tokens"]
	_, hasOutput := m["max_output_tokens"]
	_, hasUpstream := m["upstream_model"]
	if !hasContext && !hasOutput && !hasUpstream {
		return ModelCapability{}, false
	}
	return ModelCapability{
		UpstreamModel:     stringValue(m["upstream_model"]),
		MaxContextTokens:  intValue(m["max_context_tokens"]),
		MaxOutputTokens:   intValue(m["max_output_tokens"]),
		SupportsTools:     boolValue(m["supports_tools"]),
		SupportsStreaming: boolValue(m["supports_streaming"]),
		SupportsThinking:  boolValue(m["supports_thinking"]),
		SupportsCache:     boolValue(m["supports_prompt_cache"]),
		SupportsImages:    boolValue(m["supports_images"]),
		Notes:             stringValue(m["notes"]),
	}, true
}

func applyDefaultModelCapabilities(cfg *Config) {
	if cfg.ModelCapabilities == nil {
		cfg.ModelCapabilities = map[string]ModelCapability{}
	}
	aliases := map[string]string{}
	for alias, upstream := range cfg.Providers.Kimi.ModelMap {
		aliases[alias] = upstream
	}
	for _, key := range cfg.Auth.VirtualKeys {
		for _, model := range key.AllowedModels {
			if _, ok := aliases[model]; !ok {
				if upstream := cfg.Providers.Kimi.ModelMap[model]; upstream != "" {
					aliases[model] = upstream
				}
			}
		}
	}
	defaultCap := ModelCapability{
		UpstreamModel:     "kimi-for-coding",
		MaxContextTokens:  262144,
		MaxOutputTokens:   32768,
		SupportsTools:     true,
		SupportsStreaming: true,
		SupportsThinking:  true,
		SupportsCache:     true,
		SupportsImages:    false,
		Notes:             "Client-facing alias backed by Kimi for Coding; not a 1M Claude/OpenAI model.",
	}
	for alias, upstream := range aliases {
		if upstream != "kimi-for-coding" {
			continue
		}
		if _, ok := cfg.ModelCapabilities[alias]; !ok {
			cap := defaultCap
			cap.UpstreamModel = upstream
			cfg.ModelCapabilities[alias] = cap
		}
	}
	if _, ok := cfg.ModelCapabilities["kimi-for-coding"]; !ok {
		cfg.ModelCapabilities["kimi-for-coding"] = defaultCap
	}
}

func stringValue(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func intValue(v interface{}) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case string:
		var n int
		_, _ = fmt.Sscanf(x, "%d", &n)
		return n
	default:
		return 0
	}
}

func boolValue(v interface{}) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return strings.EqualFold(x, "true")
	default:
		return false
	}
}

func resolveProviderKeys(p *ProviderConfig) {
	for i := range p.Keys {
		p.Keys[i].Key = resolveEnvVars(p.Keys[i].Key)
	}
}

func resolveAuthKeys(a *AuthConfig) {
	for i := range a.VirtualKeys {
		a.VirtualKeys[i].Key = resolveEnvVars(a.VirtualKeys[i].Key)
	}
}

func resolveEnvVars(s string) string {
	return envVarPattern.ReplaceAllStringFunc(s, func(match string) string {
		varName := match[2 : len(match)-1]
		if val := os.Getenv(varName); val != "" {
			return val
		}
		return match
	})
}
