// Package auth 实现虚拟 Key 鉴权中间件。
// 虚拟 Key 格式: Bearer sk-virtual-<id>
package auth

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"

	"github.com/gofiber/fiber/v2"
	"llm-gateway/internal/config"
)

// VirtualKeyInfo 保存解析后的虚拟 Key 信息。
type VirtualKeyInfo struct {
	Key                  string   `json:"-"`
	KeyHash              string   `json:"key_hash"`
	User                 string   `json:"user"`
	APIKey               string   `json:"api_key"`
	AllowedModels        []string `json:"allowed_models"`
	RPMLimit             int      `json:"rpm_limit"`
	TPMLimit             int      `json:"tpm_limit"`
	ConcurrencyLimit     int      `json:"concurrency_limit"`
	DailySpendLimitCents int64    `json:"daily_spend_limit_cents"`
}

// Auth 负责虚拟 Key 的校验和查询。
type Auth struct {
	disabled bool
	keys     map[string]*VirtualKeyInfo
	mu       sync.RWMutex
}

// NewAuth 从配置创建 Auth 实例。
func NewAuth(cfg *config.AuthConfig) *Auth {
	a := &Auth{
		disabled: cfg.Disabled,
		keys:     make(map[string]*VirtualKeyInfo),
	}
	for _, vk := range cfg.VirtualKeys {
		keyHash := HashKey(vk.Key)
		a.keys[keyHash] = &VirtualKeyInfo{
			Key:                  vk.Key,
			KeyHash:              keyHash,
			User:                 defaultString(vk.User, "default"),
			APIKey:               MaskKey(vk.Key),
			AllowedModels:        vk.AllowedModels,
			RPMLimit:             vk.RPMLimit,
			TPMLimit:             vk.TPMLimit,
			ConcurrencyLimit:     vk.ConcurrencyLimit,
			DailySpendLimitCents: vk.DailySpendLimitCents,
		}
	}
	return a
}

// Middleware 返回 Fiber 中间件，校验 Bearer 或 API Key Header 中的虚拟 Key。
func (a *Auth) Middleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		if a.disabled {
			return c.Next()
		}

		key, source := extractRequestKey(c)
		if key == "" {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "missing authorization header",
			})
		}

		if source == "authorization" {
			parts := strings.Fields(key)
			if len(parts) == 0 {
				return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
					"error": "invalid authorization format",
				})
			}
			if strings.EqualFold(parts[0], "Bearer") {
				if len(parts) < 2 {
					return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
						"error": "invalid authorization format",
					})
				}
				key = strings.Join(parts[1:], " ")
			} else {
				key = parts[0]
			}
			key = strings.TrimSpace(key)
			for strings.HasPrefix(strings.ToLower(key), "bearer ") {
				key = strings.TrimSpace(key[len("bearer "):])
			}
		}

		keyHash := HashKey(key)
		info, ok := a.GetByHash(keyHash)
		if ok {
			info.Key = key
			c.Locals("virtual_key", info)
			c.Locals("virtual_key_hash", info.KeyHash)
			c.Locals("virtual_key_user", info.User)
			c.Locals("virtual_key_api_key", info.APIKey)
			return c.Next()
		}

		return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
			"error": "invalid virtual key",
		})
	}
}

func extractRequestKey(c *fiber.Ctx) (string, string) {
	if authHeader := strings.TrimSpace(c.Get("Authorization")); authHeader != "" {
		return authHeader, "authorization"
	}
	for _, header := range []string{"X-API-Key", "Anthropic-API-Key", "Api-Key"} {
		if key := strings.TrimSpace(c.Get(header)); key != "" {
			return key, strings.ToLower(header)
		}
	}
	return "", ""
}

// GetByHash 查询虚拟 Key 信息。
func (a *Auth) GetByHash(keyHash string) (*VirtualKeyInfo, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	for hash, info := range a.keys {
		if subtle.ConstantTimeCompare([]byte(hash), []byte(keyHash)) == 1 {
			copyInfo := *info
			copyInfo.AllowedModels = append([]string(nil), info.AllowedModels...)
			return &copyInfo, true
		}
	}
	return nil, false
}

// AddOrUpdateByHash registers a virtual key hash.
func (a *Auth) AddOrUpdateByHash(info VirtualKeyInfo) {
	a.mu.Lock()
	defer a.mu.Unlock()
	copyInfo := info
	copyInfo.AllowedModels = append([]string(nil), info.AllowedModels...)
	copyInfo.Key = ""
	a.keys[copyInfo.KeyHash] = &copyInfo
}

// RemoveByHash unregisters a virtual key hash.
func (a *Auth) RemoveByHash(keyHash string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.keys, keyHash)
}

// Get 查询明文虚拟 Key 信息。
func (a *Auth) Get(key string) (*VirtualKeyInfo, bool) {
	return a.GetByHash(HashKey(key))
}

// All returns all active virtual keys.
func (a *Auth) All() []VirtualKeyInfo {
	a.mu.RLock()
	defer a.mu.RUnlock()
	result := make([]VirtualKeyInfo, 0, len(a.keys))
	for _, info := range a.keys {
		copyInfo := *info
		copyInfo.AllowedModels = append([]string(nil), info.AllowedModels...)
		result = append(result, copyInfo)
	}
	return result
}

// HashKey returns a stable SHA256 hash for key lookup.
func HashKey(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])
}

// MaskKey returns a display-safe key preview.
func MaskKey(key string) string {
	if len(key) <= 10 {
		return "***"
	}
	return key[:6] + "..." + key[len(key)-4:]
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func (a *Auth) getByPlaintextForCompatibility(key string) (*VirtualKeyInfo, bool) {
	a.mu.RLock()
	defer a.mu.RUnlock()
	info, ok := a.keys[key]
	if !ok {
		return nil, false
	}
	// 返回副本避免外部修改
	copyInfo := *info
	return &copyInfo, true
}

// IsModelAllowed 检查指定模型是否在虚拟 Key 的允许列表中。
func (a *Auth) IsModelAllowed(key, model string) bool {
	info, ok := a.Get(key)
	if !ok {
		return false
	}
	for _, m := range info.AllowedModels {
		if m == model {
			return true
		}
	}
	return false
}

// ModelCheckMiddleware 返回检查模型权限的中间件（应在 Auth.Middleware 之后使用）。
func (a *Auth) ModelCheckMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		info, ok := c.Locals("virtual_key").(*VirtualKeyInfo)
		if !ok {
			return c.Status(fiber.StatusUnauthorized).JSON(fiber.Map{
				"error": "virtual key not found in context",
			})
		}

		// 从请求体中解析 model（这里简化处理，实际由 handler 解析后传入）
		// 更合理的做法是在 handler 中检查，这里仅作占位
		_ = info
		return c.Next()
	}
}

// ExtractVirtualKey 从 Fiber Context 中提取虚拟 Key 信息。
func ExtractVirtualKey(c *fiber.Ctx) (*VirtualKeyInfo, bool) {
	info, ok := c.Locals("virtual_key").(*VirtualKeyInfo)
	return info, ok
}

// FormatBearer 格式化 Bearer Token。
func FormatBearer(key string) string {
	return fmt.Sprintf("Bearer %s", key)
}
