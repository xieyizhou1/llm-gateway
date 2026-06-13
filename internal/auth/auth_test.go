package auth

import (
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v2"
	"llm-gateway/internal/config"
)

func TestNewAuth(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           "sk-virtual-test",
				AllowedModels: []string{"gpt-4", "claude-sonnet"},
				RPMLimit:      100,
			},
		},
	}

	a := NewAuth(cfg)
	if a == nil {
		t.Fatal("expected non-nil Auth")
	}
}

func TestAuthGet(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           "sk-virtual-test",
				AllowedModels: []string{"gpt-4"},
				RPMLimit:      100,
			},
		},
	}

	a := NewAuth(cfg)

	info, ok := a.Get("sk-virtual-test")
	if !ok {
		t.Fatal("expected key to be found")
	}
	if info.Key != "sk-virtual-test" {
		t.Errorf("expected key sk-virtual-test, got %s", info.Key)
	}
	if info.RPMLimit != 100 {
		t.Errorf("expected rpm_limit 100, got %d", info.RPMLimit)
	}

	_, ok = a.Get("sk-virtual-invalid")
	if ok {
		t.Error("expected invalid key to not be found")
	}
}

func TestAuthIsModelAllowed(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           "sk-virtual-test",
				AllowedModels: []string{"gpt-4", "claude-sonnet"},
				RPMLimit:      100,
			},
		},
	}

	a := NewAuth(cfg)

	if !a.IsModelAllowed("sk-virtual-test", "gpt-4") {
		t.Error("expected gpt-4 to be allowed")
	}
	if !a.IsModelAllowed("sk-virtual-test", "claude-sonnet") {
		t.Error("expected claude-sonnet to be allowed")
	}
	if a.IsModelAllowed("sk-virtual-test", "gpt-3.5") {
		t.Error("expected gpt-3.5 to not be allowed")
	}
	if a.IsModelAllowed("sk-virtual-invalid", "gpt-4") {
		t.Error("expected invalid key to not allow any model")
	}
}

func TestAuthGetReturnsCopy(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{
			{
				Key:           "sk-virtual-test",
				AllowedModels: []string{"gpt-4"},
				RPMLimit:      100,
			},
		},
	}

	a := NewAuth(cfg)
	info, _ := a.Get("sk-virtual-test")
	info.RPMLimit = 999

	info2, _ := a.Get("sk-virtual-test")
	if info2.RPMLimit != 100 {
		t.Error("expected original value to be unchanged")
	}
}

func TestAuthMiddlewareMissingHeader(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	a := NewAuth(cfg)
	app := fiber.New()
	app.Use(a.Middleware())
	app.Get("/test", func(c *fiber.Ctx) error { return c.SendString("ok") })

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 401 {
		t.Errorf("expected 401, got %d", resp.StatusCode)
	}
}

func TestAuthMiddlewareInvalidFormat(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	a := NewAuth(cfg)
	app := fiber.New()
	app.Use(a.Middleware())
	app.Get("/test", func(c *fiber.Ctx) error { return c.SendString("ok") })

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Basic sk-test")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 401 {
		t.Errorf("expected 401, got %d", resp.StatusCode)
	}
}

func TestAuthMiddlewareInvalidKey(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	a := NewAuth(cfg)
	app := fiber.New()
	app.Use(a.Middleware())
	app.Get("/test", func(c *fiber.Ctx) error { return c.SendString("ok") })

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer sk-invalid")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 401 {
		t.Errorf("expected 401, got %d", resp.StatusCode)
	}
}

func TestAuthMiddlewareSuccess(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	a := NewAuth(cfg)
	app := fiber.New()
	app.Use(a.Middleware())
	app.Get("/test", func(c *fiber.Ctx) error {
		vk, ok := ExtractVirtualKey(c)
		if !ok || vk.Key != "sk-test" {
			return c.Status(500).SendString("extract failed")
		}
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestAuthMiddlewareAuthorizationHeaderVariants(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	tests := []string{
		"sk-test",
		"Bearer    sk-test",
		"Bearer Bearer sk-test",
	}

	for _, headerValue := range tests {
		t.Run(headerValue, func(t *testing.T) {
			a := NewAuth(cfg)
			app := fiber.New()
			app.Use(a.Middleware())
			app.Get("/test", func(c *fiber.Ctx) error { return c.SendString("ok") })

			req := httptest.NewRequest("GET", "/test", nil)
			req.Header.Set("Authorization", headerValue)
			resp, err := app.Test(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.StatusCode != 200 {
				t.Errorf("expected 200, got %d", resp.StatusCode)
			}
		})
	}
}

func TestAuthMiddlewareAPIKeyHeaders(t *testing.T) {
	cfg := &config.AuthConfig{
		VirtualKeys: []config.VirtualKey{{Key: "sk-test", AllowedModels: []string{"gpt-4"}}},
	}
	tests := []struct {
		name   string
		header string
		value  string
	}{
		{name: "x-api-key", header: "X-API-Key", value: "sk-test"},
		{name: "anthropic-api-key", header: "Anthropic-API-Key", value: "sk-test"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := NewAuth(cfg)
			app := fiber.New()
			app.Use(a.Middleware())
			app.Get("/test", func(c *fiber.Ctx) error {
				vk, ok := ExtractVirtualKey(c)
				if !ok || vk.Key != "sk-test" {
					return c.Status(500).SendString("extract failed")
				}
				return c.SendString("ok")
			})

			req := httptest.NewRequest("GET", "/test", nil)
			req.Header.Set(tt.header, tt.value)
			resp, err := app.Test(req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp.StatusCode != 200 {
				t.Errorf("expected 200, got %d", resp.StatusCode)
			}
		})
	}
}

func TestExtractVirtualKeyMissing(t *testing.T) {
	app := fiber.New()
	app.Get("/test", func(c *fiber.Ctx) error {
		vk, ok := ExtractVirtualKey(c)
		if ok || vk != nil {
			return c.Status(500).SendString("should be missing")
		}
		return c.SendString("ok")
	})

	req := httptest.NewRequest("GET", "/test", nil)
	resp, err := app.Test(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
}

func TestFormatBearer(t *testing.T) {
	result := FormatBearer("sk-test")
	if result != "Bearer sk-test" {
		t.Errorf("expected 'Bearer sk-test', got %s", result)
	}
}
