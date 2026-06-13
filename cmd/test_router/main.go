package main

import (
	"fmt"
	"os"

	"llm-gateway/internal/config"
	"llm-gateway/internal/router"
)

func main() {
	cfg, err := config.Load("/etc/llm-gateway/config.yaml")
	if err != nil {
		fmt.Println("Load error:", err)
		os.Exit(1)
	}

	fmt.Println("Kimi ModelMap:", cfg.Providers.Kimi.ModelMap)
	fmt.Println("DeepSeek ModelMap:", cfg.Providers.DeepSeek.ModelMap)

	r := router.NewRouter(cfg, nil)
	
	// Test Route
	result, err := r.Route(nil, "claude-sonnet-4-6")
	if err != nil {
		fmt.Println("Route error:", err)
	} else {
		fmt.Println("Route result:", result.Provider, result.ModelMap)
	}
	
	// Test priorityProvidersForModel
	fmt.Println("Health:", r.HealthStatus())
}
