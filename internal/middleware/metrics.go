// Package middleware 提供 Prometheus Metrics 采集。
package middleware

import (
	"strconv"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// RequestDuration 记录 HTTP 请求耗时。
	RequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_gateway_request_duration_seconds",
		Help:    "HTTP request duration in seconds",
		Buckets: prometheus.DefBuckets,
	}, []string{"method", "path", "status"})

	// RequestTotal 记录 HTTP 请求总数。
	RequestTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_gateway_requests_total",
		Help: "Total number of HTTP requests",
	}, []string{"method", "path", "status"})

	// ActiveRequests 记录当前活跃请求数。
	ActiveRequests = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "llm_gateway_active_requests",
		Help: "Number of active requests",
	})

	// ProviderRequests 记录各 Provider 的请求数。
	ProviderRequests = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_gateway_provider_requests_total",
		Help: "Total requests forwarded to each provider",
	}, []string{"provider", "model"})

	// ProviderErrors 记录各 Provider 的错误数。
	ProviderErrors = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_gateway_provider_errors_total",
		Help: "Total errors from each provider",
	}, []string{"provider", "status_code"})

	// KeyPoolHealth 记录 Key Pool 健康状态。
	KeyPoolHealth = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_gateway_keypool_healthy_keys",
		Help: "Number of healthy keys per provider",
	}, []string{"provider"})
)

// MetricsMiddleware 采集请求指标。
func MetricsMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		ActiveRequests.Inc()
		defer ActiveRequests.Dec()

		start := time.Now()
		err := c.Next()
		duration := time.Since(start).Seconds()

		status := strconv.Itoa(c.Response().StatusCode())
		method := c.Method()
		path := c.Route().Path
		if path == "" {
			path = c.Path()
		}

		RequestDuration.WithLabelValues(method, path, status).Observe(duration)
		RequestTotal.WithLabelValues(method, path, status).Inc()

		return err
	}
}

// RecordProviderRequest 记录 Provider 请求指标。
func RecordProviderRequest(provider, model string) {
	ProviderRequests.WithLabelValues(provider, model).Inc()
}

// RecordProviderError 记录 Provider 错误指标。
func RecordProviderError(provider string, statusCode int) {
	ProviderErrors.WithLabelValues(provider, strconv.Itoa(statusCode)).Inc()
}

// UpdateKeyPoolHealth 更新 Key Pool 健康指标。
func UpdateKeyPoolHealth(provider string, healthyCount int) {
	KeyPoolHealth.WithLabelValues(provider).Set(float64(healthyCount))
}
