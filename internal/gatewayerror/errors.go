// Package gatewayerror normalizes client-facing error responses.
package gatewayerror

import "github.com/gofiber/fiber/v2"

type ErrorBody struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

func Respond(c *fiber.Ctx, status int, code, message string) error {
	return c.Status(status).JSON(ErrorBody{Error: ErrorDetail{Message: message, Type: "llm_gateway_error", Code: code}})
}
