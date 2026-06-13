.PHONY: build test vet clean run docker

BINARY=gateway
CMD=cmd/gateway/main.go

build:
	go build -o $(BINARY) $(CMD)

test:
	go test -v ./...

vet:
	go vet ./...

clean:
	rm -f $(BINARY)

run: build
	./$(BINARY)

docker:
	docker build -t llm-gateway:latest .

fmt:
	go fmt ./...
