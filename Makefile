SERVICE := omr-dev
COMPOSE := docker compose

.PHONY: help build up down restart shell install logs ps gpu gpu-test image-size clean

help:
	@echo "Targets:"
	@echo "  make build    - build the dev image"
	@echo "  make up       - start the dev container in the background"
	@echo "  make down     - stop and remove the dev container"
	@echo "  make restart  - down + up"
	@echo "  make shell    - open a bash shell inside the running container"
	@echo "  make install    - re-install requirements.txt (root; needed if you edit it without rebuilding)"
	@echo "  make logs       - tail container logs"
	@echo "  make ps         - show container status"
	@echo "  make gpu        - run nvidia-smi inside the container (verifies GPU passthrough)"
	@echo "  make gpu-test   - run a small torch GPU op (verifies sm_120 kernels are present)"
	@echo "  make image-size - print the size of the built image"
	@echo "  make clean      - down, remove volumes, prune dangling images (DESTRUCTIVE)"

build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

shell:
	$(COMPOSE) exec $(SERVICE) bash

# Runs as root because deps live in /opt/conda; see Dockerfile.
install:
	$(COMPOSE) exec -u root $(SERVICE) uv pip install --system -r requirements.txt

logs:
	$(COMPOSE) logs -f $(SERVICE)

ps:
	$(COMPOSE) ps

gpu:
	$(COMPOSE) exec $(SERVICE) nvidia-smi

gpu-test:
	$(COMPOSE) exec $(SERVICE) python -c "import torch; x = torch.randn(1024, 1024, device='cuda'); print('OK', (x @ x).sum().item())"

image-size:
	@docker image ls omr-jianpu:dev --format 'Image: {{.Repository}}:{{.Tag}}  Size: {{.Size}}'

clean:
	@echo "This will remove the container, its volumes, and dangling images."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ] || exit 1
	$(COMPOSE) down -v
	docker system prune -f
