.PHONY: up down logs build test

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

build:
	docker compose build --no-cache

test:
	docker compose exec api pytest -q


