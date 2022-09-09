PROJECT_NAME = egsis
DOCKER_REGISTRY := ryukinix/egsis
VERSION := latest
UID = $(shell id -u)
GID = $(shell id -g)
DOCKER_RUN = docker run \
					--user $(UID):$(GID) \
					-e HOME=/tmp --rm \
					-t \
					-v $(PWD)/tests:/app/tests \
					-w /app
MOUNT_NOTEBOOK = -v $(PWD)/notebooks:/app/notebooks
EXPOSE_PORT = --net=host


install: # install locally
	python -m venv .venv
	source .venv/bin/activate
	pip install -U pdm setuptools wheel
	pdm install

run: build
	$(DOCKER_RUN) $(PROJECT_NAME)

pull:
	docker pull $(DOCKER_REGISTRY)

build:
	docker build -t $(PROJECT_NAME) .

publish: build
	docker tag $(PROJECT_NAME) $(DOCKER_REGISTRY):$(VERSION)
	docker push $(DOCKER_REGISTRY):$(VERSION)

check: build
	$(DOCKER_RUN) $(PROJECT_NAME) check
	sed -i "s|/app|$(PWD)|g" tests/coverage.xml


lint: build
	$(DOCKER_RUN) $(PROJECT_NAME) lint egsis/ tests/

notebook: build
	$(DOCKER_RUN) -i $(MOUNT_NOTEBOOK) $(EXPOSE_PORT) $(PROJECT_NAME) jupyter lab
