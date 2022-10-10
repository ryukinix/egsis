PROJECT_NAME = egsis
DOCKER_REGISTRY := ryukinix/egsis
VERSION := latest
UID = $(shell id -u)
GID = $(shell id -g)
DOCKER = docker
ifeq ($(DOCKER), podman)
	DOCKER_FLAGS=--userns keep-id
else
	DOCKER_FLAGS=
endif

DOCKER_RUN = $(DOCKER) run $(DOCKER_FLAGS) \
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
	$(DOCKER) pull $(DOCKER_REGISTRY)

build:
	$(DOCKER) build -t $(PROJECT_NAME) .

publish: build
	$(DOCKER) tag $(PROJECT_NAME) $(DOCKER_REGISTRY):$(VERSION)
	$(DOCKER) push $(DOCKER_REGISTRY):$(VERSION)

check: build
	$(DOCKER_RUN) $(PROJECT_NAME) check
	sed -i "s|/app|$(PWD)|g" tests/coverage.xml


lint: build
	$(DOCKER_RUN) $(PROJECT_NAME) lint egsis/ tests/

notebook: build
	$(DOCKER_RUN) -i $(MOUNT_NOTEBOOK) $(EXPOSE_PORT) $(PROJECT_NAME) jupyter lab
