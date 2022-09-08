PROJECT_NAME = egsis
DOCKER_REGISTRY := ryukinix/egsis
VERSION := latest
UID = $(shell id -u)
GID = $(shell id -g)
DOCKER_RUN = docker run --user $(UID):$(GID) --rm -t -v $(PWD):/tmp -w /tmp


build:
	docker build -t $(PROJECT_NAME) .

publish: build
	docker tag $(PROJECT_NAME) $(DOCKER_REGISTRY):$(VERSION)
	docker push $(DOCKER_REGISTRY):$(VERSION)

check: build
	$(DOCKER_RUN) $(PROJECT_NAME) check
