IMAGE       := amazon-review-runtime
CONTAINER	:= amazon-review-runtime-test
PORT		:= 8000

build:
	docker build -t $(IMAGE) .


run: build
	@docker rm -f $(CONTAINER) >/dev/null 2>&1 || true
	docker run -d --rm --name $(CONTAINER) -p $(PORT):$(PORT) $(IMAGE)
	# sleep for a few seconds to give the container time to start
	sleep 2 

test: run
	python3 -m pytest -s
