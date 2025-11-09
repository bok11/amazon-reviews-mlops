import asyncio
import time
import httpx
import statistics
import pytest
import pytest_asyncio
import pandas as pd
from pathlib import Path

API = "http://127.0.0.1:8000/predict"
CONCURRENCY = 20  # we do 20 concurrent connections
P99_THRESHOLD_MS = 300  # SLO target of 300ms


# the basic function calling the endpoint, and timing the response time.
async def worker(client, texts, results):
    for text in texts:
        t0 = time.perf_counter()
        r = await client.post(API, json={"review": text})
        r.raise_for_status()
        results.append((time.perf_counter() - t0) * 1000)  # ms


# helper function to call our worker function concurrently
async def run_load_test(texts, concurrency):
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        chunks = [texts[i::concurrency] for i in range(concurrency)]
        await asyncio.gather(*(worker(client, chunk, results) for chunk in chunks))
    return results


@pytest.mark.asyncio
async def test_model_latency():
    # load the dataset, same as in our training script
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "../data" / "Books_10k.jsonl"

    df = pd.read_json(data_path, lines=True)
    texts = df["text"].astype(str)

    # warm up, we did load the model on startup when writing the runtime but
    # it has little cost to do this extra check
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(API, json={"review": "warmup text"})

    # Now we run the test
    results = await run_load_test(texts, CONCURRENCY)

    # Print performance for the user or the CI system
    results.sort()
    avg = statistics.mean(results)
    p99 = results[int(len(results) * 0.99)]

    print(f"\n--- Performance Results ---")
    print(f"Samples: {len(results)}")
    print(f"Average latency: {avg:.2f} ms")
    print(f"p99 latency: {p99:.2f} ms")

    # Assert that we actually made it.
    assert p99 < P99_THRESHOLD_MS, f"p99 too high: {p99:.1f} ms ≥ {P99_THRESHOLD_MS} ms"
