# Scope Load Testing

Automated load testing for [Daydream Scope](https://github.com/daydreamlive/scope) cloud inference on the Livepeer network.

## Quick Start

```bash
git clone https://github.com/livepeer/scope-load-testing.git
cd scope-load-testing
pip install -e .

# Run a single test
export DAYDREAM_API_KEY=sk_your_key_here
loadtest run --scenario longlive_v2v_1m

# List all scenarios
loadtest scenarios
```

That's it. No GPU, no Docker, no Scope installation needed. The harness connects to the Daydream SDK service which provisions remote GPU inference on the Livepeer network — the same path real users take.

## How It Works

```
loadtest CLI → Daydream SDK (sdk.daydream.monster) → Livepeer Orchestrator → Scope Runner (GPU)
```

The harness starts a stream, publishes input frames (for v2v/i2v), captures output frames, validates quality, switches prompts, and checks for stalls — all via HTTP.

## CLI Commands

```bash
# Run a single scenario
loadtest run --scenario longlive_v2v_5m

# List available scenarios
loadtest scenarios

# Start the scheduler daemon (continuous testing)
loadtest schedule

# Show test coverage for today
loadtest coverage

# Show performance baselines
loadtest baselines
```

## Configuration

All config lives in `config/default.yaml`. The scenario matrix generates 16 test combinations from 4 entries:

```yaml
scenarios:
  - pipeline: longlive
    modes: [t2v, v2v, i2v]
    durations: [1, 5, 15]     # minutes
    prompts_pool: nature
    # ...
```

**Adding a new pipeline = one YAML entry. No code changes.**

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DAYDREAM_API_KEY` | Yes | - | Daydream API key (`sk_...`) |
| `SDK_URL` | No | `https://sdk.daydream.monster` | SDK service URL |

## Scenarios

16 scenarios covering:

| Pipeline | Modes | Durations |
|----------|-------|-----------|
| longlive | t2v, v2v, i2v | 1m, 5m, 15m |
| ltx2 | t2v, i2v | 1m, 5m |
| longlive+rife | v2v | 5m, 15m |
| depth+longlive+rife | v2v | 5m |

## Docker

```bash
cp .env.example .env   # add your API key
docker compose up -d    # starts the scheduler
docker compose logs -f  # watch progress
```

## Test Results (staging)

| Scenario | Status | Connect | First Frame | Duration |
|----------|--------|---------|-------------|----------|
| longlive v2v 1m | PASS | 12.9s | 0.6s | 60s |
| longlive v2v 5m | PASS | 12.9s | 0.2s | 300s |
| longlive v2v 15m | PASS | 12.9s | 0.5s | 900s |

## Development

```bash
pip install -e ".[dev]"
pytest                                  # all tests (113 total)
pytest --ignore=tests/test_executor.py  # fast tests only (<1s)
```

## Project Structure

```
src/loadtest/
├── cli.py            # CLI commands
├── config.py         # YAML config loading
├── scenarios.py      # Scenario matrix expansion
├── sdk_client.py     # Daydream SDK HTTP client
├── sdk_executor.py   # Test lifecycle via SDK
├── scope_client.py   # Direct Scope HTTP client (for local testing)
├── executor.py       # Test lifecycle via direct Scope
├── scheduler.py      # Budget planning and orchestrator rotation
├── discovery.py      # Orchestrator discovery and health
├── coverage.py       # Per-orchestrator coverage tracking
├── metrics.py        # Prometheus metrics and push
├── validators.py     # Frame quality and prompt sensitivity
├── results.py        # Error taxonomy and log capture
└── regression.py     # Rolling baselines and drift detection
```
