# Scope Load Testing

Automated load testing for [Daydream Scope](https://github.com/daydreamlive/scope) cloud inference on the Livepeer network.

## Deploy Anywhere (Docker)

```bash
git clone https://github.com/livepeer/scope-load-testing.git
cd scope-load-testing
echo "DAYDREAM_API_KEY=sk_your_key_here" > .env
docker compose up -d          # starts the scheduler
docker compose logs -f        # watch progress
```

Config, scenarios, and prompts are baked into the image. Only `.env` is needed.

### Docker Commands

```bash
# Scheduler (continuous testing)
docker compose up -d

# Single scenario
docker compose run --rm harness run --scenario longlive_v2v_1m

# List scenarios
docker compose run --rm harness scenarios

# Coverage report
docker compose run --rm harness coverage

# Stop
docker compose down
```

## Run Without Docker

```bash
pip install -e .
export DAYDREAM_API_KEY=sk_your_key_here
loadtest run --scenario longlive_v2v_1m
loadtest scenarios
loadtest schedule
```

## How It Works

```
loadtest CLI → Daydream SDK (sdk.daydream.monster) → Livepeer Orchestrator → Scope Runner (GPU)
```

No local GPU needed. The harness starts streams via the SDK, publishes input frames (for v2v/i2v), captures output, validates quality, switches prompts, and checks for stalls — all over HTTP.

## Scenarios

16 scenarios generated from `config/default.yaml`:

| Pipeline | Modes | Durations |
|----------|-------|-----------|
| longlive | t2v, v2v, i2v | 1m, 5m, 15m |
| ltx2 | t2v, i2v | 1m, 5m |
| longlive+rife | v2v | 5m, 15m |
| depth+longlive+rife | v2v | 5m |

**Adding a new pipeline = one YAML entry in `config/default.yaml`. No code changes.**

## Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| `DAYDREAM_API_KEY` | Yes | - |
| `SDK_URL` | No | `https://sdk.daydream.monster` |

## Test Results (staging, warm runner)

| Scenario | Status | Connect | First Frame |
|----------|--------|---------|-------------|
| longlive v2v 1m | PASS | 12.9s | 0.6s |
| longlive v2v 5m | PASS | 12.9s | 0.2s |
| longlive v2v 15m | PASS | 12.9s | 0.5s |

## Configuration

`config/default.yaml` contains everything: budget, thresholds, scenario matrix. Key settings:

```yaml
budget:
  daily_percent: 20          # % of 24hrs each orchestrator is under test
  max_run_duration_mins: 30  # hard cap per run

scenarios:
  - pipeline: longlive
    modes: [t2v, v2v, i2v]
    durations: [1, 5, 15]
    prompts_pool: nature
```

## Development

```bash
pip install -e ".[dev]"
pytest                                  # 105 tests
pytest --ignore=tests/test_executor.py  # fast tests (<1s)
```
