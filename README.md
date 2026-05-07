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
| `PUSHGATEWAY_URL` | No | `http://pushgateway:9091` (set by docker-compose) |

## Grafana Monitoring Setup

The harness pushes metrics to a Prometheus Push Gateway after each test run. To see data in Grafana:

```
loadtest harness → pushes metrics → Push Gateway (:9091) → Prometheus scrapes → Grafana queries
```

### Step 1: Deploy with docker-compose (includes push gateway)

```bash
echo "DAYDREAM_API_KEY=sk_..." > .env
docker compose up -d    # starts harness + pushgateway
```

The push gateway runs at `http://<your-vm>:9091`.

### Step 2: Add scrape target to your Prometheus

Add this to your Prometheus `prometheus.yml` (the one Grafana already reads from):

```yaml
scrape_configs:
  - job_name: 'scope_loadtest'
    honor_labels: true
    scrape_interval: 30s
    static_configs:
      - targets: ['<your-vm-ip>:9091']
```

Replace `<your-vm-ip>` with the IP or hostname of the machine running docker-compose.

Then reload Prometheus: `curl -X POST http://your-prometheus:9090/-/reload`

### Step 3: Import the Grafana dashboard

Import `dashboards/grafana/scope-loadtest.json` into Grafana. Set the data source to the Prometheus instance from step 2.

### Verify data is flowing

```bash
# Check push gateway has metrics
curl -s http://<your-vm>:9091/metrics | grep scope_loadtest

# Run a test to generate data
docker compose run --rm harness run --scenario longlive_v2v_1m
```

After the test completes, metrics appear in the push gateway within seconds and in Grafana after the next Prometheus scrape (~30s).

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
