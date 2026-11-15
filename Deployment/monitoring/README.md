# Monitoring (Phase 2 drafts)

Work plan v4, section 5: draft Grafana dashboards and Prometheus alert rules against a mock
exporter, and document the alert thresholds. Phase 3 (v4 section 6) moves the same rules and
dashboard onto the live Cloud Run service (sidecar -> Managed Service for Prometheus ->
Grafana) and routes the retrain alerts to `retrain_dag`.

| File | What it is |
|---|---|
| `mock_service.py` | Mock of the serving API (v4 2.4): `POST /predict`, `GET /health`, and `GET /metrics` with the 2.4 metric names, counted from real traffic, plus any `*.prom` files in `--textfile-dir` (the canary's). Bad input is a 422 (a request, not a service error); any failure on its side is a counted 500. Standard library only |
| `prometheus/alert_rules.yml` | The **active** rules (9): service health and canary. Uses only metrics that exist today; Prometheus loads it |
| `prometheus/alert_rules_planned.yml` | The **planned** rules (7), not loaded yet: latency (waits on the int8 package's p95), the canary retrain rule (waits on the promoted model's reference AUC), and drift (waits on `drift_ingest_dag`'s metrics). All three `action: retrain` rules are here |
| `prometheus/alert_rules_test.yml` | `promtool` unit tests: no flag-rate alert when traffic goes idle, service down fires, a missing canary fires |
| `prometheus/prometheus.yml` | Scrape config for the mock; loads the active rules only |
| `grafana/dashboards/distillate_dashboard.json` | Dashboard: service, decisions, canary, drift (v4 section 6 panels). Panels that need Phase 3 metrics are titled "(planned)"; instances and gate history are placeholders |
| `grafana/provisioning/` | Data source and dashboard provider, for the local stack |
| `docker-compose.yml` | Prometheus and Grafana, for viewing the drafts locally |
| `thresholds.md` | Every threshold and why it is set where it is |
| `tests/` | The mock follows 2.4; active rules and unplanned panels use only metrics that exist (canary names read from `canary.py`); the thresholds doc covers every rule |
| `ruff.toml` | Data-Pipeline's ruff settings, so lint gives the same result from the repo root or here |

CI: `.github/workflows/monitoring-ci.yml` runs ruff, the tests, and `promtool check rules` and
`promtool test rules` on every PR that touches this folder or `Data-Pipeline/scripts/canary.py`.

## Running it locally

From this folder:

```bash
# 1. The mock, serving the canary's metrics file too
python mock_service.py --host 0.0.0.0 --port 8000 --textfile-dir ../../Data-Pipeline/reports/canary

# 2. In another terminal, from Data-Pipeline: replay the canary sets against the mock
#    (127.0.0.1, not localhost: on Windows localhost tries IPv6 first, about 2 s per request)
python -m scripts.canary --url http://127.0.0.1:8000

# 3. Prometheus and Grafana
docker compose up -d
```

Prometheus is at http://localhost:9090 (Status > Rules shows the alert rules loaded), and
Grafana at http://localhost:3000 (admin / admin), dashboard "Distillate: service, canary and
drift". The mock's scores carry no signal, so canary AUCs against it sit near 0.5; the point is
the plumbing, not the numbers.

**Checked end to end:** the canary replayed all 6,000 real canary comments through the mock in
about 4 s with no errors (Wikipedia mean AUC 0.509; flag false positive rate about 11%, as the
mock is designed to flag), and Prometheus scraped the mock and loaded the rules with no errors.

Check and test the rule files with Prometheus's own tool:

```bash
docker run --rm -v "$PWD/prometheus:/p" --entrypoint promtool prom/prometheus:v2.55.1 check rules /p/alert_rules.yml /p/alert_rules_planned.yml
docker run --rm -v "$PWD/prometheus:/p" --entrypoint promtool prom/prometheus:v2.55.1 test rules /p/alert_rules_test.yml
```

## Tests

```bash
python -m pytest tests -v
```

Needs `pytest` and `pyyaml`, plus `pandas`, `numpy` and `scikit-learn` for the test that reads
`Data-Pipeline/scripts/canary.py` (all in Data-Pipeline's `requirements-dev.txt`).
