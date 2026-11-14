# Monitoring (Phase 2 drafts)

Work plan v4, section 5: draft Grafana dashboards and Prometheus alert rules against a mock
exporter, and document the alert thresholds. Phase 3 (v4 section 6) moves the same rules and
dashboard onto the live Cloud Run service (sidecar -> Managed Service for Prometheus ->
Grafana) and routes the retrain alerts to `retrain_dag`.

| File | What it is |
|---|---|
| `mock_service.py` | Mock of the serving API (v4 2.4): `POST /predict`, `GET /health`, and `GET /metrics` with the 2.4 metric names, counted from real traffic, plus any `*.prom` files in `--textfile-dir` (the canary's). Standard library only |
| `prometheus/alert_rules.yml` | 14 alert rules: service health, canary, drift. Three carry `action: retrain` |
| `prometheus/prometheus.yml` | Scrape config for the mock |
| `grafana/dashboards/distillate_dashboard.json` | Dashboard: service, decisions, canary, drift (v4 section 6 panels; instances and gate history are Phase 3 placeholders) |
| `grafana/provisioning/` | Data source and dashboard provider, for the local stack |
| `docker-compose.yml` | Prometheus and Grafana, for viewing the drafts locally |
| `thresholds.md` | Every threshold and why it is set where it is |
| `tests/` | The mock follows 2.4; the rules, dashboard and thresholds doc agree and use only metrics that exist |

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
mock is designed to flag), and `promtool check rules` reports `SUCCESS: 14 rules found`.

Check the rule file with Prometheus's own validator:

```bash
docker run --rm -v "$PWD/prometheus:/p" --entrypoint promtool prom/prometheus:v2.55.1 check rules /p/alert_rules.yml
```

## Tests

```bash
python -m pytest tests -v
```

Needs only `pytest` and `pyyaml` (both in Data-Pipeline's `requirements-dev.txt`).
