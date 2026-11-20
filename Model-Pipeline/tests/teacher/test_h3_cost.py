import json

import pytest

from src.teacher.config import load_config
from src.teacher.h3_cost import at_rate, build, hourly_price, per_million, run

PRICE = {"region": "us-east1", "currency": "USD", "checked": "2026-11-19", "source": "test",
         "gpu": {"name": "nvidia-tesla-t4", "count": 1, "per_hour": 0.35, "sku": "g"},
         "machine": {"name": "n1-standard-4", "vcpus": 4, "memory_gib": 15,
                     "per_vcpu_hour": 0.031611, "per_gib_hour": 0.004237,
                     "vcpu_sku": "c", "ram_sku": "r"}}


def sweep(sustainable: int | None = 6):
    def one(rate, p95, keeps_up):
        return {"offered_rate": rate, "sent": rate * 300, "seconds": 300, "achieved_rate": rate,
                "latency_p50_s": 0.5, "latency_p95_s": p95, "latency_p99_s": p95 + 0.5, "errors": 0,
                "gpu_util_mean": 50.0, "keeps_up": keeps_up}
    return {"sustainable_rate": sustainable, "keeps_up_rule": "rule",
            "runs": [one(5, 1.2, True), one(6, 1.9, True), one(8, 25.0, False)],
            "saturation": {"achieved_rate": 7.5, "concurrency": 64, "latency_p95_s": 9.0},
            "environment": {"model_version": "m", "prompt_version": "p", "vllm_version": "v",
                            "gpu": "Tesla T4", "code_commit": "abc"},
            "parity_with_offline": {"max_abs_p": 0.007}, "texts": {"comments": 2000}}


def test_n1_standard_4_plus_t4_per_hour():
    hourly = hourly_price(PRICE)
    assert hourly["machine"] == pytest.approx(4 * 0.031611 + 15 * 0.004237, abs=1e-4)
    assert hourly["total"] == 0.54


def test_cost_per_million_comments():
    # $0.54/h at 6 comments/s = 21,600 comments/h -> $25 per million
    assert per_million(0.54, 6) == pytest.approx(25.0)


def test_instances_round_up_and_cost_counts_idle_capacity():
    out = at_rate(7, sustainable=6, per_hour=0.54)
    assert out["instances"] == 2 and out["utilization"] == pytest.approx(7 / 12, abs=1e-4)
    assert out["cost_per_million_comments"] == pytest.approx(2 * 0.54 / (7 * 3600) * 1e6,
                                                             abs=0.01)
    assert at_rate(6, 6, 0.54)["instances"] == 1


def test_bad_target_rate():
    with pytest.raises(ValueError):
        at_rate(0, 6, 0.54)


def test_report_without_a_target_rate():
    report = build(sweep(), PRICE, None)
    assert report["at_target_rate"] is None
    p = report["per_instance"]
    assert p["sustainable_rate"] == 6 and p["latency_at_sustainable_s"]["p95"] == 1.9
    assert p["cost_per_million_comments"] == pytest.approx(25.0, abs=0.01)
    assert p["cost_per_million_at_max_throughput"] < p["cost_per_million_comments"]
    assert [r["keeps_up"] for r in report["runs"]] == [True, True, False]
    assert report["price"]["skus"] == {"gpu": "g", "vcpu": "c", "ram": "r"}
    assert not any("note" in r for r in report["runs"])


def test_a_miss_from_low_arrivals_is_noted():
    s = sweep()
    # 3/s nominal, but the seed sent 2.81/s; the server answered all of them quickly.
    s["runs"].insert(0, {"offered_rate": 3, "sent": 844, "seconds": 300, "achieved_rate": 2.815,
                         "latency_p50_s": 0.25, "latency_p95_s": 0.57, "latency_p99_s": 0.74,
                         "errors": 0, "keeps_up": False})
    runs = build(s, PRICE, None)["runs"]
    assert "2.81/s against 3/s" in runs[0]["note"]
    assert runs[0]["cost_per_million_one_instance_at_this_rate"] == pytest.approx(50.0)
    assert "note" not in runs[-1]      # 8/s: a real miss (p95 25 s)


def test_run_costs_are_at_the_nominal_rate_and_none_where_it_does_not_keep_up():
    s = sweep()
    s["runs"][1]["achieved_rate"] = 6.13    # Poisson arrivals ran a little above 6/s
    runs = build(s, PRICE, None)["runs"]
    costs = [r["cost_per_million_one_instance_at_this_rate"] for r in runs]
    assert costs[:2] == [pytest.approx(30.0), pytest.approx(25.0)]
    assert costs[2] is None                 # 8/s: one instance can't serve it


def test_no_sustainable_rate_is_an_error():
    with pytest.raises(ValueError):
        build(sweep(sustainable=None), PRICE, None)


def test_run_writes_the_report(tmp_path):
    cfg = load_config()
    (tmp_path / "h3").mkdir()
    (tmp_path / "h3" / "h3_sweep.json").write_text(json.dumps(sweep()))
    cfg["paths"]["h3_dir"] = str(tmp_path / "h3")
    cfg["h3"] = {**cfg["h3"], "report": str(tmp_path / "out" / "h3.json"), "price": PRICE}
    report = run(cfg, target_rate=3)
    assert json.loads((tmp_path / "out" / "h3.json").read_text()) == report
    assert report["at_target_rate"]["instances"] == 1


def test_committed_config_prices():
    price = load_config()["h3"]["price"]
    assert price["region"] == "us-east1" and price["gpu"]["count"] == 1
    assert hourly_price(price)["total"] == pytest.approx(0.54, abs=0.001)
