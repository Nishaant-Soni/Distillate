# Teacher shard fixtures

Synthetic shards in the exact format of real teacher shards (`docs/teacher_prompts_and_shard_spec.md` §5), for testing shard validators. No real comments or labels: the ids are made up (they start with `f1x`) and the scores are random.

| Path | What | A correct validator says |
|---|---|---|
| `source_ids.csv` | the 25 source ids (column `id`) | |
| `valid/shard_0000` to `shard_0002` | three shards of 10, 10, and 5 rows (shard size 10), each a `.parquet` plus `.manifest.json` | no errors |
| `broken/` | the same three shards, except `shard_0001.manifest.json` says `row_count` 11 while its Parquet has 10 rows | exactly one error, the row count of `shard_0001` |

Settings for a validator run: shard size 10, `prompt_version` `p2-zs-v2b`, `model_version` `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`.

Checked with the teacher lane's own validator (from `Model-Pipeline/`):

```bash
python -m src.teacher.validate_local --local-root tests/fixtures/teacher --prefix valid/ \
    --ids tests/fixtures/teacher/source_ids.csv --shard-size 10      # exit 0
python -m src.teacher.validate_local --local-root tests/fixtures/teacher --prefix broken/ \
    --ids tests/fixtures/teacher/source_ids.csv --shard-size 10      # exit 1, one FAIL line
```

`tests/teacher/test_fixtures.py` runs the same checks and fails if these files drift from what `make_fixtures.py` writes. To regenerate: `python tests/fixtures/teacher/make_fixtures.py`.
