"""The shard spec is the contract with the other lanes; it must match the code word for word."""

from src.teacher.config import LABELS, PIPELINE_ROOT
from src.teacher.prompts import P1_USER_TEMPLATE, P2_QUESTIONS, P2_USER_TEMPLATE, SYSTEM_MESSAGE
from src.teacher.shards import REQUIRED_MANIFEST

SPEC = (PIPELINE_ROOT / "docs" / "teacher_prompts_and_shard_spec.md").read_text()


def test_prompt_text_matches_the_code():
    assert SYSTEM_MESSAGE in SPEC
    assert P2_USER_TEMPLATE in SPEC
    assert P1_USER_TEMPLATE in SPEC
    assert P2_QUESTIONS["v2b"]["obscene"] in SPEC


def test_labels_and_manifest_fields_are_listed():
    assert ", ".join(f"`{label}`" for label in LABELS) in SPEC
    for field in REQUIRED_MANIFEST:
        assert f"`{field}`" in SPEC
