"""Column names shared by acquire, preprocess and dedupe.

Keeping them in one place means a renamed column only has to change here.
"""

LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack"]

# Kaggle "jigsaw-toxic-comment-classification-challenge" -> our names.
WIKI_RENAME = {
    "comment_text": "text",
    "toxic": "toxicity",
    "severe_toxic": "severe_toxicity",
    "identity_hate": "identity_attack",
}

# The 24 identity columns in Kaggle "jigsaw-unintended-bias-in-toxicity-classification".
IDENTITY_COLUMNS = [
    "asian", "atheist", "bisexual", "black", "buddhist", "christian", "female",
    "heterosexual", "hindu", "homosexual_gay_or_lesbian",
    "intellectual_or_learning_disability", "jewish", "latino", "male", "muslim",
    "other_disability", "other_gender", "other_race_or_ethnicity", "other_religion",
    "other_sexual_orientation", "physical_disability", "psychiatric_or_mental_illness",
    "transgender", "white",
]

# Civil Comments label we read but don't use (dropped in preprocess, per labels config).
CIVIL_EXTRA_LABELS = ["sexual_explicit"]

# Raw files (data/raw/...), using our names.
WIKI_RAW_COLUMNS = ["id", "text", *LABELS]
CIVIL_RAW_COLUMNS = [
    "id", "text", "created_date", "publication_id", "article_id",
    *LABELS, *CIVIL_EXTRA_LABELS, "identity_annotator_count", *IDENTITY_COLUMNS,
]

# Processed files (data/processed/...), matching CONTRACTS.md section 2.2,
# plus the three derived columns that every processed file carries.
DERIVED_COLUMNS = ["char_len", "has_identity_term", "has_obfuscation"]
WIKI_CLEAN_COLUMNS = ["id", "text", *LABELS, *DERIVED_COLUMNS]
CIVIL_CLEAN_COLUMNS = [
    "id", "text", "created_date", "publication_id", "article_id",
    *LABELS, *[f"{label}_frac" for label in LABELS], *DERIVED_COLUMNS,
]
# Only the identities file keeps the identity columns (contract: bias evaluation only).
CIVIL_IDENTITY_EXTRA_COLUMNS = ["identity_annotator_count", *IDENTITY_COLUMNS]
