"""Exceptions shared by Praniti's pipeline stages (acquire, preprocess, dedupe).

A specific exception makes the Airflow task fail with a message that says exactly
what went wrong and how to fix it, instead of a generic stack trace.
"""


class DataUnavailableError(RuntimeError):
    """The source data could not be downloaded or found."""


class FileCorruptionError(RuntimeError):
    """A downloaded or written file cannot be read."""


class ManifestMismatchError(RuntimeError):
    """Row counts or checksums differ from what we expect."""


class LabelError(ValueError):
    """Labels are missing, non-numeric, or outside their allowed range."""
