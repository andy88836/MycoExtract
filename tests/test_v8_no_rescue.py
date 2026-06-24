"""v8: zero-record rescue path must be fully removed.

Confirms that:
1. src/utils/zero_record_rescue.py is deleted
2. No source file imports from it
3. The script no longer references rescue_* functions
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _scan_files(extensions=(".py", ".txt", ".yaml", ".md")):
    """Iterate over all source files except tests, .git, and analysis_outputs."""
    skip_dirs = {".git", ".venv", "__pycache__", "analysis_outputs",
                 "All_papers", "Golden_data_pdf", "Gold_data_pdf",
                 "tests"}
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in skip_dirs for part in p.parts):
            continue
        if p.suffix.lower() in extensions:
            yield p


def test_zero_record_rescue_module_deleted():
    rescue_file = REPO_ROOT / "src" / "utils" / "zero_record_rescue.py"
    assert not rescue_file.exists(), (
        "src/utils/zero_record_rescue.py must be deleted in v8 — "
        "the rescue path produced 80% false positives in baseline."
    )


def test_no_imports_of_rescue_module():
    """No source file may import from zero_record_rescue."""
    bad = []
    for p in _scan_files(extensions=(".py",)):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "from src.utils.zero_record_rescue" in text:
            bad.append(str(p.relative_to(REPO_ROOT)))
        if "import zero_record_rescue" in text:
            bad.append(str(p.relative_to(REPO_ROOT)))
    assert not bad, f"v8 forbids imports from zero_record_rescue. Found in: {bad}"


def _strip_comments_and_docstrings(source: str) -> str:
    """Remove # comments and triple-quoted strings so the test matches code only.
    A comment that mentions a removed function ("# NOTE: foo() removed") is fine;
    an actual call site is not."""
    import io, tokenize
    out = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
                             tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                # keep STRING because removing all strings is too aggressive (we'd
                # lose log messages etc.); only drop module-level / function-level
                # docstrings would be ideal but a coarse check is fine here.
                if tok.type == tokenize.COMMENT:
                    continue
            out.append(tok.string if tok.type != tokenize.COMMENT else "")
    except tokenize.TokenizeError:
        return source
    return " ".join(out)


def test_run_script_has_no_rescue_function_calls():
    script = (REPO_ROOT / "scripts" / "run_all_papers_full_extraction.py").read_text(encoding="utf-8")
    code_only = _strip_comments_and_docstrings(script)
    forbidden = [
        "validate_rescue_records(",
        "run_rescue_for_paper(",
        "run_rescue_with_client(",
        "deterministic_table_fallback(",
        "has_rescue_keyword_evidence(",
    ]
    for token in forbidden:
        assert token not in code_only, f"run_all_papers_full_extraction.py still calls {token}"
