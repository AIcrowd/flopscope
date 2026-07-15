from scripts.cost_sheet.refs import current_sha, registry_entry_line, permalink

def test_registry_entry_line_finds_matmul():
    line = registry_entry_line("matmul")
    assert isinstance(line, int) and line > 0

def test_registry_entry_line_missing_returns_none():
    assert registry_entry_line("definitely_not_an_op_xyz") is None

def test_permalink_shape():
    url = permalink("src/flopscope/_registry.py", 42, "abc123")
    assert url == "https://github.com/AIcrowd/flopscope/blob/abc123/src/flopscope/_registry.py#L42"

def test_current_sha_is_hex():
    sha = current_sha()
    assert len(sha) >= 7 and all(c in "0123456789abcdef" for c in sha)
