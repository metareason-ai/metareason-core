"""Tests for file: reference resolution in spec loading."""

import pytest

from metareason.pipeline.loader import (
    _resolve_file_reference,
    _resolve_file_references,
    load_spec,
    load_calibrate_spec,
)


class TestResolveFileReference:
    def test_resolves_file_prefix(self, tmp_path):
        (tmp_path / "data.txt").write_text("hello world")
        result = _resolve_file_reference("file:data.txt", tmp_path)
        assert result == "hello world"

    def test_strips_whitespace(self, tmp_path):
        (tmp_path / "data.txt").write_text("  hello  \n")
        result = _resolve_file_reference("file:data.txt", tmp_path)
        assert result == "hello"

    def test_passthrough_without_prefix(self, tmp_path):
        result = _resolve_file_reference("normal string", tmp_path)
        assert result == "normal string"

    def test_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "data.txt").write_text("from subdir")
        result = _resolve_file_reference("file:sub/data.txt", tmp_path)
        assert result == "from subdir"

    def test_rejects_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="resolves outside"):
            _resolve_file_reference("file:../../etc/passwd", tmp_path)

    def test_rejects_absolute_path(self, tmp_path):
        with pytest.raises(ValueError, match="resolves outside"):
            _resolve_file_reference("file:/etc/passwd", tmp_path)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _resolve_file_reference("file:missing.txt", tmp_path)


class TestResolveFileReferencesRecursive:
    def test_resolves_in_dict(self, tmp_path):
        (tmp_path / "a.txt").write_text("content A")
        data = {"key": "file:a.txt", "other": "plain"}
        result = _resolve_file_references(data, tmp_path)
        assert result == {"key": "content A", "other": "plain"}

    def test_resolves_in_list(self, tmp_path):
        (tmp_path / "a.txt").write_text("content A")
        data = ["file:a.txt", "plain"]
        result = _resolve_file_references(data, tmp_path)
        assert result == ["content A", "plain"]

    def test_resolves_nested(self, tmp_path):
        (tmp_path / "t.txt").write_text("template content")
        (tmp_path / "r.txt").write_text("rubric content")
        data = {
            "pipeline": [{"template": "file:t.txt"}],
            "oracles": {"judge": {"rubric": "file:r.txt"}},
        }
        result = _resolve_file_references(data, tmp_path)
        assert result["pipeline"][0]["template"] == "template content"
        assert result["oracles"]["judge"]["rubric"] == "rubric content"

    def test_leaves_non_strings_alone(self, tmp_path):
        data = {"count": 42, "flag": True, "rate": 0.5, "empty": None}
        result = _resolve_file_references(data, tmp_path)
        assert result == data


class TestLoadSpecFileReferences:
    def _write_spec(self, tmp_path, template="Hello", rubric="Score 1-5"):
        spec_yaml = tmp_path / "spec.yaml"
        spec_yaml.write_text(
            f"""
spec_id: test
pipeline:
  - template: "{template}"
    adapter:
      name: ollama
    model: llama2
    temperature: 0.7
    top_p: 0.9
    max_tokens: 100
sampling:
  method: latin_hypercube
  optimization: maximin
n_variants: 1
oracles:
  judge:
    type: llm_judge
    model: llama2
    adapter:
      name: ollama
    rubric: "{rubric}"
"""
        )
        return spec_yaml

    def test_resolves_template_and_rubric(self, tmp_path):
        (tmp_path / "tmpl.txt").write_text("Explain {{ topic }}")
        (tmp_path / "rubric.txt").write_text("Rate 1-5")
        spec_yaml = self._write_spec(
            tmp_path, template="file:tmpl.txt", rubric="file:rubric.txt"
        )
        spec = load_spec(spec_yaml)
        assert spec.pipeline[0].template == "Explain {{ topic }}"
        assert spec.oracles["judge"].rubric == "Rate 1-5"

    def test_inline_strings_unchanged(self, tmp_path):
        spec_yaml = self._write_spec(tmp_path)
        spec = load_spec(spec_yaml)
        assert spec.pipeline[0].template == "Hello"
        assert spec.oracles["judge"].rubric == "Score 1-5"

    def test_rejects_traversal(self, tmp_path):
        spec_yaml = self._write_spec(
            tmp_path, template="file:../../etc/passwd"
        )
        with pytest.raises(ValueError, match="resolves outside"):
            load_spec(spec_yaml)

    def test_file_not_found(self, tmp_path):
        spec_yaml = self._write_spec(tmp_path, template="file:missing.txt")
        with pytest.raises(FileNotFoundError):
            load_spec(spec_yaml)
