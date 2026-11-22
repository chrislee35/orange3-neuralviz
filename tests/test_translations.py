import json
import re
from pathlib import Path

import pytest

I18N = Path(__file__).resolve().parents[1] / "orangecontrib" / "neuralviz" / "i18n"
LANGUAGES = ["Slovenian", "Spanish", "Japanese"]


def _load(name: str) -> list[str]:
    return json.loads((I18N / f"{name}.json").read_text(encoding="utf-8"))


def _placeholders(text: str) -> list[str]:
    return re.findall(r"\{[^}]*\}", text)


@pytest.mark.parametrize("language", LANGUAGES)
def test_translation_matches_english(language: str) -> None:
    english, translated = _load("English"), _load(language)
    assert len(english) == len(translated)
    assert translated[1] == language and translated[0] != english[0]
    for i, (en, tr) in enumerate(zip(english[2:], translated[2:], strict=True), start=2):
        assert tr.strip(), f"message {i} ({en!r}) is empty"
        assert _placeholders(en) == _placeholders(tr), f"message {i} ({en!r}): placeholders differ"


def test_source_indexes_match_english() -> None:
    """Every ``_tr.m[i, "text"]`` in the source quotes the English message stored at index ``i``."""
    english = _load("English")
    root = I18N.parent
    seen = 0
    for path in root.rglob("*.py"):
        for match in re.finditer(r'_tr\.m\[(\d+), "((?:[^"\\]|\\.)*)"\]', path.read_text(encoding="utf-8")):
            assert english[int(match.group(1))] == match.group(2), f"{path.name}: {match.group(0)}"
            seen += 1
    assert seen > 40


@pytest.mark.parametrize("language", LANGUAGES)
def test_widget_loads_in_language(language: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from orangecanvas.localization import Translator

    monkeypatch.setattr("orangecanvas.localization.DEFAULT_LANGUAGE", language)
    assert Translator("orangecontrib.neuralviz").m[0] == _load(language)[0]
