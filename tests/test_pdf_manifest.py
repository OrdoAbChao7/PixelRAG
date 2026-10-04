"""The PDF manifest's `complete` flag must mean what the URL manifests' does.

#139 made the case that a manifest which always claims completeness is worse
than no flag at all: a consumer cannot tell a whole capture from a fragment,
so it reads the fragment as the whole. #141 fixed that for the two URL
backends. The PDF backend writes the same key and still hardcodes it, so a
caller that asked for a subset of pages gets a directory that claims to be the
entire document.
"""

import json
import sys
from types import ModuleType
from pathlib import Path

import pytest
from PIL import Image
from pixelrag_render import render_pdf


@pytest.fixture
def three_page_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "doc.pdf"
    pages = [Image.new("RGB", (612, 792), "white") for _ in range(3)]
    pages[0].save(pdf, save_all=True, append_images=pages[1:])
    fake_pdf2image = ModuleType("pdf2image")
    fake_pdf2image.convert_from_path = lambda **kwargs: pages[
        kwargs.get("first_page", 1) - 1 : kwargs.get("last_page", len(pages))
    ]
    monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
    return pdf


def _manifest(tile_dirs):
    return json.loads((Path(tile_dirs[0]) / "tiles.json").read_text())


def test_pdf_manifest_reports_a_full_render_as_complete(three_page_pdf, tmp_path):
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50)
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 3
    assert manifest["complete"] is True
    assert manifest["requested_pages"] is None


def test_pdf_manifest_reports_a_page_subset_as_incomplete(three_page_pdf, tmp_path):
    """The whole point of the flag: two of three pages is not the document."""
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[1, 3])
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 2, f"test setup rendered wrong: {manifest}"
    assert manifest["complete"] is False
    assert manifest["requested_pages"] == [1, 3]


def test_pdf_manifest_records_a_full_page_range_as_incomplete(three_page_pdf, tmp_path):
    """An explicit range that happens to cover everything is still a request.

    The caller asked for specific pages; the backend has not checked them
    against the document's length, so it must not claim to have captured the
    document.
    """
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[1, 2, 3])
    manifest = _manifest(dirs)

    assert len(manifest["tiles"]) == 3
    assert manifest["complete"] is False
    assert manifest["requested_pages"] == [1, 2, 3]


def test_pdf_subset_preserves_original_page_numbers(three_page_pdf, tmp_path):
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[3])
    manifest = _manifest(dirs)
    chunks = json.loads((Path(dirs[0]) / "chunks.json").read_text())

    assert manifest["tiles"] == ["tile_0003.jpg"]
    assert chunks["chunks"][0]["tile_index"] == 3
    assert (Path(dirs[0]) / "tile_0003.jpg").exists()


def test_sparse_pdf_subset_keeps_selected_page_number_gaps(three_page_pdf, tmp_path):
    dirs = render_pdf(three_page_pdf, tmp_path / "out", dpi=50, pages=[1, 3])
    manifest = _manifest(dirs)
    chunks = json.loads((Path(dirs[0]) / "chunks.json").read_text())

    assert manifest["tiles"] == ["tile_0001.jpg", "tile_0003.jpg"]
    assert [chunk["tile_index"] for chunk in chunks["chunks"]] == [1, 3]
    assert not (Path(dirs[0]) / "tile_0002.jpg").exists()
