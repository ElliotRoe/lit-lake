from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from litlake.providers.extraction import extract_html_text


class ExtractHtmlTextTests(unittest.TestCase):
    def test_fragment_skips_trafilatura_and_uses_fallback(self) -> None:
        fake_extract = Mock(return_value="unexpected")
        fake_module = SimpleNamespace(extract=fake_extract)
        html = '<div class="zotero-note znv1"><p>Hello note</p><p>Second line</p></div>'

        with patch.dict("sys.modules", {"trafilatura": fake_module}):
            result = extract_html_text(html, require_trafilatura=False)

        self.assertEqual(result.metadata, {"mode": "html_fallback"})
        self.assertIn("Hello note", result.text)
        self.assertIn("Second line", result.text)
        fake_extract.assert_not_called()

    def test_full_html_uses_trafilatura_when_available(self) -> None:
        fake_extract = Mock(return_value="Hello from trafilatura")
        fake_module = SimpleNamespace(extract=fake_extract)
        html = "<html><body><article><p>Hello from doc</p></article></body></html>"

        with patch.dict("sys.modules", {"trafilatura": fake_module}):
            result = extract_html_text(html, require_trafilatura=False)

        self.assertEqual(result.metadata, {"mode": "trafilatura_markdown"})
        self.assertEqual(result.text, "Hello from trafilatura")
        fake_extract.assert_called_once_with(html, output_format="markdown")


if __name__ == "__main__":
    unittest.main()
