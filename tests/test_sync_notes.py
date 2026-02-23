from __future__ import annotations

import unittest

from litlake.sync import _note_html_to_text


class SyncNotesTests(unittest.TestCase):
    def test_note_html_to_text_extracts_text(self) -> None:
        html = '<div class="zotero-note znv1"><div><p>Hello note</p><p>Second line</p></div></div>'
        text = _note_html_to_text(html)
        self.assertIsNotNone(text)
        assert text is not None
        self.assertIn("Hello note", text)
        self.assertIn("Second line", text)

    def test_note_html_to_text_empty_or_none(self) -> None:
        self.assertIsNone(_note_html_to_text(None))
        self.assertIsNone(_note_html_to_text("   "))


if __name__ == "__main__":
    unittest.main()
