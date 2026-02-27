from __future__ import annotations

import unittest

from litlake.providers.chunking import chunk_text_with_spans, map_chunk_spans_to_page_ranges


class ChunkPageRangeTests(unittest.TestCase):
    def test_maps_chunk_spans_to_single_pages(self) -> None:
        chunk_spans = [("alpha beta", 0, 10), ("gamma delta", 12, 23)]
        page_texts = ["alpha beta", "gamma delta"]

        spans = map_chunk_spans_to_page_ranges(chunk_spans, page_texts)

        self.assertEqual(spans, [(1, 1), (2, 2)])

    def test_maps_chunk_span_across_page_boundary(self) -> None:
        chunk_spans = [("BETA GAMMA", 6, 17)]
        page_texts = ["alpha beta", "gamma delta"]

        spans = map_chunk_spans_to_page_ranges(chunk_spans, page_texts)

        self.assertEqual(spans, [(1, 2)])

    def test_returns_null_span_when_no_page_texts(self) -> None:
        spans = map_chunk_spans_to_page_ranges([("anything", 0, 8)], [])

        self.assertEqual(spans, [(None, None)])

    def test_chunk_text_with_spans_returns_source_offsets(self) -> None:
        text = "alpha beta.\n\ngamma delta."

        chunks = chunk_text_with_spans(text, target_tokens=2)

        self.assertGreaterEqual(len(chunks), 2)
        first_text, first_start, first_end = chunks[0]
        self.assertEqual(first_text, text[first_start:first_end])
        last_text, last_start, last_end = chunks[-1]
        self.assertEqual(last_text, text[last_start:last_end])


if __name__ == "__main__":
    unittest.main()
