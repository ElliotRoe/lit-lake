from __future__ import annotations

import unittest

from litlake.text_utils import map_chunks_to_page_ranges


class ChunkPageRangeTests(unittest.TestCase):
    def test_maps_chunks_to_single_pages(self) -> None:
        chunks = ["alpha beta", "gamma delta"]
        page_texts = ["alpha beta", "gamma delta"]

        spans = map_chunks_to_page_ranges(chunks, page_texts)

        self.assertEqual(spans, [(1, 1), (2, 2)])

    def test_maps_chunk_across_page_boundary(self) -> None:
        chunks = ["beta\n\ngamma"]
        page_texts = ["alpha beta", "gamma delta"]

        spans = map_chunks_to_page_ranges(chunks, page_texts)

        self.assertEqual(spans, [(1, 2)])

    def test_returns_null_span_when_chunk_not_found(self) -> None:
        chunks = ["missing text"]
        page_texts = ["alpha beta", "gamma delta"]

        spans = map_chunks_to_page_ranges(chunks, page_texts)

        self.assertEqual(spans, [(None, None)])


if __name__ == "__main__":
    unittest.main()
