from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from litlake.providers.extraction import LocalPdfExtractionProvider
from litlake.storage import FileLocator
from tests.fixtures.pdf_factory import (
    MB,
    build_stall_regression_fixture_set,
    should_run_true_large_fixture,
)


class PdfFixtureFactoryTests(unittest.TestCase):
    def test_build_fixture_set_creates_expected_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_stall_regression_fixture_set(Path(tmp), large_size_bytes=2 * MB)
            names = {spec.name for spec in specs}
            self.assertEqual(
                names,
                {"small_1", "small_2", "medium_1", "medium_2", "large_1", "large_2", "malformed_1"},
            )

            for spec in specs:
                self.assertTrue(spec.path.exists(), msg=f"missing fixture file for {spec.name}")
                if spec.min_size_bytes:
                    self.assertGreaterEqual(spec.path.stat().st_size, spec.min_size_bytes)

    def test_valid_fixture_extracts_and_malformed_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_stall_regression_fixture_set(Path(tmp), large_size_bytes=2 * MB)
            provider = LocalPdfExtractionProvider()

            for spec in specs:
                locator = FileLocator(
                    storage_kind="local",
                    file_path=str(spec.path),
                    storage_uri=str(spec.path),
                )
                if spec.expected_outcome == "success":
                    result = provider.extract(locator)
                    self.assertTrue(result.text.strip(), msg=f"expected non-empty extraction for {spec.name}")
                else:
                    with self.assertRaises(Exception):
                        provider.extract(locator)

    def test_default_large_fixture_exceeds_50mb_when_enabled(self) -> None:
        if not should_run_true_large_fixture():
            self.skipTest("Set LIT_LAKE_RUN_LARGE_FIXTURES=1 to run true >50MB fixture generation")

        with tempfile.TemporaryDirectory() as tmp:
            specs = build_stall_regression_fixture_set(Path(tmp))
            large_specs = [spec for spec in specs if spec.name.startswith("large_")]
            self.assertEqual(len(large_specs), 2)
            for spec in large_specs:
                self.assertGreaterEqual(spec.path.stat().st_size, 50 * MB)


if __name__ == "__main__":
    unittest.main()
