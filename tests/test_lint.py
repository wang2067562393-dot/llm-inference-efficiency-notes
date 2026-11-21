import json
import shutil
import tempfile
import unittest
from pathlib import Path

from evalkit.cli import main as cli_main
from evalkit.lint import format_report, lint_task

FIXTURES = Path(__file__).parent / "fixtures"


def rules(findings):
    return {f.rule for f in findings}


class TestLintGoodTask(unittest.TestCase):
    def test_good_task_has_no_high_findings(self):
        findings = lint_task(FIXTURES / "good_task")
        high = [f for f in findings if f.severity == "high"]
        self.assertEqual(high, [], f"干净的任务包不应有 high 问题：{high}")

    def test_good_task_has_no_leak_rules(self):
        findings = lint_task(FIXTURES / "good_task")
        self.assertFalse({"R006", "R007", "R008"} & rules(findings))


class TestLintLeakyTask(unittest.TestCase):
    def setUp(self):
        self.findings = lint_task(FIXTURES / "leaky_task")

    def test_detects_gold_file_in_environment(self):
        self.assertIn("R006", rules(self.findings))

    def test_detects_solution_copied_into_environment(self):
        self.assertIn("R007", rules(self.findings))

    def test_detects_missing_test_entry(self):
        self.assertIn("R005", rules(self.findings))

    def test_high_findings_are_reported(self):
        self.assertTrue(all(f.message for f in self.findings))
        self.assertTrue(any(f.severity == "high" for f in self.findings))


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "task"
        shutil.copytree(FIXTURES / "good_task", self.root)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_manifest(self, path, sha):
        (self.root / "manifest.json").write_text(
            json.dumps({"files": [{"path": "instruction.md", "sha256": sha}]}),
            encoding="utf-8",
        )

    def test_manifest_hash_mismatch_is_high(self):
        self._write_manifest("instruction.md", "0" * 64)
        findings = lint_task(self.root)
        self.assertIn("R008", rules(findings))
        self.assertTrue(
            any(f.rule == "R008" and f.severity == "high" for f in findings)
        )

    def test_manifest_missing_file_is_high(self):
        (self.root / "manifest.json").write_text(
            json.dumps({"files": {"nope.txt": "0" * 64}}), encoding="utf-8"
        )
        findings = lint_task(self.root)
        self.assertTrue(any(f.rule == "R008" and "不存在" in f.message for f in findings))

    def test_manifest_correct_hash_passes(self):
        from evalkit.lint import _sha256

        digest = _sha256(self.root / "instruction.md")
        self._write_manifest("instruction.md", "sha256:" + digest)
        findings = lint_task(self.root)
        self.assertFalse(any(f.rule == "R008" and f.severity == "high" for f in findings))

    def test_unparsable_task_toml_is_high(self):
        (self.root / "task.toml").write_text("this is not toml =\n", encoding="utf-8")
        findings = lint_task(self.root)
        self.assertTrue(any(f.rule == "R002" and f.severity == "high" for f in findings))


class TestCliAndFormat(unittest.TestCase):
    def test_format_report_pass(self):
        self.assertIn("PASS", format_report([]))

    def test_format_report_counts(self):
        text = format_report(lint_task(FIXTURES / "leaky_task"))
        self.assertIn("合计", text)
        self.assertIn("high", text)

    def test_cli_exit_code_good(self):
        self.assertEqual(cli_main(["lint", str(FIXTURES / "good_task")]), 0)

    def test_cli_exit_code_leaky(self):
        self.assertEqual(cli_main(["lint", str(FIXTURES / "leaky_task")]), 1)

    def test_cli_json_output(self):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            cli_main(["lint", str(FIXTURES / "leaky_task"), "--json"])
        payload = json.loads(buf.getvalue())
        self.assertIsInstance(payload, list)
        self.assertIn("rule", payload[0])

    def test_non_directory_rejected(self):
        with self.assertRaises(ValueError):
            lint_task(FIXTURES / "does-not-exist")


if __name__ == "__main__":
    unittest.main()
