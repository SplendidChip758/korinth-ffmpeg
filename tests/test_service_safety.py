"""Dependency-free regression tests for deployment and assembly safeguards."""

import ast
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE)
ENV_SOURCE = (ROOT / "korinth-ffmpeg.env.example").read_text(encoding="utf-8")
INSTALL_SOURCE = (ROOT / "install.sh").read_text(encoding="utf-8")


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers


def load_isolated_functions(names, directory):
    functions = [node for node in APP_TREE.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name in names]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "Path": Path,
        "Optional": __import__("typing").Optional,
        "HTTPException": FakeHTTPException,
        "ASSEMBLE_STATE_DIR": directory,
        "SAFE_JOB": re.compile(r"^[A-Za-z0-9_-]{1,64}$"),
        "json": json,
        "os": os,
    }
    exec(compile(module, str(ROOT / "app.py"), "exec"), namespace)
    return namespace


class StateFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp.name) / "_assemble"
        self.api = load_isolated_functions(
            {"_assemble_state_path", "_write_assemble_state", "_read_assemble_state"},
            self.state_dir,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_state_round_trip(self):
        state = {"state": "running", "callback": {"token": "private-test-token"}}
        self.api["_write_assemble_state"]("episode-01", state)
        self.assertEqual(self.api["_read_assemble_state"]("episode-01"), state)

    def test_state_directory_is_private(self):
        self.api["_write_assemble_state"]("episode-01", {"state": "running"})
        self.assertEqual(stat.S_IMODE(self.state_dir.stat().st_mode), 0o700)

    def test_state_file_is_private(self):
        self.api["_write_assemble_state"]("episode-01", {"state": "running"})
        self.assertEqual(stat.S_IMODE((self.state_dir / "episode-01.json").stat().st_mode), 0o600)

    def test_existing_public_directory_is_corrected(self):
        self.state_dir.mkdir(mode=0o755)
        os.chmod(self.state_dir, 0o755)
        self.api["_write_assemble_state"]("episode-01", {"state": "running"})
        self.assertEqual(stat.S_IMODE(self.state_dir.stat().st_mode), 0o700)

    def test_atomic_write_leaves_no_partial_file(self):
        self.api["_write_assemble_state"]("episode-01", {"state": "done"})
        self.assertFalse((self.state_dir / "episode-01.json.part").exists())

    def test_path_traversal_is_rejected(self):
        for job in ("../secret", "nested/path", "", "x" * 65):
            with self.subTest(job=job), self.assertRaises(FakeHTTPException) as raised:
                self.api["_assemble_state_path"](job)
            self.assertEqual(raised.exception.status_code, 400)

    def test_missing_state_returns_none(self):
        self.assertIsNone(self.api["_read_assemble_state"]("missing"))


class ConfigurationTests(unittest.TestCase):
    def test_callback_settings_are_documented(self):
        for name in ("CALLBACK_URL", "CALLBACK_TOKEN", "CALLBACK_TIMEOUT",
                     "CALLBACK_TICK", "CALLBACK_BACKOFF_MAX",
                     "CALLBACK_MAX_AGE_SECONDS", "STATE_RETENTION_SECONDS"):
            with self.subTest(name=name):
                self.assertRegex(ENV_SOURCE, rf"(?m)^{name}=")

    def test_safe_concurrency_default(self):
        self.assertRegex(ENV_SOURCE, r"(?m)^MAX_BACKGROUND_ASSEMBLIES=1$")
        self.assertIn("threading.BoundedSemaphore(MAX_BACKGROUND_ASSEMBLIES)", APP_SOURCE)
        self.assertIn("status_code=429", APP_SOURCE)

    def test_callback_overrides_disabled_by_default(self):
        self.assertRegex(ENV_SOURCE, r"(?m)^ALLOW_CALLBACK_OVERRIDE=false$")
        self.assertIn('parsed.scheme != "https"', APP_SOURCE)
        self.assertIn("CALLBACK_ALLOWED_HOSTS", APP_SOURCE)

    def test_health_does_not_return_callback_tokens(self):
        function = next(node for node in APP_TREE.body
                        if isinstance(node, ast.FunctionDef) and node.name == "health")
        source = ast.get_source_segment(APP_SOURCE, function)
        self.assertIn('"callback_configured": bool(CALLBACK_URL)', source)
        self.assertIn('"callback_authenticated": bool(CALLBACK_TOKEN)', source)
        self.assertNotIn('"callback_token": CALLBACK_TOKEN', source)

    def test_installer_does_not_print_preserved_token(self):
        self.assertNotIn("TOKEN_VALUE=", INSTALL_SOURCE)
        self.assertIn('TOKEN_STATUS="preserved', INSTALL_SOURCE)

    def test_request_body_limit_is_enforced_before_handler(self):
        self.assertIn('request.headers.get("content-length")', APP_SOURCE)
        self.assertIn("status_code=413", APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
