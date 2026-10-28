import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RAGDockerContextTests(unittest.TestCase):
    def test_root_readmes_are_included_in_docker_build_context(self):
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        lines = {line.strip() for line in dockerignore.splitlines()}
        self.assertIn("README*.md", lines)
        self.assertIn("!README.md", lines)
        self.assertIn("!READMEENG.md", lines)

    def test_root_readmes_are_copied_into_runtime_image(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY README.md READMEENG.md ./", dockerfile)


if __name__ == "__main__":
    unittest.main()
