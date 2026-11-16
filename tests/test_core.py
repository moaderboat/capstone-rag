"""Run with:  python -m unittest discover tests"""
import unittest

from capstone_rag.chunk import chunk_markdown
from capstone_rag.redact import MASK, find_secrets, redact
from capstone_rag.index import BM25, tokenize


class RedactTests(unittest.TestCase):
    def check(self, text, secret):
        clean, _ = redact(text)
        self.assertNotIn(secret, clean)
        self.assertIn(MASK, clean)

    def test_vault_token(self):
        self.check("root token: hvs.AbCdEfGhIjKlMnOpQrStUvWx12", "hvs.AbCdEfGhIjKlMnOpQrStUvWx12")

    def test_labeled_password_variants(self):
        self.check("Password: Hunter2Hunter2", "Hunter2Hunter2")
        self.check('MQTT_PASSWORD = "Sup3rS3cret!"', "Sup3rS3cret!")
        self.check("--password=`tr0ub4dor&3`", "tr0ub4dor&3")
        self.check("client_secret: abcd1234efgh", "abcd1234efgh")

    def test_backtick_pair(self):
        self.check("local admin `admin`/`Xy9!qwerty` (SSO broken)", "Xy9!qwerty")

    def test_hex_key(self):
        key = "a" * 16 + "0123456789abcdef" * 3
        self.check(f"KEY={key}", key)

    def test_learned_literal_scrubbed_elsewhere(self):
        source = "Password: Gr33nTomato99"
        elsewhere = "| Mirth | https://10.30.0.54:8443 | `admin` | Gr33nTomato99 |"
        known = find_secrets(source)
        clean, hits = redact(elsewhere, known)
        self.assertNotIn("Gr33nTomato99", clean)
        self.assertIn("known_literal", hits)

    def test_leaves_normal_text_alone(self):
        text = "Mirth Connect listens on https://10.30.0.54:8443 and MQTT on 10.30.0.51:1883."
        self.assertEqual(redact(text)[0], text)

    def test_placeholders_not_learned(self):
        self.assertEqual(find_secrets("password: password\ntoken: <your token>"), set())


class ChunkTests(unittest.TestCase):
    def test_code_comments_are_not_headings(self):
        md = "# Guide\n## Vault\nUnseal it.\n```bash\n# On Vault VM:\nvault operator unseal\n```\nDone.\n## Next\nMore."
        chunks = chunk_markdown(md, "g.md", min_chars=1)
        heads = [c.heading for c in chunks]
        self.assertEqual(heads, ["Guide > Vault", "Guide > Next"])
        self.assertIn("# On Vault VM:", chunks[0].text)

    def test_long_section_is_split(self):
        md = "# H\n" + "\n\n".join(["word " * 100] * 10)
        chunks = chunk_markdown(md, "h.md", max_chars=1000, overlap=100)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c.text) <= 1100 for c in chunks))


class RetrievalTests(unittest.TestCase):
    def test_tokenize_keeps_ips_and_service_names(self):
        toks = tokenize("Restart balloon-control-api.service on 10.30.0.25")
        self.assertIn("10.30.0.25", toks)
        self.assertIn("balloon", toks)
        self.assertIn("balloon-control-api.service", toks)

    def test_bm25_ranks_exact_match_first(self):
        docs = [tokenize(t) for t in ["vault unseal keys", "grafana dashboard panels", "mqtt broker port 1883"]]
        scores = BM25(docs).scores(tokenize("which port is the mqtt broker on"))
        self.assertEqual(max(range(3), key=lambda i: scores[i]), 2)


if __name__ == "__main__":
    unittest.main()
