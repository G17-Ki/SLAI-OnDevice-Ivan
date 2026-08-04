#!/usr/bin/env python3
from __future__ import annotations

import unittest

import normalisers as n


EXPECTED_NORMALIZATIONS = {
    "en": [
        ("Hello, WORLD!", "hello world"),
        ("I have twenty three apples.", "i have 23 apples"),
        ("Mr. Smith can't come.", "mister smith can not come"),
        ("colour & color", "color color"),
        ("The price is $20.50", "the price is $20.50"),
        ("[noise] Hello (aside) world", "hello world"),
        ("one hundred and five", "105"),
        ("3rd place", "3rd place"),
        ("A.B.C.", "a b c"),
        ("  Multiple\tspaces\n", "multiple spaces"),
        ("She'll arrive at 7 pm.", "she will arrive at 7 pm"),
        ("Dr Jones", "doctor jones"),
    ],
    "zh": [
        ("繁體中文", "繁体中文"),
        ("後臺，軟體！", "后台软体"),
        ("ＡＢＣ１２３，测试", "ABC123测试"),
        ("你好 世界", "你好世界"),
        ("「上海」？", "上海"),
        ("臺灣", "台湾"),
        ("學習（測試）", "学习测试"),
        ("Hello，世界！", "Hello世界"),
        ("一、二；三：", "一二三"),
        ("門裏", "门里"),
        ("甲。乙？丙！", "甲乙丙"),
        ("１２．５％", "125"),
    ],
    "es": [
        ("¿Está aquí? ¡Sí!", "está aquí sí"),
        ("ÉL y él", "él y él"),
        ("El pingüino nada.", "el pingüino nada"),
        ("La canción terminó.", "la canción terminó"),
        ("Niño, niña; año.", "niño niña año"),
        ("¿Qué tal?", "qué tal"),
        ("¡Atención!", "atención"),
        ("uno—dos", "uno dos"),
        ("d'Artagnan", "d artagnan"),
        ("  más\tespacios ", "más espacios"),
        ("Ｃafé", "café"),
        ("sí ≠ si", "sí ≠ si"),
    ],
    "fr": [
        ("L’eau est bonne.", "l'eau est bonne"),
        ("EST-CE vrai ?", "est-ce vrai"),
        ("Dit-il : oui.", "dit-il oui"),
        ("Le garçon arrive.", "le garçon arrive"),
        ("Le cœur bat.", "le cœur bat"),
        ("L'élève écoute.", "l'élève écoute"),
        ("Qu’en pensez-vous ?", "qu'en pensez-vous"),
        ("un—deux", "un deux"),
        ("Aujourd’hui", "aujourd'hui"),
        ("  très\tbien ", "très bien"),
        ("Ça va !", "ça va"),
        ("œuf, oeuvre", "œuf oeuvre"),
    ],
    "ru": [
        ("Ёлка", "елка"),
        ("Всё хорошо.", "все хорошо"),
        ("молоко́", "молоко"),
        ("Где-то", "где-то"),
        ("Кто-нибудь?", "кто-нибудь"),
        ("Скажи-ка!", "скажи-ка"),
        ("ПРИВЕТ, МИР!", "привет мир"),
        ("по—русски", "по русски"),
        ("  много\tпробелов ", "много пробелов"),
        ("Е́лка", "елка"),
        ("раз, два; три", "раз два три"),
        ("№ 5", "no 5"),
    ],
}


PAIRED_GOLDENS = [
    ("en", "twenty three apples", "23 apples", ["23", "apples"], ["23", "apples"], (0, 0, 0)),
    ("en", "hello world", "hello", ["hello", "world"], ["hello"], (0, 0, 1)),
    ("en", "hello", "hello there", ["hello"], ["hello", "there"], (0, 1, 0)),
    ("es", "está aquí", "esta aquí", ["está", "aquí"], ["esta", "aquí"], (1, 0, 0)),
    ("es", "¿Hola, mundo!", "hola mundo", ["hola", "mundo"], ["hola", "mundo"], (0, 0, 0)),
    ("es", "sí", "", ["sí"], [], (0, 0, 1)),
    ("fr", "l'eau est froide", "l eau est froide", ["l'eau", "est", "froide"], ["l", "eau", "est", "froide"], (1, 1, 0)),
    ("fr", "est-ce vrai", "est ce vrai", ["est-ce", "vrai"], ["est", "ce", "vrai"], (1, 1, 0)),
    ("fr", "côte", "cote", ["côte"], ["cote"], (1, 0, 0)),
    ("ru", "ёлка", "елка", ["елка"], ["елка"], (0, 0, 0)),
    ("ru", "мир", "", ["мир"], [], (0, 0, 1)),
    ("ru", "мир", "мир мир", ["мир"], ["мир", "мир"], (0, 1, 0)),
    ("zh", "繁体AI", "繁體ＡＩ", ["繁", "体", "AI"], ["繁", "体", "AI"], (0, 0, 0)),
    ("zh", "中文AI", "中AI", ["中", "文", "AI"], ["中", "AI"], (0, 0, 1)),
    ("zh", "中AI", "中BI", ["中", "AI"], ["中", "BI"], (1, 0, 0)),
]


class NormalizerGoldenTests(unittest.TestCase):
    def test_sixty_golden_normalizations(self) -> None:
        self.assertEqual(sum(map(len, EXPECTED_NORMALIZATIONS.values())), 60)
        for language, cases in EXPECTED_NORMALIZATIONS.items():
            for raw, expected in cases:
                with self.subTest(language=language, raw=raw):
                    actual, rules = n.normalize(raw, language)
                    self.assertEqual(actual, expected)
                    self.assertIsInstance(rules, list)

    def test_language_specific_non_equivalences_are_preserved(self) -> None:
        self.assertNotEqual(n.norm_es("esta")[0], n.norm_es("está")[0])
        self.assertNotEqual(n.norm_es("el")[0], n.norm_es("él")[0])
        self.assertNotEqual(n.norm_fr("cote")[0], n.norm_fr("côte")[0])
        self.assertEqual(n.norm_fr("l’eau")[0], "l'eau")
        self.assertEqual(n.tokenize_normalized(n.norm_fr("l'eau")[0], "fr"), ["l'eau"])
        self.assertEqual(n.tokenize_normalized(n.norm_fr("est-ce")[0], "fr"), ["est-ce"])
        self.assertEqual(n.tokenize_normalized(n.norm_ru("кто-нибудь")[0], "ru"), ["кто-нибудь"])

    def test_fifteen_paired_end_to_end_golden_scores(self) -> None:
        self.assertEqual(len(PAIRED_GOLDENS), 15)
        for language, reference, hypothesis, ref_tokens, hyp_tokens, expected_counts in PAIRED_GOLDENS:
            with self.subTest(language=language, reference=reference, hypothesis=hypothesis):
                row = n.score_pair(reference, hypothesis, language)
                self.assertEqual(row["reference_tokens"], ref_tokens)
                self.assertEqual(row["hypothesis_tokens"], hyp_tokens)
                self.assertEqual(
                    (row["substitutions"], row["insertions"], row["deletions"]),
                    expected_counts,
                )
                self.assertEqual(row["reference_token_count"], len(ref_tokens))

    def test_zh_han_and_embedded_latin_tokenization(self) -> None:
        normalized, _ = n.norm_zh("中ＡI，测试 GPT-4！")
        self.assertEqual(normalized, "中AI测试GPT4")
        self.assertEqual(n.tokenize_normalized(normalized, "zh"), ["中", "AI", "测", "试", "GPT4"])
        mixed, _ = n.norm_zh("中AIБ12≠")
        self.assertEqual(n.tokenize_normalized(mixed, "zh"), ["中", "AI", "Б", "12", "≠"])

    def test_edit_breakdown(self) -> None:
        substitution = n.levenshtein_counts(["a", "b"], ["a", "c"])
        insertion = n.levenshtein_counts(["a"], ["a", "b"])
        deletion = n.levenshtein_counts(["a", "b"], ["a"])
        self.assertEqual((substitution.substitutions, substitution.insertions, substitution.deletions), (1, 0, 0))
        self.assertEqual((insertion.substitutions, insertion.insertions, insertion.deletions), (0, 1, 0))
        self.assertEqual((deletion.substitutions, deletion.insertions, deletion.deletions), (0, 0, 1))

    def test_both_sides_receive_same_rules(self) -> None:
        row = n.score_pair("ЁЛКА", "елка", "ru")
        self.assertEqual(row["errors"], 0)
        row = n.score_pair("繁體", "繁体", "zh")
        self.assertEqual(row["errors"], 0)

    def test_ids_are_versioned_and_hashed(self) -> None:
        for language in n.LANGUAGES:
            self.assertRegex(n.normalizer_id(language), rf"^norm_{language}/1\.0\.0\+[0-9a-f]{{8}}$")
        self.assertEqual(n.basic_normalizer_id(), "whisper_basic/0.1.15/default")

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(TypeError):
            n.normalize(None, "en")
        with self.assertRaises(ValueError):
            n.score_pair("a", "a", "en", normalizer="typo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
