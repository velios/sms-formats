import unittest

from sms_format import SmsFormat, compile_regex, validate_cross_match, validate_format_examples


class ExampleNormalizationTests(unittest.TestCase):
    def test_nfc_regex_accepts_decomposed_and_composed_examples(self):
        examples = ["lu\u0301c", "lúc"]
        fmt = SmsFormat(regex="^(lúc)$", regex_group_names=["comment"], examples=examples.copy())
        self.assertEqual(validate_format_examples(fmt), [])
        self.assertEqual(fmt.examples, examples)
        self.assertEqual(fmt.regex, "^(lúc)$")

    def test_cross_match_uses_normalized_text_and_reports_original_example(self):
        original = "lu\u0301c"
        fmt = SmsFormat(regex="^lúc$", regex_group_names=[], examples=[original])
        other = SmsFormat(regex="^lúc$", regex_group_names=[], examples=[])
        errors = validate_cross_match(
            [
                (fmt, compile_regex(fmt.regex, "own.txt"), "own.txt"),
                (other, compile_regex(other.regex, "other.txt"), "other.txt"),
            ]
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].kind, "cross_match")
        self.assertEqual(errors[0].example_text, original)
        self.assertEqual(errors[0].other_file_path, "other.txt")

    def test_normalization_does_not_rewrite_regex(self):
        fmt = SmsFormat(regex="^lu\u0301c$", regex_group_names=[], examples=["lu\u0301c"])
        errors = validate_format_examples(fmt)
        self.assertEqual([error.kind for error in errors], ["example_no_match"])
        self.assertEqual(errors[0].example_text, "lu\u0301c")

    def test_nfc_does_not_fold_letters_or_compatibility_characters(self):
        for pattern, example in [("^ежик$", "ёжик"), ("^1$", "①")]:
            with self.subTest(example=example):
                fmt = SmsFormat(regex=pattern, regex_group_names=[], examples=[example])
                self.assertEqual(
                    [error.kind for error in validate_format_examples(fmt)], ["example_no_match"]
                )


if __name__ == "__main__":
    unittest.main()
