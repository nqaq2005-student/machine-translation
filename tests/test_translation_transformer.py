import unittest

from translation_transformer import (
    BidirectionalTranslationTransformer,
    TransformerConfig,
)


class TranslationTransformerTests(unittest.TestCase):
    def test_default_config_is_around_50m_parameters(self) -> None:
        params = TransformerConfig().estimated_parameters()
        self.assertGreaterEqual(params, 45_000_000)
        self.assertLessEqual(params, 60_000_000)

    def test_supports_both_directions(self) -> None:
        model = BidirectionalTranslationTransformer()
        self.assertEqual(model.get_direction_token("vi-en"), "<2en>")
        self.assertEqual(model.get_direction_token("en-vi"), "<2vi>")

    def test_build_training_sample_has_direction_prefix(self) -> None:
        model = BidirectionalTranslationTransformer()
        sample = model.build_training_sample(
            source_text="xin chào",
            target_text="hello",
            direction="vi-en",
        )
        self.assertEqual(sample["source"], "<2en> xin chào")
        self.assertEqual(sample["target"], "hello")

    def test_invalid_direction_raises(self) -> None:
        model = BidirectionalTranslationTransformer()
        with self.assertRaises(ValueError):
            model.get_direction_token("fr-en")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
