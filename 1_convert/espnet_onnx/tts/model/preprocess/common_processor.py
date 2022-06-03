import numpy as np
from .text_cleaner import TextCleaner


class CommonPreprocessor:
    def __init__(
        self,
        tokenizer,
        token_id_converter,
        cleaner_config,
    ):
        self.text_cleaner = TextCleaner(cleaner_config.cleaner_types)
        self.tokenizer = tokenizer
        self.token_id_converter = token_id_converter

    def __call__(self, data: str) -> np.ndarray:
        text = self.text_cleaner(data)
        tokens = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter.tokens2ids(tokens)
        return np.array(text_ints, dtype=np.int64)
