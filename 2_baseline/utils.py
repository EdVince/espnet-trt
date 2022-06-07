from abc import ABC
import onnxruntime
import g2p_en
import numpy as np
import tacotron_cleaner.cleaners
from typing import Dict, Iterable, List, Union
from pathlib import Path
from typeguard import check_argument_types

class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]] = ['<blank>', '<unk>', 'AH0', 'N', 'T', 'D', 'S', 'R', 'L', 'DH', 'K', 'Z', 'IH1', 'IH0', 'M', 'EH1', 'W', 'P', 'AE1', 'AH1', 'V', 'ER0', 'F', ',', 'AA1', 'B', 'HH', 'IY1', 'UW1', 'IY0', 'AO1', 'EY1', 'AY1', '.', 'OW1', 'SH', 'NG', 'G', 'ER1', 'CH', 'JH', 'Y', 'AW1', 'TH', 'UH1', 'EH2', 'OW0', 'EY2', 'AO0', 'IH2', 'AE2', 'AY2', 'AA2', 'UW0', 'EH0', 'OY1', 'EY0', 'AO2', 'ZH', 'OW2', 'AE0', 'UW2', 'AH2', 'AY0', 'IY2', 'AW2', 'AA0', "'", 'ER2', 'UH2', '?', 'OY2', '!', 'AW0', 'UH0', 'OY0', '..', '<sos/eos>'],
        unk_symbol: str = "<unk>",
    ):
        assert check_argument_types()

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token2id.get(i, self.unk_id) for i in tokens]

class CommonPreprocessor:
    def __init__(self,tokenizer,token_id_converter):
        self.tokenizer = tokenizer
        self.token_id_converter = token_id_converter

    def __call__(self, data: str) -> np.ndarray:
        text = tacotron_cleaner.cleaners.custom_english_cleaners(data)
        tokens = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter.tokens2ids(tokens)
        return np.array(text_ints, dtype=np.int64)

class PhonemeTokenizer:

    def g2p(self, text):
        phones = g2p_en.G2p()(text)
        phones = list(filter(lambda s: s != " ", phones))
        return phones

    def text2tokens(self, line: str):
        tokens = []
        while len(line) != 0:
            t = line[0]
            tokens.append(t)
            line = line[1:]
        line = "".join(tokens)
        tokens = self.g2p(line)
        return tokens

    def tokens2text(self, tokens):
        return "".join(tokens)


class Text2Speech(ABC):

    def __init__(self):
        self.preprocess = CommonPreprocessor(tokenizer=PhonemeTokenizer(), token_id_converter=TokenIDConverter())
        self.model = onnxruntime.InferenceSession('baseline.onnx', providers=['CPUExecutionProvider'])

    def __call__(self, text):
        text = self.preprocess(text)
        wav, y_length = self.model.run(['wav', 'y_length'], {'text': text})
        trueData = {'text':text, 'wav':wav, 'y_length':y_length}
        return wav[:y_length[0]], trueData

