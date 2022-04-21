import unicodedata
import collections


class FullTokenizer:
    """
    An exact copy of Google's FullTokenizer class and also
    the necessary secondary classes that follow.

    Args:
        vocab_file (str): Path to vocab file.
        do_lower_case (bool): Whether to lower case everything or not.
    """

    def __init__(self, vocab_file: str, do_lower_case: bool = True) -> None:
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text: str) -> list[str]:
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer:
    """
    Copy of BasicTokenizer class from google's BERT repo. Chinese support
    is left out as I'm only going to be working with English in this implementation.

    Args:
        do_lower_case (bool): Whether to lower case everything or not.
    """

    def __init__(self, do_lower_case: bool = True) -> None:
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> list[str]:
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))

        return output_tokens

    @staticmethod
    def _run_strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)

        return "".join(output)

    @staticmethod
    def _run_split_on_punc(text: str) -> list[str]:
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def _clean_text(text: str) -> str:
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)

        return "".join(output)


class WordpieceTokenizer:
    """
    Copy of WordpieceTokenizer class from google's BERT repo.

    Args:
        vocab (dict): Vocabulary of words.
        unk_token (str): Token for words/subwords not found in the vocabulary.
        max_input_chars_per_word (int): Maximum characters allowed in a single word
    """

    def __init__(self, vocab: dict, unk_token: str = "[UNK]", max_input_chars_per_word: int = 200) -> None:
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text: str) -> list[str]:
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)

        return output_tokens


def _is_whitespace(char: str) -> bool:
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char: str) -> bool:
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def convert_to_unicode(text: str) -> str:
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def convert_by_vocab(vocab: dict, items: list) -> list:
    output = []
    for item in items:
        output.append(vocab[item])

    return output


def convert_tokens_to_ids(vocab: dict[str, int], tokens: list[str]) -> list[int]:

    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab: dict[int, str], ids: list[int]) -> list[str]:

    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    tokens = text.split()

    return tokens


def load_vocab(vocab_path: str) -> dict:
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_path, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1

    return vocab
