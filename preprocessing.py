import os
import numpy as np
import torch
import tokenization
import json
import collections

from task_datasets import SQuADDataset


def truncate_seq_pair(tokens_a: list, tokens_b: list, max_length: int) -> None:
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prep_sentence_pairs_data(data: list, vocab_path: str, max_seq_len: int) -> tuple:
    tokenizer = tokenization.FullTokenizer(vocab_path)
    input_ids_list = []
    token_type_ids_list = []
    labels = []
    for row in data:
        label = float(row[0])
        text_a = row[1]
        text_b = row[2]
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        for x in [input_ids, token_type_ids]:
            x.extend([0]*(max_seq_len - len(x)))
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        labels.append(label)

    return np.array(input_ids_list), np.array(token_type_ids_list), np.array(labels)


def prep_single_sentence_data(data: list, vocab_path: str, max_seq_len: int) -> tuple:
    tokenizer = tokenization.FullTokenizer(vocab_path)
    input_ids_list = []
    token_type_ids_list = []
    labels = []
    for row in data:
        label = float(row[0])
        text = row[1]
        tokens = tokenizer.tokenize(text)
        tokens = tokens[:max_seq_len-2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * len(tokens)
        for x in [input_ids, token_type_ids]:
            x.extend([0]*(max_seq_len - len(x)))
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        labels.append(label)

    return np.array(input_ids_list), np.array(token_type_ids_list), np.array(labels)


class SQuADOpsHandler:
    """
    An object for handling all squad data related ops for loading data, preprocessing, and predictions.

    Args:
        max_context_length (int): Maximum possible context length in one example.
        max_query_length (int): Maximum possible query length.
        doc_stride (int): Size of sliding window for context with a length larger than maximum length.
        use_squad_v1 (bool): Whether to use only SQuAD v1 examples or not.
        data_path (str): Path to folder containing the train and dev datastets.
        vocab_path (str): Path to vocab file for the tokenizer.
    """
    def __init__(self, max_context_length: int, max_query_length: int, doc_stride: int, max_answer_length: int,
                 use_squad_v1: bool, do_lower_case: bool, data_path: str, vocab_path: str) -> None:
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.max_answer_length = max_answer_length
        self.use_squad_v1 = use_squad_v1
        self.do_lower_case = do_lower_case
        self.data_path = data_path
        self.tokenizer = tokenization.FullTokenizer(vocab_path)

    def get_train_dataset(self, file_name: str = "train-v2.0.json") -> SQuADDataset:
        train_path = os.path.join(self.data_path, file_name)
        examples = self._load_squad_examples(train_path)
        features = self._examples_to_features(examples, False)
        train_dataset = self._features_to_dataset(features)

        return train_dataset

    def get_dev_dataset_and_eval_items(self, file_name: str = "dev-v2.0.json") -> tuple:
        dev_path = os.path.join(self.data_path, file_name)
        examples = self._load_squad_examples(dev_path)
        features = self._examples_to_features(examples, True)
        dev_dataset = self._features_to_dataset(features)
        eval_items = self._extract_eval_items(features)

        return dev_dataset, eval_items

    @staticmethod
    def _is_whitespace(c: str) -> bool:
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _load_squad_examples(self, path: str) -> list:
        with open(path, 'r') as f:
            input_data = json.load(f)

        input_data = input_data['data']
        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if self._is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qa_id = qa["id"]
                    qa_all_answers = qa["answers"]
                    question_text = qa["question"]
                    is_impossible = qa["is_impossible"]
                    if self.use_squad_v1:
                        if is_impossible:
                            continue
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            tokenization.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                    examples.append({'question_text': question_text,
                                     'doc_tokens': doc_tokens,
                                     'orig_answer_text': orig_answer_text,
                                     'start_position': start_position,
                                     'end_position': end_position,
                                     'is_impossible': is_impossible,
                                     'qa_id': qa_id,
                                     'qa_all_answers': qa_all_answers})

        return examples

    def _examples_to_features(self, examples: list, for_eval: bool) -> list:
        features = []
        for example in examples:
            query_tokens = self.tokenizer.tokenize(example['question_text'])

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example['doc_tokens']):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            if example['is_impossible']:
                tok_start_position = -1
                tok_end_position = -1
            else:
                tok_start_position = orig_to_tok_index[example['start_position']]
                if example['end_position'] < len(example['doc_tokens']) - 1:
                    tok_end_position = orig_to_tok_index[example['end_position'] + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, example['orig_answer_text']
                )

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_context_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])  # pylint: disable=invalid-name
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                token_is_max_context = None
                if for_eval:
                    token_is_max_context = {}

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                    if for_eval:
                        is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                                    split_token_index)
                        token_is_max_context[len(tokens)] = is_max_context
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # Zero-pad up to the sequence length.
                for x in [input_ids, segment_ids]:
                    x.extend([0]*(self.max_context_length - len(x)))

                assert len(input_ids) == self.max_context_length
                assert len(segment_ids) == self.max_context_length

                if not example['is_impossible']:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        if self.use_squad_v1:
                            continue
                        else:
                            start_position = 0
                            end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                else:
                    start_position = 0
                    end_position = 0

                if for_eval:
                    features.append((input_ids, segment_ids, start_position, end_position, example['qa_id'],
                                     example['qa_all_answers'], tokens, token_to_orig_map, token_is_max_context,
                                     example['doc_tokens'], example['orig_answer_text']))
                else:
                    features.append((input_ids, segment_ids, start_position, end_position))

        return features

    @staticmethod
    def _check_is_max_context(doc_spans: list, cur_span_index: int, position: int) -> bool:
        """Check if this is the 'max context' doc span for the token."""

        # Because of the sliding window approach taken to scoring documents, a single
        # token can appear in multiple documents. E.g.
        #  Doc: the man went to the store and bought a gallon of milk
        #  Span A: the man went to the
        #  Span B: to the store and bought
        #  Span C: and bought a gallon of
        #  ...
        #
        # Now the word 'bought' will have two scores from spans B and C. We only
        # want to consider the score with "maximum context", which we define as
        # the *minimum* of its left and right context (the *sum* of left and
        # right context will always be the same, of course).
        #
        # In the example the maximum context for 'bought' would be span C since
        # it has 1 left context and 3 right context, while span B has 4 left context
        # and 0 right context.

        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _improve_answer_span(self, doc_tokens: list, input_start: int, input_end: int, orig_answer_text: str) -> tuple:
        """Returns tokenized answer spans that better match the annotated answer."""

        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.

        tok_answer_text = " ".join(self.tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return new_start, new_end

        return input_start, input_end

    @staticmethod
    def _features_to_dataset(features: list) -> SQuADDataset:
        input_ids = []
        token_type_ids = []
        start_labels = []
        end_labels = []
        for feature in features:
            input_ids.append(feature[0])
            token_type_ids.append(feature[1])
            start_labels.append(feature[2])
            end_labels.append(feature[3])
        input_ids = np.array(input_ids)
        token_type_ids = np.array(token_type_ids)
        start_labels = np.array(start_labels)
        end_labels = np.array(end_labels)
        dataset = SQuADDataset(input_ids, token_type_ids, start_labels, end_labels)

        return dataset

    @staticmethod
    def _extract_eval_items(features: list) -> list:
        eval_items = []
        for feature in features:
            eval_items.append({
                'qa_id': feature[4],
                'qa_all_answers': feature[5],
                'tokens': feature[6],
                'token_to_orig_map': feature[7],
                'token_is_max_context': feature[8],
                'doc_tokens': feature[9],
                'orig_answer_text': feature[10]
            })
        return eval_items

    def logits_to_pred_indices(self, s_scores: torch.Tensor, e_scores: torch.Tensor, eval_items: dict) -> list:
        seq_len = s_scores.shape[1]
        to_keep = (seq_len**2 - seq_len) // 2 + seq_len
        scores = s_scores.unsqueeze(dim=2) + e_scores.unsqueeze(dim=1)
        scores = torch.triu(scores, diagonal=0)
        scores.masked_fill_(scores == 0, -torch.inf)
        scores = scores.view(scores.shape[0], -1)
        scores = torch.softmax(scores, dim=-1)
        sorted_probs, sorted_arg_lists = torch.sort(scores, dim=1, descending=True)
        sorted_probs = sorted_probs[:, :to_keep].tolist()
        sorted_arg_lists = sorted_arg_lists[:, :to_keep].tolist()
        pred_indices = []
        for idx, (arg_list, probs_list) in enumerate(zip(sorted_arg_lists, sorted_probs)):
            for arg, prob in zip(arg_list, probs_list):
                pred_start, pred_end = arg // seq_len, arg % seq_len
                if self._are_pred_indices_valid(pred_start, pred_end, eval_items[idx]):
                    pred_indices.append(((pred_start, pred_end), prob))
                    break

        return pred_indices

    def pred_indices_to_final_answers(self, pred_indices_list: list, eval_items: list) -> list:
        final_answers = {}
        for idx, indices_prob_tup in enumerate(pred_indices_list):
            answer_prob = indices_prob_tup[1]
            pred_start, pred_end = indices_prob_tup[0][0], indices_prob_tup[0][1]
            eval_items_for_example = eval_items[idx]
            tok_tokens = eval_items_for_example['tokens'][pred_start: pred_end + 1]
            orig_doc_start = eval_items_for_example['token_to_orig_map'][pred_start]
            orig_doc_end = eval_items_for_example['token_to_orig_map'][pred_end]
            orig_tokens = eval_items_for_example['doc_tokens'][orig_doc_start: orig_doc_end + 1]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = self._get_final_text(tok_text, orig_text)

            example_id = eval_items_for_example['qa_id']
            if example_id in final_answers:
                if final_answers[example_id][1] < answer_prob:
                    final_answers[example_id] = (final_text, answer_prob)
            else:
                final_answers[example_id] = (final_text, answer_prob)

        return final_answers

    def _are_pred_indices_valid(self, start_index: int, end_index: int, eval_items_for_example: dict) -> bool:
        if start_index >= len(eval_items_for_example['tokens']):
            return False
        if end_index >= len(eval_items_for_example['tokens']):
            return False
        if start_index not in eval_items_for_example['token_to_orig_map']:
            return False
        if end_index not in eval_items_for_example['token_to_orig_map']:
            return False
        # if not eval_items_for_example['token_is_max_context'].get(start_index, False):
        #     return False
        if end_index < start_index:
            return False
        length = end_index - start_index + 1
        if length > self.max_answer_length:
            return False

        return True

    def _get_final_text(self, pred_text: str, orig_text: str) -> str:
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tokenizer = tokenization.BasicTokenizer(do_lower_case=self.do_lower_case)

        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            return orig_text
        end_position = start_position + len(pred_text) - 1

        orig_ns_text, orig_ns_to_s_map = self._strip_spaces(orig_text)
        tok_ns_text, tok_ns_to_s_map = self._strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {v: k for k, v in tok_ns_to_s_map.items()}

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]

        return output_text

    @staticmethod
    def _strip_spaces(text: str) -> tuple:
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for i, c in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)

        return ns_text, ns_to_s_map



