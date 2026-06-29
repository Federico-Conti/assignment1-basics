import ast
import json
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def gpt2_string_to_token_bytes(token: str) -> bytes:
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    return bytes([byte_decoder[c] for c in token])


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in vocab.items()}
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(merges)
        }
        self.special_tokens = special_tokens or []

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = special_bytes
                self.token_to_id[special_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        ids = []

        # caso semplice: nessuno special token
        if not self.special_tokens:
            for match in re.finditer(PAT, text):
                pretoken = match.group()
                token_parts = [bytes([b]) for b in pretoken.encode("utf-8")]
                merged_parts = self._apply_merges(token_parts)

                for part in merged_parts:
                    ids.append(self.token_to_id[part])

            return ids

        # caso con special token
        # bisogna splittare mantenendo gli special token

        """
        text = "the cat<|endoftext|>ate food"
        ["the cat", "<|endoftext|>", "ate food"]
        """
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(tok) for tok in special_tokens_sorted) + ")"

        chunks = re.split(special_pattern, text)

        """
        "the cat"   -> ["the", " cat"]
        "ate food"  -> ["ate", " food"]

        "<|endoftext|>" -> [special_token_id]

        """
        for chunk in chunks:
            if chunk == "":
                continue

            # se il chunk è uno special token, lo aggiungiamo direttamente
            if chunk in self.special_tokens:
                ids.append(self.token_to_id[chunk.encode("utf-8")])
                continue

            # classic BPE
            for match in re.finditer(PAT, chunk):
                pretoken = match.group()
                token_parts = [bytes([b]) for b in pretoken.encode("utf-8")]
                merged_parts = self._apply_merges(token_parts)

                for part in merged_parts:
                    ids.append(self.token_to_id[part])
        return ids

    def _apply_merges(self, token_parts: list[bytes]) -> list[bytes]:
        while len(token_parts) > 1:
            best_pair = None
            best_rank = None

            for i in range(len(token_parts) - 1):
                pair = (token_parts[i], token_parts[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged_parts = []
            i = 0
            while i < len(token_parts):
                if (
                    i < len(token_parts) - 1
                    and token_parts[i] == best_pair[0]
                    and token_parts[i + 1] == best_pair[1]
                ):
                    merged_parts.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged_parts.append(token_parts[i])
                    i += 1
            token_parts = merged_parts

        return token_parts

    def decode(self, ids: list[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")
    
    def encode_iterable(self, iterable):
        for text in iterable:
            yield from self.encode(text)
 
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            if str(vocab_filepath).endswith(".json"):
                gpt2_vocab = json.load(f)
                vocab = {
                    token_id: gpt2_string_to_token_bytes(token)
                    for token, token_id in gpt2_vocab.items()
                }
            else:
                vocab = {}
                for line in f:
                    token_id, token_bytes = line.strip().split("\t")
                    vocab[int(token_id)] = ast.literal_eval(token_bytes)

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if "\t" in stripped_line:
                    left, right = stripped_line.split("\t")
                    merges.append((ast.literal_eval(left), ast.literal_eval(right)))
                else:
                    left, right = stripped_line.split()
                    merges.append((
                        gpt2_string_to_token_bytes(left),
                        gpt2_string_to_token_bytes(right),
                    ))

        return cls(vocab, merges, special_tokens)
