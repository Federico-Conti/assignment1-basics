from collections import Counter
from pathlib import Path
import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_text(
    text: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Special tokens: confini rigidi per il training BPE.
    Non devono comparire nei conteggi.
    """

    pretoken_counts: Counter[tuple[bytes, ...]] = Counter()

    if special_tokens:
        # Più sicuro ordinare per lunghezza decrescente se ci sono token sovrapposti.
        escaped_tokens = [
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        ]
        special_pattern = "|".join(escaped_tokens)
        segments = re.split(special_pattern, text)
    else:
        segments = [text]

    for segment in segments:
        for match in re.finditer(PAT, segment):
            pretoken_str = match.group()
            pretoken_bytes = pretoken_str.encode("utf-8")
            pretoken_tuple = tuple(bytes([byte]) for byte in pretoken_bytes)
            pretoken_counts[pretoken_tuple] += 1

    return dict(pretoken_counts)


def count_pairs(
    pretoken_counts: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """
    Conta globalmente le coppie adiacenti, ma solo dentro ciascun pre-token.
    """

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for pretoken, count in pretoken_counts.items():
        for i in range(len(pretoken) - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] += count

    return dict(pair_counts)


def merge_pretoken(
    pretoken: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """
    Applica un merge a un singolo pre-token.

    Esempio:
        pretoken = (b"l", b"o", b"w")
        pair = (b"l", b"o")

        output = (b"lo", b"w")
    """

    merged_token = pair[0] + pair[1]
    result: list[bytes] = []

    i = 0
    while i < len(pretoken):
        if (
            i < len(pretoken) - 1
            and pretoken[i] == pair[0]
            and pretoken[i + 1] == pair[1]
        ):
            result.append(merged_token)
            i += 2
        else:
            result.append(pretoken[i])
            i += 1

    return tuple(result)


def apply_merge_to_pretoken_counts(
    pretoken_counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    """
    Applica lo stesso merge a tutti i pre-token del corpus.
    """

    new_counts: Counter[tuple[bytes, ...]] = Counter()

    for pretoken, count in pretoken_counts.items():
        merged_pretoken = merge_pretoken(pretoken, pair)
        new_counts[merged_pretoken] += count

    return dict(new_counts)


def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """
    Inizializza il vocabolario con:
        1. tutti i 256 byte
        2. gli special token
    """

    vocab: dict[int, bytes] = {}

    for i in range(256):
        vocab[i] = bytes([i])

    existing_tokens = set(vocab.values())

    for token in special_tokens:
        token_bytes = token.encode("utf-8")

        if token_bytes in existing_tokens:
            continue

        vocab[len(vocab)] = token_bytes
        existing_tokens.add(token_bytes)

    return vocab


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Training BPE naive.

    Restituisce:
        vocab:
            dict[int, bytes]
        merges:
            list[tuple[bytes, bytes]]
    """

    vocab = initialize_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size={vocab_size} è più piccolo del vocabolario iniziale "
            f"di size {len(vocab)}"
        )

    text = Path(input_path).read_text(encoding="utf-8")
    pretoken_counts = pretokenize_text(text, special_tokens)
    # print_counts("INITIAL PRE-TOKEN COUNTS", pretoken_counts)
    
    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        
        if not pair_counts:
            break

        # print_counts("PAIR COUNTS", pair_counts)

        # Prima massimizza il count, poi in caso di pareggio
        # sceglie la coppia lessicograficamente maggiore.
        best_pair = max(
            pair_counts.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

        # print(f"\nBest pair: {best_pair} -> {best_pair[0] + best_pair[1]}")

        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token 
        merges.append(best_pair)

        pretoken_counts = apply_merge_to_pretoken_counts(
            pretoken_counts,
            best_pair,
        )
        # print_counts("UPDATED PRE-TOKEN COUNTS AFTER ONE MERGE", pretoken_counts)

    return vocab, merges


def print_counts(
    title: str,
    counts: dict,
) -> None:
    print(f"\n{title}:")
    print("{")
    for key, value in counts.items():
        print(f"    {key}: {value},")
    print("}")
