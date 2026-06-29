import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.bpe import train_bpe


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


def token_bytes_to_gpt2_string(token_bytes: bytes) -> str:
    byte_encoder = bytes_to_unicode()
    return "".join(byte_encoder[b] for b in token_bytes)


def save_vocab(vocab: dict[int, bytes], path: Path) -> None:
    gpt2_vocab = {
        token_bytes_to_gpt2_string(token_bytes): token_id
        for token_id, token_bytes in sorted(vocab.items())
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(gpt2_vocab, f)


def save_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_token = token_bytes_to_gpt2_string(left)
            right_token = token_bytes_to_gpt2_string(right)
            f.write(f"{left_token} {right_token}\n")


def main() -> int:
    here = Path(__file__).resolve().parent
    input_path = here / "trainBPE.txt"
    output_dir = here / "out"
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"

    special_tokens = ["<|endoftext|>"]
    vocab_size = 270

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
