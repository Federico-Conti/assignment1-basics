from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_basics.tokenizer import Tokenizer


def main() -> int:
    here = Path(__file__).resolve().parent
    input_path = here / "prompt.txt"
    vocab_path = here / "out" / "vocab.json"
    merges_path = here / "out" / "merges.txt"

    special_tokens = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
    )

    text = input_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(text)

    print("ENCODED IDS:")
    print(ids)

    print("\nDECODED TEXT:")
    print(tokenizer.decode(ids))

    print("\nTOKEN BY TOKEN:")
    for token_id in ids:
        print(f"{token_id}: {tokenizer.decode([token_id])!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
