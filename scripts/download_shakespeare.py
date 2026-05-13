"""Download the TinyShakespeare corpus (~1.1 MB) for training experiments."""

import os
import sys
import urllib.request

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
       "master/data/tinyshakespeare/input.txt")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "data"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tinyshakespeare.txt")

    if os.path.exists(out_path):
        print(f"already present: {out_path} "
              f"({os.path.getsize(out_path)} bytes)")
        return 0

    print(f"downloading: {URL}")
    urllib.request.urlretrieve(URL, out_path)
    print(f"saved: {out_path} ({os.path.getsize(out_path)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
