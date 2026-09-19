#!/usr/bin/env python3
"""Run the test suite with every non-loopback socket blocked.

The suite is supposed to make no network calls: LLM responses come from
fakes, the embedder is stubbed, and the vector store runs in memory. That is
a property worth enforcing rather than asserting, because it decays quietly --
an earlier version of this project ran the whole suite against live Gemini
because nothing disabled it, and separately made 84 requests per run to
huggingface_hub checking for model updates.

Exits non-zero if any test fails or if anything tried to reach the network.

    python scripts/run_tests_offline.py [pytest args...]
"""

import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_attempts = []
_real_connect = socket.socket.connect


def _blocked_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if isinstance(host, str) and not host.startswith("127.") and host not in ("::1", "localhost"):
        _attempts.append(host)
        raise OSError(f"network access blocked in offline test run: {host}")
    return _real_connect(self, address)


def main() -> int:
    socket.socket.connect = _blocked_connect

    import pytest

    args = sys.argv[1:] or ["tests/", "-q"]
    code = pytest.main(args)

    if _attempts:
        unique = sorted(set(_attempts))
        print(f"\nFAIL: {len(_attempts)} outbound connection attempt(s) to "
              f"{len(unique)} host(s):")
        for host in unique[:10]:
            print(f"  {host}")
        return 1

    print("\nNo outbound connections attempted.")
    return code


if __name__ == "__main__":
    sys.exit(main())
