"""Exercise 10 (stretch): Implement your own KV cache.

Two parts:

Part A — A trie-based PrefixCache that supports longest-prefix lookup,
storage of arbitrary "state" objects, and TTL eviction. Compared to a
naive flat dict that keys on every prefix tuple.

Part B — Wire the trie to a real model. Loads distilgpt2 from
HuggingFace, demonstrates the wall-clock difference between
`generate(use_cache=False)` and `generate(use_cache=True)`, then drives
the forward pass by hand: cache `past_key_values` after a cold prefill,
reuse them on a longer prompt that begins with the same prefix, and
only feed the new tokens through the model.

Part B is skipped with a friendly message if `torch` or `transformers`
are not installed. To run it: `uv add torch transformers` (downloads
~350 MB for distilgpt2 the first time).
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ----------------------------- Part A: data structure -----------------------------


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    state: Any = None
    last_access: float = 0.0


class PrefixCache:
    """Trie of token IDs. Each node optionally carries a state object."""

    def __init__(self) -> None:
        self._root = TrieNode(last_access=time.monotonic())

    def put(self, token_ids: list[int], state: Any) -> None:
        # Set the state at every node along the path. KV cache state is
        # decomposable — past_key_values for an N-token prefix is reusable
        # for any prefix shorter than N by slicing — so every cached node
        # carries a handle the caller can truncate to its matched depth.
        node = self._root
        now = time.monotonic()
        for tok in token_ids:
            node = node.children.setdefault(tok, TrieNode())
            node.last_access = now
            node.state = state

    def get_longest_match(self, token_ids: list[int]) -> tuple[int, Any]:
        node = self._root
        best_len, best_state = 0, None
        now = time.monotonic()
        for i, tok in enumerate(token_ids, start=1):
            child = node.children.get(tok)
            if child is None:
                break
            child.last_access = now
            node = child
            if node.state is not None:
                best_len, best_state = i, node.state
        return best_len, best_state

    def evict_older_than(self, seconds: float) -> int:
        cutoff = time.monotonic() - seconds
        return self._sweep(self._root, cutoff)

    def _sweep(self, node: TrieNode, cutoff: float) -> int:
        removed = 0
        dead_keys: list[int] = []
        for tok, child in node.children.items():
            removed += self._sweep(child, cutoff)
            if not child.children and child.last_access < cutoff:
                dead_keys.append(tok)
                removed += 1
        for tok in dead_keys:
            child = node.children.pop(tok)
            child.state = None
        return removed


class NaivePrefixCache:
    """Flat dict keyed on the full token tuple. To answer
    `get_longest_match`, scan back through every possible prefix length
    and rebuild the tuple each time — quadratic per query."""

    def __init__(self) -> None:
        self._d: dict[tuple[int, ...], tuple[Any, float]] = {}

    def put(self, token_ids: list[int], state: Any) -> None:
        self._d[tuple(token_ids)] = (state, time.monotonic())

    def get_longest_match(self, token_ids: list[int]) -> tuple[int, Any]:
        for k in range(len(token_ids), 0, -1):
            entry = self._d.get(tuple(token_ids[:k]))
            if entry is not None:
                state, _ = entry
                return k, state
        return 0, None


def _synthetic_workload(
    n_prompts: int = 1000,
    common_prefix_len: int = 1000,
    suffix_len: int = 20,
    seed: int = 7,
) -> list[list[int]]:
    """Many short queries that share a long common prefix — modelling a
    long, stable system prompt followed by per-call user input. This is
    where prompt caching earns its keep, and where a trie crushes a flat
    dict: the trie shares the 1000-token prefix as one path, the dict
    has to store N copies."""
    rng = random.Random(seed)
    common = [rng.randint(0, 50_000) for _ in range(common_prefix_len)]
    return [
        common + [rng.randint(0, 50_000) for _ in range(suffix_len)]
        for _ in range(n_prompts)
    ]


def part_a() -> None:
    print("=" * 60)
    print("Part A — PrefixCache trie vs naive dict")
    print("=" * 60)

    prompts = _synthetic_workload()

    for label, cache in (("trie ", PrefixCache()), ("naive", NaivePrefixCache())):
        match_lens: list[int] = []
        t0 = time.monotonic()
        for i, p in enumerate(prompts):
            n, _ = cache.get_longest_match(p)
            match_lens.append(n)
            cache.put(p, f"state-{i}")
        elapsed = time.monotonic() - t0
        avg_match = sum(match_lens) / len(match_lens)
        print(
            f"{label}: {elapsed*1000:7.1f} ms total   "
            f"avg_match={avg_match:6.1f} tokens   "
            f"final_match={match_lens[-1]} tokens"
        )

    # TTL eviction sanity check.
    cache = PrefixCache()
    cache.put([1, 2, 3], "old")
    time.sleep(0.05)
    cache.put([1, 2, 4], "new")
    removed = cache.evict_older_than(0.025)
    print(f"\nevict_older_than(25ms) removed {removed} node(s)")
    print("after eviction:")
    print(f"  match([1,2,3]) -> {cache.get_longest_match([1, 2, 3])}")
    print(f"  match([1,2,4]) -> {cache.get_longest_match([1, 2, 4])}")


# ----------------------------- Part B: real model -----------------------------


def part_b() -> None:
    print()
    print("=" * 60)
    print("Part B — wire the trie to a real model (distilgpt2)")
    print("=" * 60)

    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("Skipping: install torch + transformers to run Part B.")
        print("  uv add torch transformers")
        return

    print("Loading distilgpt2 (downloads ~350 MB on first run)...")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    model.eval()

    # ---- 1. generate(use_cache=False) vs generate(use_cache=True) ----
    long_text = "The quick brown fox jumps over the lazy dog. " * 40
    inputs = tokenizer(long_text, return_tensors="pt")
    print(f"\nprompt: {inputs.input_ids.shape[1]} tokens, generating 30 new tokens")

    with torch.no_grad():
        t = time.monotonic()
        _ = model.generate(**inputs, max_new_tokens=30, use_cache=False, do_sample=False)
        no_cache = time.monotonic() - t

        t = time.monotonic()
        _ = model.generate(**inputs, max_new_tokens=30, use_cache=True, do_sample=False)
        with_cache = time.monotonic() - t

    print(f"  generate(use_cache=False): {no_cache:5.2f}s")
    print(f"  generate(use_cache=True):  {with_cache:5.2f}s   (in-call KV reuse)")

    # ---- 2. Hand-rolled prefix cache with PrefixCache ----
    cache = PrefixCache()

    def encode(text: str) -> list[int]:
        return tokenizer(text, return_tensors="pt").input_ids[0].tolist()

    prefix_text = "The capital of France is Paris and the capital of Germany is Berlin. "
    extended_text = prefix_text + "The capital of Spain is"

    prefix_ids = encode(prefix_text)
    extended_ids = encode(extended_text)

    # Cold prefill of the prefix; capture past_key_values into the trie.
    with torch.no_grad():
        t = time.monotonic()
        out = model(input_ids=torch.tensor([prefix_ids]), use_cache=True)
        cold_prefix_time = time.monotonic() - t
    cache.put(prefix_ids, out.past_key_values)
    print(
        f"\ncold prefix prefill ({len(prefix_ids)} tokens): "
        f"{cold_prefix_time*1000:6.1f} ms"
    )

    # Warm path: longest-prefix-match in the trie, then forward only the new tokens.
    matched_len, cached_pkv = cache.get_longest_match(extended_ids)
    new_tokens = extended_ids[matched_len:]
    print(f"trie matched {matched_len}/{len(extended_ids)} tokens; "
          f"feeding {len(new_tokens)} new tokens to the model")

    with torch.no_grad():
        t = time.monotonic()
        _ = model(
            input_ids=torch.tensor([new_tokens]),
            past_key_values=cached_pkv,
            use_cache=True,
        )
        warm_time = time.monotonic() - t

    # Reference: same extended prompt processed cold, no cache.
    with torch.no_grad():
        t = time.monotonic()
        _ = model(input_ids=torch.tensor([extended_ids]), use_cache=True)
        cold_full_time = time.monotonic() - t

    print(f"\nwarm forward (with cached past_key_values): {warm_time*1000:6.1f} ms")
    print(f"cold forward (no cache, full prompt):       {cold_full_time*1000:6.1f} ms")
    print(f"speedup from prefix reuse: {cold_full_time / warm_time:.2f}x")


if __name__ == "__main__":
    part_a()
    if "--no-model" not in sys.argv:
        part_b()
