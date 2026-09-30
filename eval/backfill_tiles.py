"""Fill tiles the search serve returned without an image, from StarTrail-org/pixelrag-tiles.

    python backfill_tiles.py <dump_root> <tar_cache_dir>

Reads <dump_root>/*/missing_tiles.jsonl (written by --dump-retrieval), downloads each
needed shard tar once, and writes the tile as PNG at <dump_root>/<bench>/<tile>. The HF
tiles are lossless WebP, so the PNG is pixel-identical to the original render. Tiles not
found in the tar are listed at the end; pack_bench.py refuses a bench that still lacks one.
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone

from huggingface_hub import hf_hub_download
from PIL import Image

ARTICLES_PER_SHARD = 8284  # shard = article_id // 8284, as in the corpus layout


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


def main():
    dump_root, cache = sys.argv[1], sys.argv[2]
    # (article_id, chunk stem) -> destination paths across benches
    wanted: dict[tuple[str, str], list[str]] = defaultdict(list)
    for f in glob.glob(os.path.join(dump_root, "*", "missing_tiles.jsonl")):
        bench_dir = os.path.dirname(f)
        for line in open(f):
            m = json.loads(line)
            dst = os.path.join(bench_dir, m["tile"])
            if not os.path.exists(dst):
                stem = os.path.splitext(os.path.basename(m["tile"]))[0]
                wanted[(str(m["article_id"]), stem)].append(dst)

    by_shard: dict[int, set] = defaultdict(set)
    for aid, stem in wanted:
        by_shard[int(aid) // ARTICLES_PER_SHARD].add((aid, stem))
    _log(f"need {len(wanted)} tiles from {len(by_shard)} shards: {sorted(by_shard)}")

    found = set()
    for shard, keys in sorted(by_shard.items()):
        _log(f"shard={shard:03d} download_start tiles={len(keys)}")
        tar_path = hf_hub_download(
            "StarTrail-org/pixelrag-tiles", f"shard_{shard:03d}.tar",
            repo_type="dataset", local_dir=cache,
        )
        with tarfile.open(tar_path) as tar:
            for member in tar:
                parts = member.name.split("/")
                if len(parts) < 2 or not parts[-2].endswith(".png.tiles"):
                    continue
                key = (parts[-2].removesuffix(".png.tiles"), os.path.splitext(parts[-1])[0])
                if key not in keys:
                    continue
                img = Image.open(io.BytesIO(tar.extractfile(member).read()))
                for dst in wanted[key]:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    img.save(dst + ".part", format="PNG")
                    os.replace(dst + ".part", dst)
                found.add(key)
                _log(f"shard={shard:03d} tile={key[0]}/{key[1]} status=ok member={member.name}")
        for key in sorted(keys - found):
            _log(f"shard={shard:03d} tile={key[0]}/{key[1]} status=not_in_tar")
        os.remove(tar_path)  # re-downloadable; 1-4 GB each
        _log(f"shard={shard:03d} done found={len(keys & found)}/{len(keys)}")

    missing = sorted(set(wanted) - found)
    _log(f"backfill_end filled={len(found)} still_missing={len(missing)}")
    for aid, stem in missing:
        print(f"  still missing: {aid}/{stem}  -> {wanted[(aid, stem)]}")


if __name__ == "__main__":
    main()
