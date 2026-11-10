#!/usr/bin/env bash
# fetch-model <url> <destination> <sha256>
# Downloads one model file during `docker build` and fails the build unless
# the SHA-256 matches. URLs are pinned to a Hugging Face commit, so a repo
# update can never change what an image contains.
set -euo pipefail

url="$1"
dest="$2"
sha="$3"

mkdir -p "$(dirname "$dest")"
echo "fetch-model: $(basename "$dest")"
wget --quiet --tries=5 --waitretry=10 --retry-connrefused --timeout=60 \
     --output-document="$dest.part" "$url"
echo "$sha  $dest.part" | sha256sum --check --status || {
    echo "fetch-model: SHA-256 mismatch for $dest" >&2
    echo "  expected $sha" >&2
    echo "  got      $(sha256sum "$dest.part" | cut -d' ' -f1)" >&2
    rm -f "$dest.part"
    exit 1
}
mv "$dest.part" "$dest"
echo "fetch-model: ok $(du -h "$dest" | cut -f1)"
