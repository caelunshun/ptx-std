#!/usr/bin/env bash
### LLM-generated ###
# Emits the PTX of the tma_ptxas example for each given target (default: sm_90 sm_100a
# sm_103a) and assembles it with ptxas. Each target uses its own target directory, since changing
# the target CPU rebuilds `core`.
set -euo pipefail
cd "$(dirname "$0")/../.."

archs=("$@")
if [ ${#archs[@]} -eq 0 ]; then
    archs=(sm_90 sm_100a sm_103a)
fi

for arch in "${archs[@]}"; do
    target_dir="target/tma-ptxas/$arch"
    # Force cargo to re-emit the PTX of the example (the library and `core` stay cached).
    find "$target_dir" -name 'tma_ptxas-*.s' -delete 2>/dev/null || true
    touch examples/tma_ptxas/main.rs
    RUSTFLAGS="-C target-cpu=$arch -C target-feature=+ptx93,+$arch" \
        cargo rustc --quiet --release --example tma_ptxas --target-dir "$target_dir" \
        -- --emit=asm -C codegen-units=1
    ptx=$(find "$target_dir" -name 'tma_ptxas-*.s')
    ptxas -arch="$arch" "$ptx" -o /dev/null
    echo "$arch: ok ($(grep -cE '^\s*cp\.reduce\.async\.bulk\.' "$ptx") cp.reduce.async.bulk instructions in $ptx)"
done
