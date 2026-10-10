#!/usr/bin/env bash
### LLM-generated ###
# Emits the PTX of the tcgen05_ptxas example for each given target (default: all supported tcgen05
# targets) and assembles it with ptxas. Each target uses its own target directory, since changing
# the target CPU rebuilds `core`.
set -euo pipefail
cd "$(dirname "$0")/../.."

archs=("$@")
if [ ${#archs[@]} -eq 0 ]; then
    archs=(sm_100a sm_100f sm_103a sm_110a)
fi

for arch in "${archs[@]}"; do
    target_dir="target/tcgen05-ptxas/$arch"
    # Force cargo to re-emit the PTX of the example (the library and `core` stay cached).
    find "$target_dir" -name 'tcgen05_ptxas-*.s' -delete 2>/dev/null || true
    touch examples/tcgen05_ptxas/main.rs
    RUSTFLAGS="-C target-cpu=$arch -C target-feature=+ptx93,+$arch" \
        cargo rustc --quiet --release --example tcgen05_ptxas --target-dir "$target_dir" \
        -- --emit=asm -C codegen-units=1
    ptx=$(find "$target_dir" -name 'tcgen05_ptxas-*.s')
    ptxas -arch="$arch" "$ptx" -o /dev/null
    echo "$arch: ok ($(grep -c 'tcgen05\.' "$ptx") tcgen05 instructions in $ptx)"
done
