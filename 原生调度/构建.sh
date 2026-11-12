#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
HOME_DIR=$(dirname -- "$HERE")
CACHE=${HQ_FIFO_BUILD_CACHE:-/public_bme2/Share200T/管吉松/envs/af3-build-cache/hq-priority}
SOURCE="$CACHE/hyperqueue-0.27.0"
TOOLCHAIN="$CACHE/rust"
AF3=/public_bme2/Share200T/管吉松/envs/af3
export PATH="$TOOLCHAIN/bin:$AF3/bin:$PATH"
export LD_LIBRARY_PATH="$AF3/lib:${LD_LIBRARY_PATH:-}"
export CARGO_HOME="$CACHE/cargo"
export CARGO_TARGET_DIR="$CACHE/target"
export CARGO_BUILD_JOBS=1
export CARGO_HTTP_MULTIPLEXING=false
DIST_SERVER=${HQ_FIFO_RUST_DIST_SERVER:-https://rsproxy.cn}
REGISTRY=(--config 'source.crates-io.replace-with="rsproxy"'
          --config 'source.rsproxy.registry="sparse+https://rsproxy.cn/index/"')

case "${1:-}" in
    bootstrap)
        mkdir -p "$CACHE" "$TOOLCHAIN"
        for component in cargo rustc rust-std; do
            case "$component" in
                cargo) checksum=e74edd2cf7d0f1f1383b4f00eb90c843750bc489e2ccf7214e6476678a907425 ;;
                rustc) checksum=8426a3d170a5879f5682f5fbdd024a1779b3951e7baba685af2d6dc32a6dfc15 ;;
                rust-std) checksum=047ea7098803d3500fa1072e9cee5392697e21525559e4458128a2bf874aa382 ;;
            esac
            package="$component-1.95.0-x86_64-unknown-linux-gnu"
            archive="$CACHE/$package.tar.xz"
            if [[ ! -f "$archive" ]] || ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status; then
                curl -fL --continue-at - --connect-timeout 10 --max-time 900 \
                    "$DIST_SERVER/dist/2026-04-16/$package.tar.xz" -o "$archive"
            fi
            printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check
            tar -xJf "$archive" -C "$CACHE"
            bash "$CACHE/$package/install.sh" --prefix="$TOOLCHAIN" --disable-ldconfig
        done
        rustc --version
        cargo --version
        cargo "${REGISTRY[@]}" fetch --locked --manifest-path "$SOURCE/Cargo.toml"
        ;;
    导出补丁)
        ORIGINAL=$(mktemp -d)
        PATCH_FILES=(crates/hyperqueue/src/client/resources.rs crates/tako/src/internal/common/priority.rs crates/tako/src/internal/common/resources/map.rs crates/tako/src/internal/common/resources/request.rs crates/tako/src/internal/scheduler/main.rs crates/tako/src/internal/scheduler/mapping.rs crates/tako/src/internal/scheduler/mod.rs crates/tako/src/internal/server/workerload.rs crates/tako/src/internal/worker/resources/allocator.rs crates/tako/src/internal/worker/resources/pool.rs)
        ARCHIVE_FILES=()
        for file in "${PATCH_FILES[@]}"; do ARCHIVE_FILES+=("hyperqueue-0.27.0/$file"); done
        tar -xzf "$CACHE/hyperqueue-0.27.0.tar.gz" -C "$ORIGINAL" "${ARCHIVE_FILES[@]}"
        PATCH_TMP=$(mktemp "$HERE/显存范围.patch.XXXXXX")
        for file in "${PATCH_FILES[@]}"; do
            if diff -u --label "a/$file" --label "b/$file" "$ORIGINAL/hyperqueue-0.27.0/$file" "$SOURCE/$file" >> "$PATCH_TMP"; then
                :
            else
                RESULT=$?
                [[ "$RESULT" -eq 1 ]] || exit "$RESULT"
            fi
        done
        git -C "$SOURCE" apply --check --reverse "$PATCH_TMP"
        mv "$PATCH_TMP" "$HERE/显存范围.patch"
        rm -rf "$ORIGINAL"
        ;;
    build)
        git -C "$SOURCE" apply --check --reverse "$HERE/FIFO.patch"
        git -C "$SOURCE" apply --check --reverse "$HERE/显存范围.patch"
        export LIBCLANG_PATH=/usr/lib64/clang-private
        export LD_LIBRARY_PATH="$LIBCLANG_PATH:$LD_LIBRARY_PATH"
        export CC="$AF3/bin/x86_64-conda-linux-gnu-cc"
        export CXX="$AF3/bin/x86_64-conda-linux-gnu-c++"
        export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="$CC"
        cargo "${REGISTRY[@]}" build --release --locked --offline --manifest-path "$SOURCE/Cargo.toml" \
            -p hyperqueue --bin hq
        install -m 755 "$CARGO_TARGET_DIR/release/hq" "$HOME_DIR/bin/hq.fifo.new"
        "$HOME_DIR/bin/hq.fifo.new" --version
        HQ_TEST_BINARY="$HOME_DIR/bin/hq.fifo.new" PYTHONDONTWRITEBYTECODE=1 \
            "$AF3/bin/python" -m unittest discover -s "$HOME_DIR/测试" -p test_hq_fifo.py -v
        mv "$HOME_DIR/bin/hq.fifo.new" "$HOME_DIR/bin/hq.fifo"
        ;;
    *)
        printf '用法：bash %s bootstrap|build|导出补丁\n' "$0" >&2
        exit 2
        ;;
esac