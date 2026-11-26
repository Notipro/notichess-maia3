# notichess-maia3

A small, self-contained program that plays **Maia-3 5M**, the human move
prediction model of the [CSSLab](https://github.com/CSSLab) (University
of Toronto), behind a subset of the UCI protocol. For a position, it gives the
probability that a human of a given rating plays each legal move. It does not
search.

It is used by [NotiChess](https://notichess.com) as an external engine, in a
separate process, for its games against Maia. It runs on macOS (Apple silicon
and Intel), Windows and Linux (x86_64) without Python, PyTorch or any native
library: the network runs on [tract](https://github.com/sonos/tract), a pure
Rust inference engine.

This project is not affiliated with or endorsed by the CSSLab.

## Maia-3 and attribution

Maia-3 is built on Chessformer:

> Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, Ashton
> Anderson. *Chessformer: A Unified Architecture for Chess Modeling.* The
> Fourteenth International Conference on Learning Representations (ICLR 2026).
> [arXiv:2605.19091](https://arxiv.org/abs/2605.19091),
> [OpenReview](https://openreview.net/forum?id=2ltBRzEHyd).

```bibtex
@inproceedings{monroe2026chessformer,
title={Chessformer: A Unified Architecture for Chess Modeling},
author={Daniel Monroe and George Eilender and Philip Chalmers and Zhenwei Tang and Ashton Anderson},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=2ltBRzEHyd}
}
```

- Official code: [github.com/CSSLab/maia3](https://github.com/CSSLab/maia3),
  AGPL-3.0.
- Weights: [huggingface.co/UofTCSSLab/Maia3-5M](https://huggingface.co/UofTCSSLab/Maia3-5M).
  The model card gives CC BY 4.0 for the paper and refers to the code
  repository for the code and weights license, which is the AGPL-3.0.
- Website: [maiachess.com](https://maiachess.com).

One model covers every level: it is conditioned on two continuous ratings, the
player to move and the opponent.

- **Training data.** The paper (arXiv:2605.19091v1, Section 4.1, Dataset):
  "We construct a training dataset consisting of blitz games played on the
  online chess platform Lichess from January 2023 to July 2025." The model
  card: training data from Lichess human games, January 2023 to July 2025. The
  ratings are therefore Lichess ratings.
- **Rating range.** The official code takes any rating: `interpolate_elo` in
  `maia3/models.py` clamps it to [0, 5000], and `maia3-uci` (`maia3/uci.py`)
  declares `Elo`, `SelfElo` and `OppoElo` from 0 to 5000, default 1500; the
  presets of `maia3/presets.py` set no rating. The 600 to 2600 bounds of this
  program are our choice, after the paper (Appendix A.1, Human Emulation):
  "the rating spectrum is divided into 22 bins: 20 bins uniformly spanning 600
  to 2600 Elo with 100-point intervals, plus two bins for players rated below
  600 or above 2600."

## Protocol

The program reads commands on stdin and writes on stdout, one line each.
Nothing else is ever written to stdout; diagnostics go to stderr.

| Command | Reply |
|---|---|
| `uci` | `id name notichess-maia3 <version>`, `id author ...`, the three options below, `uciok` |
| `setoption name <name> value <value>` | none |
| `isready` | loads `WeightsFile` if it changed, then `readyok`; on failure a single `error <reason>` line and nothing else |
| `ucinewgame` | none; back to the initial position |
| `position startpos [moves ...]`, `position fen <fen> [moves ...]` | none |
| `go ...` (arguments ignored) | the policy, see below |
| `quit` (or end of stdin) | the program exits |

Options:

```
option name WeightsFile type string default <empty>
option name EloSelf type spin default 1500 min 600 max 2600
option name EloOppo type spin default 1500 min 600 max 2600
```

- `WeightsFile`: path of `maia3-5m.onnx`.
- `EloSelf`: rating of the side to move (Maia, when she plays); `EloOppo`:
  rating of its opponent. Integers; values outside 600 to 2600 are clamped.
- Any other option (`VerboseMoveStats`, `PolicyTemperature`, `Threads`, ...)
  is accepted and ignored.

`go` replies one line per legal move, most likely first, then the number of
legal moves, then the most likely move:

```
info string e2e4 (P: 64.2814%)
info string d2d4 (P: 24.1652%)
...
info string node (20)
bestmove e2e4
```

Probabilities are the softmax of the move logits over the legal moves only,
without temperature, in percent with four decimals. Moves are in standard UCI:
castling as the king's move (`e1g1`), promotions in lower case (`e7e8q`).
Sampling a move from the policy is left to the caller. With no legal move
(checkmate, stalemate) the reply is `info string node (0)` and
`bestmove 0000`. The policy is given whenever a legal move exists, including
with insufficient material, after a fivefold repetition or under the 75-move
rule, where `maia3-uci` answers `bestmove 0000` (python-chess
`is_game_over()`): judging the end of the game is the caller's job.

Details:

- `isready` with an empty `WeightsFile` replies `readyok`: nothing to load.
- `go` without a loaded model, or after a refused `position` (bad FEN,
  illegal move), replies a single `error <reason>` line.
- An `error` line names the weights file by its file name only, never its
  folder: it ends up in the caller's logs.
- A FEN keeps its castling rights only for a king on e1 (e8) and a rook on
  the a or h file, as python-chess does in standard chess.
- A line that is not valid UTF-8 is read with replacement characters.
- Maia-3 reads the current position and the 7 before it. They come from the
  moves of the `position` command, starting at `startpos` or at the FEN; a
  shorter history is padded with its first position, as the official code
  does (`maia3-uci --use-uci-history`). So always send the whole game from its
  start, not only the current FEN.
- `go` is synchronous (a few milliseconds); `stop` is accepted and does
  nothing.

## Model file

`maia3-5m.onnx` is not in this repository. It lives in the permanent release
[`model-maia3-5m`](../../releases/tag/model-maia3-5m), with
`maia3-5m.onnx.sha256` and `LICENSE`. That release is created once, by hand,
and never replaced: the CI and the release workflow download the model from
it and check it against the committed [`maia3-5m.onnx.sha256`](maia3-5m.onnx.sha256).
Another model would get another release tag and another hash file.

```bash
shasum -a 256 -c maia3-5m.onnx.sha256
gh release create model-maia3-5m --target main --latest=false \
  --title "Maia3-5M model (ONNX)" \
  --notes "maia3-5m.onnx: the Maia3-5M weights of the CSSLab, University of Toronto (huggingface.co/UofTCSSLab/Maia3-5M, AGPL-3.0), exported to ONNX by scripts/export_onnx.py. Used by every notichess-maia3 release." \
  maia3-5m.onnx maia3-5m.onnx.sha256 LICENSE
```

It can be rebuilt from the official checkpoint:

| | |
|---|---|
| Checkpoint | `maia3-5m.pt` of [UofTCSSLab/Maia3-5M](https://huggingface.co/UofTCSSLab/Maia3-5M), commit `b6559de2398d7140b985f28fd2c19fb5e47ddabe`, 20,968,049 bytes, SHA-256 `ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f` |
| Official code | [CSSLab/maia3](https://github.com/CSSLab/maia3) commit `1e13597c42d4858b7cfd7cfdae01e297263364b2` |
| Export | [`scripts/export_onnx.py`](scripts/export_onnx.py), PyTorch 2.5.1, ONNX opset 17, float32, batch 1 |
| Result | `maia3-5m.onnx`, 20,857,309 bytes, SHA-256 in [`maia3-5m.onnx.sha256`](maia3-5m.onnx.sha256) |

```bash
git clone https://github.com/CSSLab/maia3 /tmp/maia3
git -C /tmp/maia3 checkout 1e13597c42d4858b7cfd7cfdae01e297263364b2
curl -L -o maia3-5m.pt \
  https://huggingface.co/UofTCSSLab/Maia3-5M/resolve/b6559de2398d7140b985f28fd2c19fb5e47ddabe/maia3-5m.pt
uv run --no-project --python 3.12 \
  --with torch==2.5.1 --with numpy==2.1.3 --with chess==1.11.1 --with onnx==1.17.0 \
  python scripts/export_onnx.py --maia3-src /tmp/maia3 \
    --checkpoint maia3-5m.pt --output maia3-5m.onnx
shasum -a 256 -c maia3-5m.onnx.sha256
```

The export is deterministic (same bytes on two runs). It builds and loads the
model with the official code and changes the forward pass in form only, after
checking that both give the same logits (largest gap 1.1e-5):

- RMSNorm written out, since PyTorch 2.5 cannot export `aten::rms_norm` to
  opset 17;
- the input projection split into three summed products (board, own rating,
  opponent rating) instead of one product of their concatenation, which tract
  0.23 cannot optimize;
- the ponder (thinking time) head left out.

## Building

```bash
cargo build --release
```

The toolchain is pinned by `rust-toolchain.toml` (Rust 1.95.0). Dependencies,
exact versions:

| Crate | Version | License | Use |
|---|---|---|---|
| [tract-onnx](https://crates.io/crates/tract-onnx) | 0.23.10 (2026-11-05) | MIT or Apache-2.0 | loads and runs the ONNX graph |
| [cozy-chess](https://crates.io/crates/cozy-chess) | 0.3.4 (2024-04-05) | MIT | FEN, legal moves, UCI move notation |
| [serde_json](https://crates.io/crates/serde_json) | 1.0.149, tests only | MIT or Apache-2.0 | reads the golden file |

cozy-chess was chosen over shakmaty because NotiChess already uses it to
validate positions, so both sides accept the same FENs and moves, and because
it is MIT licensed.

On Windows the C runtime is linked statically (`.cargo/config.toml`), so the
executable runs without the Visual C++ redistributable.

On Linux the published executable is built for musl, statically (see
[Linux](#linux)):

```bash
sudo apt-get install musl-tools   # musl-gcc, for the assembly kernels of tract
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

From macOS, [cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild)
builds the same target: `cargo zigbuild --release --target x86_64-unknown-linux-musl`.

## Linux

One executable, `notichess-maia3-linux-x86_64`, for every x86_64
distribution:

- **Static.** Built for `x86_64-unknown-linux-musl`: the musl C library
  (1.2.5, the one Rust 1.95 ships for this target) is linked in, with no
  dynamic loader and no glibc. `file` says `statically linked` (or
  `static-pie linked`) and `ldd` says `not a dynamic executable` (or
  `statically linked`); the CI checks both.
- **Kernel.** Linux 3.2 or later, the minimum of every Rust Linux target
  ("`*-linux-musl*` targets are only affected by the kernel requirements, not
  the glibc requirements",
  [Rust blog, 2022-08-01](https://blog.rust-lang.org/2022/08/01/Increasing-glibc-kernel-requirements/)).
  The containers of the CI share the runner's kernel, so old kernels are not
  tested.
- **No system dependency.** The only native code is the x86_64 assembly of
  tract-linalg, assembled by its build script (`cc` crate, `musl-gcc`) and
  calling no library. No `*-sys` crate, no `dlopen`, no network. Besides
  the weights, tract reads `/sys/devices/system/cpu` for the cache sizes,
  and does without it.
- **Processor.** The Rust code targets baseline x86-64 (SSE2). tract-linalg
  builds its AVX, FMA and AVX2, AVX-512 (and VNNI, AMX) kernels into the
  executable and chooses at run time from CPUID (`is_x86_feature_detected!`,
  `tract_linalg::x86_64::isa_set`): AVX-512F, else FMA with AVX2, else AVX,
  else its generic Rust kernels. A processor without AVX2, or without AVX,
  runs the generic kernels, slower, with the same policy within rounding.
  The CI runs the golden test under QEMU with the `qemu64` processor (SSE3, no
  AVX). The same holds for the Windows and macOS x86_64 executables.
- **Execute bit.** A download keeps no file mode, so the release asset is
  not executable as downloaded: whoever extracts it sets the bit
  (`chmod 755`) before running it.
- **Tested.** The `linux-compat` job of the CI builds the musl executable,
  runs every test with the model, checks `file` and `ldd`, then runs a UCI
  dialogue up to `bestmove`, with the model mounted read-only and no network,
  in `debian:10`, `ubuntu:18.04`, `centos:7`, `rockylinux:8`, `alpine:3`,
  `fedora:latest` and `opensuse/leap:15`.
- **No linux-aarch64 yet**, since NotiChess does not ship Linux on ARM. To
  add it: an `ubuntu-22.04-arm` runner, target `aarch64-unknown-linux-musl`
  and asset `notichess-maia3-linux-aarch64` in the matrix of `release.yml`,
  the same runner in the test matrix and a second `linux-compat` run in
  `ci.yml`; tract then builds its NEON kernels, plus the SVE, SME and dotprod
  ones its build script can assemble, also chosen at run time.

## Tests

```bash
cargo test                                               # encoding and protocol
MAIA3_ONNX=/path/to/maia3-5m.onnx cargo test --release   # plus the model tests
```

Without `MAIA3_ONNX` the model tests are skipped, unless
`MAIA3_REQUIRE_MODEL=1` (set by both workflows), which makes them fail. The
CI runs them on macOS (arm64 and x86_64), Windows and Ubuntu 22.04, and on the
musl build in its `linux-compat` job ([Linux](#linux)).

- Unit tests (`src/encoding.rs`, `src/lib.rs`): board planes, mirroring when
  Black is to move, history padding and truncation, policy indices of
  castling and promotions, softmax over legal moves, position parsing,
  castling rights.
- Protocol tests (`tests/uci.rs`): the executable is run as a process and fed
  whole dialogues, including missing or corrupt weights files in a folder
  with spaces and accents, and a line that is not UTF-8.
- Golden test (`tests/uci.rs`): `tests/fixtures/golden.json` holds the policy
  of 22 positions (opening, castling, en passant, promotions, Black to move,
  FEN starts, short and long histories, ratings from 600 to 2600) computed by
  [`scripts/golden.py`](scripts/golden.py) with the official PyTorch code.
  Every legal move's probability must match within 1e-5, and within 5e-4 in
  log for probabilities above 1e-6, in full precision through the library;
  and within 1e-5 through the executable's output (four decimals of
  percent). Largest gaps on Apple M4: 2.2e-6, 1.3e-5 in log. Swapping the
  bishop and knight underpromotions, or keeping 7 positions of history
  instead of 8, fails it.

To regenerate the golden file (same prerequisites as the export, without
`onnx`):

```bash
uv run --no-project --python 3.12 \
  --with torch==2.5.1 --with numpy==2.1.3 --with chess==1.11.1 \
  python scripts/golden.py --maia3-src /tmp/maia3 --checkpoint maia3-5m.pt
```

## Releases

Pushing a `v*` tag matching the crate version runs
`.github/workflows/release.yml`. It builds each executable on its platform,
runs every test there with the model of `model-maia3-5m`, then publishes a
release with

- `notichess-maia3-macos-aarch64`
- `notichess-maia3-macos-x86_64`
- `notichess-maia3-windows-x86_64.exe`
- `notichess-maia3-linux-x86_64`
- `SHA256SUMS` of the four executables
- `LICENSE` and `THIRD-PARTY-NOTICES`.

The model stays in `model-maia3-5m`, which must exist before the first tag.

v0.1.1 adds the Linux executable. The protocol, the options and the policy
are those of v0.1.0.

## Third-party notices

[`THIRD-PARTY-NOTICES`](THIRD-PARTY-NOTICES) lists every crate built into the
executables with its license and license texts. It is generated by
[`scripts/third_party_notices.py`](scripts/third_party_notices.py) from
`Cargo.lock`, and the CI checks it is up to date. `licenses/` holds the
license texts that a crate package does not ship (cozy-chess, copied from its
repository at the released commit), and the `COPYRIGHT` file of musl 1.2.5,
the C library linked into the Linux executable.

## Performance

Apple M4, release build, one thread:

| | |
|---|---|
| Start to `uciok` | 3 ms |
| Model load (`setoption WeightsFile` + `isready`) | 72 to 75 ms |
| `position` + `go`, round trip, 300 calls | median 2.5 to 2.7 ms, 90th percentile 2.8 to 3.3 ms |
| Peak memory | 66 MB |
| Executable, stripped (LTO) | 14.6 MB |

Linux, Intel Core i9-10900K (AVX2 and FMA), the musl executable, one thread,
five runs alternated with a glibc build of the same code:

| | musl (published) | glibc 2.17 |
|---|---|---|
| Model load | 190 to 200 ms | 145 to 159 ms |
| `position` + `go`, round trip, 300 calls | median 14.0 to 14.9 ms, 90th percentile 14.8 to 16.1 ms | median 9.6 to 10.6 ms, 90th percentile 10.4 to 11.6 ms |
| Minor page faults, 300 calls | 1.2 million | 14 thousand |
| Peak memory | 49 MB | 60 MB |
| Executable, stripped (LTO) | 20.3 MB | 21.3 MB |

The gap is the allocator: musl gives the buffers of each `go` back to the
system and faults them in again at the next one. It costs about 4 ms a move,
for an executable that runs on every distribution.

## License

Copyright (C) 2026 NotiChess contributors.

Licensed under the [GNU Affero General Public License version 3](LICENSE).
The Maia-3 model and the official code the encoding is derived from belong to
the CSSLab, University of Toronto, and are licensed under the AGPL-3.0. The
source of each published executable is this repository at the release tag.
