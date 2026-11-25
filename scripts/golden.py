# Copyright (C) 2026 NotiChess contributors
# SPDX-License-Identifier: AGPL-3.0-only
#
# The Maia-3 model and the code this script builds on (github.com/CSSLab/maia3)
# belong to the CSSLab, University of Toronto, and are licensed under the AGPL-3.0.

"""Golden policies for `golden_policies_match_official_pytorch` in tests/uci.rs,
computed with the official Maia-3 code in PyTorch (not with the ONNX export).

    uv run --no-project --python 3.12 \
      --with torch==2.5.1 --with numpy==2.1.3 --with chess==1.11.1 \
      python scripts/golden.py --maia3-src /tmp/maia3 \
        --checkpoint maia3-5m.pt --output tests/fixtures/golden.json

(`--maia3-src` and `--checkpoint`: see scripts/export_onnx.py.)

Each case goes through `Maia3UCIEngine` of `maia3/uci.py`, launched as with
`maia3-uci --model maia3-5m --use-uci-history`: `cmd_position` rebuilds the
8-board history from the moves, `get_legal_moves_mask` and `_move_from_index`
map moves to policy indices. The probabilities are the softmax of the move
logits restricted to the legal moves, as in `Maia3UCIEngine.score_moves`.
"""

import argparse
import json
import sys
from pathlib import Path

MAIA3_COMMIT = "1e13597c42d4858b7cfd7cfdae01e297263364b2"

# (name, UCI position command, elo_self = side to move, elo_oppo)
CASES = [
    ("startpos_1500", "position startpos", 1500, 1500),
    ("startpos_600", "position startpos", 600, 600),
    ("startpos_2600", "position startpos", 2600, 2600),
    ("startpos_600_vs_2600", "position startpos", 600, 2600),
    ("black_after_e4", "position startpos moves e2e4", 1500, 1500),
    ("black_sicilian_2600_vs_600", "position startpos moves e2e4 c7c5 g1f3", 2600, 600),
    ("white_can_castle_short_history",
     "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", 1500, 1500),
    ("black_can_castle_history_truncated",
     "position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d3", 1900, 1700),
    ("black_history_exactly_8",
     "position startpos moves d2d4 g8f6 c2c4 e7e6 b1c3 f8b4 e2e3", 1500, 1500),
    ("black_long_ruy_lopez",
     "position startpos moves e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 "
     "a4b3 d7d6 c2c3 e8g8 h2h3 c6a5 b3c2 c7c5 d2d4", 2200, 2200),
    ("white_both_castles_fen",
     "position fen r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 6 9", 1500, 1500),
    ("black_both_castles_fen",
     "position fen r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 6 9", 1100, 1300),
    ("white_en_passant", "position startpos moves e2e4 a7a6 e4e5 d7d5", 1500, 1500),
    ("black_en_passant", "position startpos moves a2a3 e7e5 a3a4 e5e4 d2d4", 800, 1000),
    ("white_en_passant_from_fen",
     "position fen rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 1500, 1500),
    ("white_promotions", "position fen 1n6/P7/8/8/8/8/6k1/4K3 w - - 0 1", 1500, 1500),
    ("black_promotions", "position fen 4k3/8/8/8/8/8/1p4K1/R7 b - - 0 1", 2000, 1800),
    ("white_promotion_fen_then_moves",
     "position fen 8/4P3/8/8/8/2k5/8/K7 w - - 0 1 moves a1b1 c3d3", 1500, 1500),
    ("black_in_check", "position startpos moves e2e4 d7d5 f1b5", 1500, 1500),
    ("black_fen_then_moves",
     "position fen r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4 moves e1g1 f6e4 f1e1",
     1700, 1500),
    ("black_endgame_600", "position fen 8/5k2/8/3K4/8/8/8/6R1 b - - 0 1", 600, 600),
    ("white_endgame_2600_vs_1500", "position fen 8/5k2/8/3K4/8/8/8/6R1 w - - 0 1", 2600, 1500),
]


def expected_board(chess, command):
    """Board reached by `command`, replayed independently of the engine."""
    words = command.split()
    if words[1] == "startpos":
        board, rest = chess.Board(), words[2:]
    else:
        board, rest = chess.Board(" ".join(words[2:8])), words[8:]
    for move in rest[1:]:
        board.push_uci(move)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maia3-src", required=True, help=f"checkout of github.com/CSSLab/maia3 at {MAIA3_COMMIT}")
    parser.add_argument("--checkpoint", required=True, help="maia3-5m.pt from UofTCSSLab/Maia3-5M")
    parser.add_argument("--output", default="tests/fixtures/golden.json")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.maia3_src).resolve()))
    import chess
    import torch
    from maia3.dataset import get_legal_moves_mask
    from maia3.uci import Maia3UCIEngine, parse_args

    cfg = parse_args(["--model", "maia3-5m", "--checkpoint-path", args.checkpoint,
                      "--use-uci-history", "--device", "cpu", "--no-use-amp", "--temperature", "0"])
    engine = Maia3UCIEngine(cfg)
    engine.ensure_model_loaded()

    cases = []
    for name, command, elo_self, elo_oppo in CASES:
        engine.cmd_ucinewgame()
        engine.cmd_position(command)
        # cmd_position silently ignores a bad command: check it was applied.
        board = expected_board(chess, command)
        assert engine.board.fen() == board.fen(), name
        assert len(engine.history) == min(cfg.history, len(board.move_stack) + 1), name
        assert not board.is_game_over(), name

        engine.self_elo, engine.oppo_elo = elo_self, elo_oppo
        tokens = engine._tokens_from_history(engine.history).unsqueeze(0)
        with torch.no_grad():
            logits_move, _, _ = engine.model(tokens,
                                             torch.tensor([elo_self], dtype=torch.long),
                                             torch.tensor([elo_oppo], dtype=torch.long))
        mask = get_legal_moves_mask(engine.board, engine.all_moves_dict)
        probs = torch.softmax(logits_move[0].float().masked_fill(~mask, float("-inf")), dim=-1)

        policy = {}
        for idx in mask.nonzero().flatten().tolist():
            move = engine._move_from_index(idx)
            assert move is not None, (name, idx)
            policy[move.uci()] = float(probs[idx])
        assert sorted(policy) == sorted(m.uci() for m in board.legal_moves), name
        policy = dict(sorted(policy.items(), key=lambda kv: -kv[1]))
        cases.append({"name": name, "position": command, "elo_self": elo_self,
                      "elo_oppo": elo_oppo, "policy": policy})
        top = next(iter(policy.items()))
        print(f"{name}: {len(policy)} moves, top {top[0]} {top[1]:.3f}")

    out = {
        "generated_by": "scripts/golden.py",
        "maia3_commit": MAIA3_COMMIT,
        "checkpoint": "UofTCSSLab/Maia3-5M maia3-5m.pt",
        "torch": torch.__version__,
        "python_chess": chess.__version__,
        "cases": cases,
    }
    Path(args.output).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
