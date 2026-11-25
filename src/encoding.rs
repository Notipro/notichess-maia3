// Copyright (C) 2026 NotiChess contributors
// SPDX-License-Identifier: AGPL-3.0-only
//
// The Maia-3 model and the code this encoding is derived from
// (github.com/CSSLab/maia3) belong to the CSSLab, University of Toronto,
// and are licensed under the AGPL-3.0.

//! Model input and output encoding, after `maia3/dataset.py` and
//! `maia3/utils.py` of CSSLab/maia3 (commit 1e13597c).
//!
//! Input: 8 boards (current and 7 previous), oldest first, each as 64 squares
//! x 12 one-hot planes (white P N B R Q K, then black), square = rank * 8 +
//! file. Every board is seen from its own side to move: when Black is to move
//! it is mirrored like python-chess `board.mirror()` (ranks flipped, colours
//! swapped). A short history is padded on the left with its oldest board.
//! Output: 4352 move logits in the side-to-move frame, `from * 64 + to`, then
//! the promotions `4096 + from_file * 32 + to_file * 4 + piece` (q, r, b, n).

use cozy_chess::util::display_uci_move;
use cozy_chess::{Board, Color, Move, Piece};

pub const HISTORY: usize = 8;
/// 12 planes per board, plus the `clk_ponder` column the model slices off.
pub const TOKEN_DIM: usize = 12 * HISTORY + 1;
pub const POLICY_SIZE: usize = 64 * 64 + 8 * 8 * 4;

/// Model input `[64][TOKEN_DIM]`, row-major. `history`: every position of the
/// game from its start (FEN or initial position), current last, as
/// `Maia3UCIEngine` keeps it with `--use-uci-history`.
pub fn encode(history: &[Board]) -> Vec<f32> {
    assert!(!history.is_empty());
    let kept = &history[history.len().saturating_sub(HISTORY)..];
    let pad = HISTORY - kept.len();
    let mut tokens = vec![0.0; 64 * TOKEN_DIM];
    for slot in 0..HISTORY {
        planes(&kept[slot.saturating_sub(pad)], &mut tokens, 12 * slot);
    }
    tokens
}

/// `tokenize_board`: one board into the 12 columns starting at `offset`.
fn planes(board: &Board, tokens: &mut [f32], offset: usize) {
    let mirror = board.side_to_move() == Color::Black;
    for color in Color::ALL {
        for piece in Piece::ALL {
            for square in board.colored_pieces(color, piece) {
                let (square, color) = if mirror {
                    (square.flip_rank(), !color)
                } else {
                    (square, color)
                };
                let channel = piece as usize + if color == Color::Black { 6 } else { 0 };
                tokens[square as usize * TOKEN_DIM + offset + channel] = 1.0;
            }
        }
    }
}

/// Index of a legal move of `board` in the policy (`get_all_possible_moves`
/// order, Black's moves mirrored). Castling counts as king to g or c file,
/// as python-chess writes it.
pub fn policy_index(board: &Board, mv: Move) -> usize {
    let standard: Move = display_uci_move(board, mv)
        .to_string()
        .parse()
        .expect("cozy-chess writes parsable UCI");
    let side = board.side_to_move();
    let (from, to) = (
        standard.from.relative_to(side),
        standard.to.relative_to(side),
    );
    match standard.promotion {
        None => from as usize * 64 + to as usize,
        Some(piece) => {
            let slot = match piece {
                Piece::Queen => 0,
                Piece::Rook => 1,
                Piece::Bishop => 2,
                _ => 3,
            };
            4096 + from.file() as usize * 32 + to.file() as usize * 4 + slot
        }
    }
}

/// Every legal move with its probability, most likely first: softmax of the
/// logits of the legal moves only, no temperature.
pub fn policy(board: &Board, logits: &[f32]) -> Vec<(Move, f64)> {
    let mut moves = Vec::new();
    board.generate_moves(|list| {
        moves.extend(list);
        false
    });
    let scored: Vec<(Move, f64)> = moves
        .into_iter()
        .map(|mv| (mv, logits[policy_index(board, mv)] as f64))
        .collect();
    let max = scored
        .iter()
        .map(|(_, l)| *l)
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = scored.iter().map(|(_, l)| (l - max).exp()).sum();
    let mut out: Vec<(Move, f64)> = scored
        .into_iter()
        .map(|(mv, l)| (mv, (l - max).exp() / sum))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::util::parse_uci_move;
    use cozy_chess::Square;

    fn board(fen: &str) -> Board {
        Board::from_fen(fen, false).unwrap()
    }

    fn play(start: Board, moves: &[&str]) -> Vec<Board> {
        let mut history = vec![start];
        for m in moves {
            let mut next = history.last().unwrap().clone();
            next.play(parse_uci_move(&next, m).unwrap());
            history.push(next);
        }
        history
    }

    fn at(tokens: &[f32], square: Square, slot: usize, channel: usize) -> f32 {
        tokens[square as usize * TOKEN_DIM + 12 * slot + channel]
    }

    fn index(b: &Board, uci: &str) -> usize {
        policy_index(b, parse_uci_move(b, uci).unwrap())
    }

    #[test]
    fn initial_position_fills_every_slot() {
        let tokens = encode(&[Board::default()]);
        assert_eq!(tokens.len(), 64 * 97);
        for slot in 0..HISTORY {
            assert_eq!(at(&tokens, Square::A1, slot, 3), 1.0, "white rook");
            assert_eq!(at(&tokens, Square::E8, slot, 11), 1.0, "black king");
            assert_eq!(at(&tokens, Square::D2, slot, 0), 1.0, "white pawn");
        }
        assert_eq!(tokens.iter().sum::<f32>(), 32.0 * 8.0);
        assert!((0..64).all(|sq| tokens[sq * TOKEN_DIM + 96] == 0.0));
    }

    #[test]
    fn black_to_move_is_mirrored_and_history_padded() {
        let history = play(Board::default(), &["e2e4"]);
        let tokens = encode(&history);
        // Slots 0 to 6: the initial position (padding, then itself), White to move.
        for slot in 0..7 {
            assert_eq!(at(&tokens, Square::E2, slot, 0), 1.0);
            assert_eq!(at(&tokens, Square::E1, slot, 5), 1.0);
        }
        // Slot 7: after 1.e4, Black to move: ranks flipped, colours swapped.
        assert_eq!(at(&tokens, Square::E1, 7, 5), 1.0, "black king as own king");
        assert_eq!(
            at(&tokens, Square::E8, 7, 11),
            1.0,
            "white king as opponent"
        );
        assert_eq!(
            at(&tokens, Square::E5, 7, 6),
            1.0,
            "e4 pawn seen from Black"
        );
        assert_eq!(at(&tokens, Square::E7, 7, 6), 0.0);
    }

    #[test]
    fn long_history_keeps_the_last_eight_boards() {
        let moves = [
            "g1f3", "g8f6", "f3g1", "f6g8", "b1c3", "b8c6", "c3b1", "c6b8", "e2e4",
        ];
        let history = play(Board::default(), &moves);
        assert_eq!(history.len(), 10);
        let tokens = encode(&history);
        // Slot 0 is history[2], after 1.Nf3 Nf6, White to move.
        assert_eq!(at(&tokens, Square::F3, 0, 1), 1.0);
        assert_eq!(at(&tokens, Square::G1, 0, 1), 0.0);
        assert_eq!(at(&tokens, Square::F6, 0, 7), 1.0);
        assert_eq!(tokens, encode(&history[2..]));
    }

    #[test]
    fn move_indices_follow_the_official_vocabulary() {
        let start = Board::default();
        assert_eq!(index(&start, "e2e4"), 12 * 64 + 28);
        let after_e4 = play(start, &["e2e4"]).pop().unwrap();
        assert_eq!(index(&after_e4, "e7e5"), 12 * 64 + 28, "mirrored to e2e4");

        let castles = board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(index(&castles, "e1g1"), 4 * 64 + 6);
        assert_eq!(index(&castles, "e1c1"), 4 * 64 + 2);
        let castles = board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
        assert_eq!(index(&castles, "e8g8"), 4 * 64 + 6, "mirrored to e1g1");
        assert_eq!(index(&castles, "e8c8"), 4 * 64 + 2);

        let promo = board("1n6/P7/8/8/8/8/6k1/4K3 w - - 0 1");
        assert_eq!(index(&promo, "a7a8q"), 4096);
        assert_eq!(index(&promo, "a7b8n"), 4096 + 4 + 3);
        let promo = board("4k3/8/8/8/8/8/1p4K1/R7 b - - 0 1");
        assert_eq!(index(&promo, "b2a1q"), 4096 + 32, "mirrored to b7a8q");
        assert_eq!(index(&promo, "b2b1r"), 4096 + 32 + 4 + 1);
    }

    #[test]
    fn policy_is_a_softmax_over_legal_moves_only() {
        let start = Board::default();
        // Illegal moves get the highest logit: they must not count.
        let mut logits = vec![100.0; POLICY_SIZE];
        start.generate_moves(|list| {
            list.into_iter()
                .for_each(|mv| logits[policy_index(&start, mv)] = 0.0);
            false
        });
        logits[index(&start, "e2e4")] = 2.0;
        for mv in ["d2d4", "a2a3", "b2b3"] {
            logits[index(&start, mv)] = 1.0;
        }
        let p = policy(&start, &logits);
        assert_eq!(p.len(), 20);
        assert_eq!(display_uci_move(&start, p[0].0).to_string(), "e2e4");
        assert!((p.iter().map(|(_, q)| q).sum::<f64>() - 1.0).abs() < 1e-12);
        let e = std::f64::consts::E;
        let z = e * e + 3.0 * e + 16.0;
        assert!((p[0].1 - e * e / z).abs() < 1e-12);
    }
}
