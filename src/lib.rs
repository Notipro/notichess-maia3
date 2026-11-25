// Copyright (C) 2026 NotiChess contributors
// SPDX-License-Identifier: AGPL-3.0-only
//
// The Maia-3 model and the code this encoding is derived from
// (github.com/CSSLab/maia3) belong to the CSSLab, University of Toronto,
// and are licensed under the AGPL-3.0.

//! Position parsing, encoding and model of notichess-maia3. The executable
//! (`main.rs`) puts the UCI dialogue around them; the tests also use them
//! directly, for full-precision probabilities.

pub mod encoding;
pub mod model;

use cozy_chess::util::parse_uci_move;
use cozy_chess::{Board, BoardBuilder, Color, File, Rank, Square};

/// Arguments of `position`: `startpos | fen <fen>`, then optional
/// `moves ...`, every move checked. Returns every position since the start,
/// current last.
pub fn parse_position(args: &str) -> Result<Vec<Board>, String> {
    let words: Vec<&str> = args.split_whitespace().collect();
    let split = words.iter().position(|w| *w == "moves");
    let (setup, moves) = match split {
        Some(i) => (&words[..i], &words[i + 1..]),
        None => (&words[..], &[][..]),
    };
    let mut board = match setup {
        ["startpos"] => Board::default(),
        ["fen", fen @ ..] => {
            let fen = fen.join(" ");
            let board =
                Board::from_fen(&fen, false).map_err(|e| format!("invalid FEN {fen}: {e}"))?;
            standard_castling(&board).map_err(|e| format!("invalid FEN {fen}: {e}"))?
        }
        _ => return Err(format!("invalid position command: position {args}")),
    };
    let mut history = vec![board.clone()];
    for raw in moves {
        let mv = parse_uci_move(&board, raw)
            .ok()
            .filter(|mv| board.is_legal(*mv))
            .ok_or_else(|| format!("illegal move {raw}"))?;
        board.play_unchecked(mv);
        history.push(board.clone());
    }
    Ok(history)
}

/// Castling rights as python-chess keeps them in standard chess: king on e1
/// (e8) and rook on the a or h file. cozy-chess reads a FEN with X-FEN
/// rules, so `5K1R w K` would let the f1 king castle (Chess960), which
/// python-chess, hence the model's training data, never does.
fn standard_castling(board: &Board) -> Result<Board, String> {
    let mut builder = BoardBuilder::from_board(board);
    for color in Color::ALL {
        let king_home = board.king(color) == Square::new(File::E, Rank::First.relative_to(color));
        let rights = builder.castle_rights_mut(color);
        if !king_home || rights.short != Some(File::H) {
            rights.short = None;
        }
        if !king_home || rights.long != Some(File::A) {
            rights.long = None;
        }
    }
    builder.build().map_err(|e| format!("{e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cozy_chess::util::display_uci_move;

    fn legal_moves(board: &Board) -> Vec<String> {
        let mut moves = Vec::new();
        board.generate_moves(|list| {
            moves.extend(
                list.into_iter()
                    .map(|mv| display_uci_move(board, mv).to_string()),
            );
            false
        });
        moves
    }

    #[test]
    fn replays_the_moves_from_the_start() {
        let history = parse_position("startpos moves e2e4 e7e5 g1f3").unwrap();
        assert_eq!(history.len(), 4);
        assert_eq!(history[0], Board::default());
        assert_eq!(history[3].side_to_move(), Color::Black);
        let history = parse_position("fen 8/4P3/8/8/8/2k5/8/K7 w - - 0 1 moves a1b1").unwrap();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn refuses_bad_commands_and_illegal_moves() {
        assert_eq!(
            parse_position("startpos moves e2e5").unwrap_err(),
            "illegal move e2e5"
        );
        assert!(parse_position("fen not a fen").is_err());
        assert!(parse_position("nowhere").is_err());
    }

    #[test]
    fn castling_rights_follow_python_chess() {
        // King on f1: cozy-chess alone reads `K` as a Chess960 right and the
        // castling f1h1, written f1g1, doubles the king move f1g1.
        let board = &parse_position("fen 4k3/8/8/8/8/8/8/5K1R w K - 0 1").unwrap()[0];
        assert_eq!(board.castle_rights(Color::White).short, None);
        let moves = legal_moves(board);
        assert_eq!(
            moves.iter().filter(|m| *m == "f1g1").count(),
            1,
            "{moves:?}"
        );

        // King on d1 between its rooks: no right either; Black's are kept.
        let board = &parse_position("fen r3k2r/8/8/8/8/8/8/R2K3R w KQkq - 0 1").unwrap()[0];
        let white = board.castle_rights(Color::White);
        assert_eq!((white.short, white.long), (None, None));
        let black = board.castle_rights(Color::Black);
        assert_eq!((black.short, black.long), (Some(File::H), Some(File::A)));
    }
}
