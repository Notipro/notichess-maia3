// Copyright (C) 2026 NotiChess contributors
// SPDX-License-Identifier: AGPL-3.0-only
//
// The Maia-3 model and the code this encoding is derived from
// (github.com/CSSLab/maia3) belong to the CSSLab, University of Toronto,
// and are licensed under the AGPL-3.0.

//! The engine as NotiChess sees it: a process spoken to on stdin and stdout,
//! plus the golden test against the official PyTorch code.
//!
//! Tests that need the model read its path from `MAIA3_ONNX`. Without it they
//! are skipped (they pass, with a note on stderr), unless
//! `MAIA3_REQUIRE_MODEL=1`, set by the CI, which makes them fail:
//!
//!     MAIA3_ONNX=/path/to/maia3-5m.onnx cargo test --release

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use cozy_chess::util::display_uci_move;
use notichess_maia3::model::Model;
use notichess_maia3::{encoding, parse_position};

/// Largest absolute gap allowed on any probability.
const ABS_TOLERANCE: f64 = 1e-5;
/// Largest gap allowed on ln(probability), for probabilities above
/// `LOG_FLOOR`: small probabilities must match relatively too.
const LOG_TOLERANCE: f64 = 5e-4;
const LOG_FLOOR: f64 = 1e-6;

/// Sends `input` (stdin is then closed) and returns every stdout line.
fn dialogue(input: impl AsRef<[u8]>) -> Vec<String> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_notichess-maia3"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("engine starts");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_ref())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "exit status {}", output.status);
    String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

fn model() -> Option<String> {
    let path = std::env::var("MAIA3_ONNX").ok();
    if path.is_none() {
        assert!(
            std::env::var("MAIA3_REQUIRE_MODEL").as_deref() != Ok("1"),
            "MAIA3_REQUIRE_MODEL=1 but MAIA3_ONNX is not set"
        );
        eprintln!("MAIA3_ONNX is not set: test skipped");
    }
    path
}

/// One `go` reply: (move, probability 0..1) in output order, node count, bestmove.
struct Reply {
    policy: Vec<(String, f64)>,
    nodes: usize,
    bestmove: String,
}

/// Splits the lines that follow `go` commands into replies, checking the
/// exact line format NotiChess parses. Panics on any other line.
fn replies(lines: &[String]) -> Vec<Reply> {
    let mut out = Vec::new();
    let mut policy = Vec::new();
    let mut nodes = None;
    for line in lines {
        if let Some(rest) = line.strip_prefix("info string node (") {
            nodes = Some(rest.strip_suffix(')').unwrap().parse().unwrap());
        } else if let Some(rest) = line.strip_prefix("info string ") {
            let (mv, p) = rest.split_once(" (P: ").expect(line);
            let p = p.strip_suffix("%)").expect(line);
            assert!(is_uci_move(mv), "{line}");
            assert_eq!(p.split_once('.').map(|(_, d)| d.len()), Some(4), "{line}");
            policy.push((mv.to_string(), p.parse::<f64>().unwrap() / 100.0));
        } else if let Some(mv) = line.strip_prefix("bestmove ") {
            out.push(Reply {
                policy: std::mem::take(&mut policy),
                nodes: nodes.take().expect("node line before bestmove"),
                bestmove: mv.to_string(),
            });
        } else {
            panic!("unexpected line: {line}");
        }
    }
    assert!(policy.is_empty() && nodes.is_none(), "unterminated reply");
    out
}

fn is_uci_move(s: &str) -> bool {
    let b = s.as_bytes();
    let square = |f: u8, r: u8| (b'a'..=b'h').contains(&f) && (b'1'..=b'8').contains(&r);
    (b.len() == 4 || (b.len() == 5 && b"qrbn".contains(&b[4])))
        && square(b[0], b[1])
        && square(b[2], b[3])
}

const HANDSHAKE: [&str; 6] = [
    "id name notichess-maia3 ",
    "id author ",
    "option name WeightsFile type string default <empty>",
    "option name EloSelf type spin default 1500 min 600 max 2600",
    "option name EloOppo type spin default 1500 min 600 max 2600",
    "uciok",
];

fn assert_handshake(lines: &[String]) {
    assert_eq!(
        lines[0],
        format!("id name notichess-maia3 {}", env!("CARGO_PKG_VERSION"))
    );
    assert!(lines[1].starts_with(HANDSHAKE[1]), "{}", lines[1]);
    assert_eq!(lines[2..6], HANDSHAKE[2..]);
}

#[test]
fn handshake_then_ready_without_weights() {
    // NotiChess sends `isready` right after `uciok`, before any option.
    let lines = dialogue(
        "uci\nisready\n\
         setoption name VerboseMoveStats value true\n\
         setoption name PolicyTemperature value 1.0\n\
         setoption name Threads value 1\n\
         setoption name EloSelf value 1800\n\
         setoption name EloOppo value 99999999999\n\
         ucinewgame\nposition startpos moves e2e4\nisready\nquit\n",
    );
    assert_handshake(&lines);
    assert_eq!(lines[6..], ["readyok", "readyok"]);
}

#[test]
fn missing_weights_reply_error_and_nothing_else() {
    let lines = dialogue(
        "uci\nsetoption name WeightsFile value /nonexistent/dir/maia3-5m.onnx\n\
         isready\nposition startpos\ngo nodes 1\nquit\n",
    );
    assert_handshake(&lines);
    assert_eq!(lines.len(), 8, "{lines:?}");
    for line in &lines[6..] {
        assert!(
            line.starts_with("error cannot load maia3-5m.onnx: "),
            "{line}"
        );
        assert!(!line.contains("nonexistent"), "{line}");
    }
}

#[test]
fn load_errors_name_the_file_never_its_folder() {
    // Spaces and accents, like a user's `Application Support` folder.
    let folder = std::env::temp_dir().join(format!(
        "notichess-maia3 test {}/Application Support/Élodie é",
        std::process::id()
    ));
    std::fs::create_dir_all(&folder).unwrap();
    let corrupt = folder.join("bad model é.onnx");
    std::fs::write(&corrupt, b"not an onnx file").unwrap();
    let missing = folder.join("missing model.onnx");

    for (path, name) in [
        (&corrupt, "bad model é.onnx"),
        (&missing, "missing model.onnx"),
    ] {
        let lines = dialogue(format!(
            "uci\nsetoption name WeightsFile value {}\nisready\nquit\n",
            path.display()
        ));
        assert_eq!(lines.len(), 7, "{lines:?}");
        let error = &lines[6];
        assert!(
            error.starts_with(&format!("error cannot load {name}: ")),
            "{error}"
        );
        for secret in ["Application Support", "Élodie", "notichess-maia3 test"] {
            assert!(!error.contains(secret), "{error}");
        }
    }
    std::fs::remove_dir_all(folder.parent().unwrap().parent().unwrap()).unwrap();
}

#[test]
fn go_without_weights_is_an_error() {
    let lines = dialogue("uci\nposition startpos\ngo nodes 1\nquit\n");
    assert_eq!(lines[6..], ["error WeightsFile is not set"]);
}

#[test]
fn a_line_that_is_not_utf8_is_skipped() {
    let lines = dialogue(b"\xff\xfe\xfd garbage\nuci\nisready\nquit\n");
    assert_handshake(&lines);
    assert_eq!(lines[6..], ["readyok"]);
}

#[test]
fn eof_ends_the_engine() {
    assert_eq!(dialogue("uci\n").len(), 6);
}

#[test]
fn full_game_dialogue() {
    let Some(onnx) = model() else { return };
    let lines = dialogue(format!(
        "uci\nisready\n\
         setoption name WeightsFile value {onnx}\n\
         setoption name VerboseMoveStats value true\n\
         setoption name PolicyTemperature value 1.0\n\
         setoption name Threads value 1\n\
         setoption name EloSelf value 1500\n\
         setoption name EloOppo value 1500\n\
         isready\nucinewgame\n\
         position startpos\ngo nodes 1\n\
         position startpos moves e2e4 e7e5 g1f3 b8c6 f1c4 f8c5\ngo nodes 1\n\
         position fen 4k3/8/8/8/8/8/1p4K1/R7 b - - 0 1\ngo nodes 1\n\
         position fen 7k/5Q2/6K1/8/8/8/8/8 b - - 0 1\ngo nodes 1\n\
         position startpos moves e2e4 e7e5 e1e3\ngo nodes 1\n\
         quit\n"
    ));
    assert_handshake(&lines);
    assert_eq!(lines[6..8], ["readyok", "readyok"]);
    assert_eq!(lines.last().unwrap(), "error illegal move e1e3");
    let replies = replies(&lines[8..lines.len() - 1]);
    assert_eq!(replies.len(), 4);
    for r in &replies[..3] {
        assert_eq!(r.nodes, r.policy.len());
        let sum: f64 = r.policy.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum {sum}");
        assert!(r.policy.windows(2).all(|w| w[0].1 >= w[1].1));
        assert_eq!(r.bestmove, r.policy[0].0);
    }
    assert_eq!(replies[0].nodes, 20);
    assert!(replies[1].policy.iter().any(|(m, _)| m == "e1g1"));
    let promotions: Vec<&str> = replies[2]
        .policy
        .iter()
        .map(|(m, _)| m.as_str())
        .filter(|m| m.len() == 5)
        .collect();
    assert_eq!(promotions.len(), 8, "{promotions:?}");
    // Stalemate: no legal move.
    assert_eq!(
        (replies[3].nodes, replies[3].bestmove.as_str()),
        (0, "0000")
    );
}

struct Golden {
    name: String,
    position: String,
    elo_self: i64,
    elo_oppo: i64,
    policy: HashMap<String, f64>,
}

fn golden() -> Vec<Golden> {
    let json: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/golden.json")).unwrap();
    json["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Golden {
            name: c["name"].as_str().unwrap().to_string(),
            position: c["position"].as_str().unwrap().to_string(),
            elo_self: c["elo_self"].as_i64().unwrap(),
            elo_oppo: c["elo_oppo"].as_i64().unwrap(),
            policy: c["policy"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(m, p)| (m.clone(), p.as_f64().unwrap()))
                .collect(),
        })
        .collect()
}

/// Checks `got` against the golden policy; returns the largest absolute and
/// log gaps (the log one: `None` when `check_log` is off).
fn compare(case: &Golden, got: &HashMap<String, f64>, check_log: bool) -> (f64, f64) {
    let mut moves: Vec<&String> = got.keys().collect();
    moves.sort();
    let mut want: Vec<&String> = case.policy.keys().collect();
    want.sort();
    assert_eq!(moves, want, "{}: legal moves", case.name);
    let (mut abs, mut log) = (0.0f64, 0.0f64);
    for (mv, p) in &case.policy {
        let q = got[mv];
        abs = abs.max((q - p).abs());
        assert!(
            (q - p).abs() <= ABS_TOLERANCE,
            "{}: {mv} {q} vs {p}",
            case.name
        );
        if check_log && *p > LOG_FLOOR {
            let gap = (q.ln() - p.ln()).abs();
            log = log.max(gap);
            assert!(gap <= LOG_TOLERANCE, "{}: {mv} {q} vs {p}", case.name);
        }
    }
    (abs, log)
}

#[test]
fn golden_policies_match_official_pytorch() {
    let Some(onnx) = model() else { return };
    let cases = golden();

    // Full precision, through the library.
    let model = Model::load(&onnx).unwrap();
    let (mut abs, mut log) = (0.0f64, 0.0f64);
    for case in &cases {
        let history = parse_position(case.position.strip_prefix("position ").unwrap()).unwrap();
        let board = history.last().unwrap();
        let logits = model
            .move_logits(
                encoding::encode(&history),
                case.elo_self as f32,
                case.elo_oppo as f32,
            )
            .unwrap();
        let got: HashMap<String, f64> = encoding::policy(board, &logits)
            .into_iter()
            .map(|(mv, p)| (display_uci_move(board, mv).to_string(), p))
            .collect();
        let (a, l) = compare(case, &got, true);
        (abs, log) = (abs.max(a), log.max(l));
    }

    // The same through the executable, whose output has 4 decimals of percent.
    let mut input = format!("uci\nsetoption name WeightsFile value {onnx}\nisready\n");
    for case in &cases {
        input += &format!(
            "setoption name EloSelf value {}\nsetoption name EloOppo value {}\n{}\ngo nodes 1\n",
            case.elo_self, case.elo_oppo, case.position
        );
    }
    let lines = dialogue(input + "quit\n");
    assert_eq!(lines[6], "readyok");
    let replies = replies(&lines[7..]);
    assert_eq!(replies.len(), cases.len());
    let mut printed = 0.0f64;
    for (case, reply) in cases.iter().zip(&replies) {
        let got: HashMap<String, f64> = reply.policy.iter().cloned().collect();
        printed = printed.max(compare(case, &got, false).0);
        let best = case.policy.values().cloned().fold(0.0, f64::max);
        assert!(
            case.policy[&reply.bestmove] >= best - ABS_TOLERANCE,
            "{}: bestmove",
            case.name
        );
    }
    eprintln!(
        "{} golden positions: largest gap {abs:.2e}, in log {log:.2e}, printed {printed:.2e}",
        cases.len()
    );
}
