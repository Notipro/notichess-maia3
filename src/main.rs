// Copyright (C) 2026 NotiChess contributors
// SPDX-License-Identifier: AGPL-3.0-only
//
// The Maia-3 model and the code this encoding is derived from
// (github.com/CSSLab/maia3) belong to the CSSLab, University of Toronto,
// and are licensed under the AGPL-3.0.

//! notichess-maia3: the Maia-3 5M model behind a subset of UCI.
//!
//! The output format is a contract with NotiChess, which reads it the way it
//! reads Lc0 with `VerboseMoveStats`: after `go`, one
//! `info string <move> (P: <percent>%)` line per legal move, then
//! `info string node (<legal move count>)`, then `bestmove`. Logs go to stderr;
//! nothing else is ever written to stdout.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::Instant;

use cozy_chess::util::display_uci_move;
use cozy_chess::Board;

use notichess_maia3::model::Model;
use notichess_maia3::{encoding, parse_position};

const ELO_MIN: i64 = 600;
const ELO_MAX: i64 = 2600;
const ELO_DEFAULT: i64 = 1500;

struct Engine {
    /// `WeightsFile` as last set; empty: none.
    weights: String,
    /// The model and the path it was loaded from.
    model: Option<(String, Model)>,
    /// Elo of the side to move (Maia when she plays) and of its opponent.
    elo_self: i64,
    elo_oppo: i64,
    /// Every position since the game start, current last; `Err`: the last
    /// `position` command was refused.
    history: Result<Vec<Board>, String>,
}

impl Engine {
    fn new() -> Self {
        Self {
            weights: String::new(),
            model: None,
            elo_self: ELO_DEFAULT,
            elo_oppo: ELO_DEFAULT,
            history: Ok(vec![Board::default()]),
        }
    }

    fn set_option(&mut self, args: &str) {
        let Some(rest) = args.strip_prefix("name ") else {
            return;
        };
        let (name, value) = rest
            .split_once(" value ")
            .unwrap_or_else(|| (rest.strip_suffix(" value").unwrap_or(rest), ""));
        let value = value.trim();
        match name.trim().to_ascii_lowercase().as_str() {
            "weightsfile" if value == "<empty>" => self.weights.clear(),
            "weightsfile" => self.weights = value.to_string(),
            "eloself" => set_elo(&mut self.elo_self, value),
            "elooppo" => set_elo(&mut self.elo_oppo, value),
            // VerboseMoveStats, PolicyTemperature, Threads, ...: accepted, ignored.
            _ => eprintln!("ignored option: {name}"),
        }
    }

    /// Loads `WeightsFile` if it changed; nothing to do while it is empty.
    fn ensure_model(&mut self) -> Result<(), String> {
        if self.weights.is_empty() {
            self.model = None;
            return Ok(());
        }
        if matches!(&self.model, Some((path, _)) if *path == self.weights) {
            return Ok(());
        }
        self.model = None;
        let name = file_name(&self.weights);
        let started = Instant::now();
        let model = Model::load(&self.weights).map_err(|e| {
            let reason = hide_path(&format!("{e:#}"), &self.weights, &name);
            format!("cannot load {name}: {reason}")
        })?;
        eprintln!("loaded {name} in {:?}", started.elapsed());
        self.model = Some((self.weights.clone(), model));
        Ok(())
    }

    /// The `go` reply, or the reason it cannot be computed.
    fn go(&mut self) -> Result<String, String> {
        self.ensure_model()?;
        let Some((_, model)) = &self.model else {
            return Err("WeightsFile is not set".into());
        };
        let history = self.history.as_ref().map_err(Clone::clone)?;
        let board = history
            .last()
            .expect("history starts with the start position");
        let logits = model
            .move_logits(
                encoding::encode(history),
                self.elo_self as f32,
                self.elo_oppo as f32,
            )
            .map_err(|e| format!("inference failed: {e:#}"))?;
        let policy = encoding::policy(board, &logits);

        let mut reply = String::new();
        for (mv, p) in &policy {
            let uci = display_uci_move(board, *mv);
            reply += &format!("info string {uci} (P: {:.4}%)\n", p * 100.0);
        }
        reply += &format!("info string node ({})\n", policy.len());
        match policy.first() {
            Some((mv, _)) => reply += &format!("bestmove {}\n", display_uci_move(board, *mv)),
            None => reply += "bestmove 0000\n",
        }
        Ok(reply)
    }
}

fn set_elo(slot: &mut i64, value: &str) {
    match value.parse::<i64>() {
        Ok(elo) => *slot = elo.clamp(ELO_MIN, ELO_MAX),
        Err(_) => eprintln!("ignored Elo value: {value}"),
    }
}

fn file_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// `text` with `path` and its folder replaced by the file name: the reply
/// ends up in the application's logs, and a full path holds the user's
/// name. Errors quote paths raw or escaped (`{:?}`, `\\` on Windows).
fn hide_path(text: &str, path: &str, name: &str) -> String {
    // A root folder (`/`, `C:\`) names nobody and would match everywhere.
    let folder = Path::new(path)
        .parent()
        .filter(|p| p.file_name().is_some())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut text = text.to_string();
    for (secret, by) in [(path, name), (folder.as_str(), "")] {
        if secret.is_empty() {
            continue;
        }
        let escaped = format!("{secret:?}");
        text = text
            .replace(&escaped[1..escaped.len() - 1], by)
            .replace(secret, by);
    }
    text
}

/// One reply, flushed: the reader waits for it.
fn reply(text: &str) {
    let mut out = io::stdout().lock();
    // A closed stdout means the parent is gone: nothing left to tell it.
    let _ = out.write_all(text.as_bytes()).and_then(|_| out.flush());
}

fn error(reason: &str) {
    reply(&format!("error {}\n", reason.replace(['\r', '\n'], " ")));
}

fn main() {
    let mut engine = Engine::new();
    let mut input = io::stdin().lock();
    let mut raw = Vec::new();
    loop {
        raw.clear();
        // End of input or a broken pipe: the parent is gone.
        match input.read_until(b'\n', &mut raw) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        // A line that is not UTF-8 is read lossily, not fatal.
        let line = String::from_utf8_lossy(&raw);
        let line = line.trim();
        let (command, args) = line.split_once(' ').unwrap_or((line, ""));
        let args = args.trim();
        match command {
            "uci" => reply(&format!(
                "id name notichess-maia3 {}\n\
                 id author NotiChess; Maia-3 model by the CSSLab, University of Toronto\n\
                 option name WeightsFile type string default <empty>\n\
                 option name EloSelf type spin default {ELO_DEFAULT} min {ELO_MIN} max {ELO_MAX}\n\
                 option name EloOppo type spin default {ELO_DEFAULT} min {ELO_MIN} max {ELO_MAX}\n\
                 uciok\n",
                env!("CARGO_PKG_VERSION")
            )),
            "setoption" => engine.set_option(args),
            "isready" => match engine.ensure_model() {
                Ok(()) => reply("readyok\n"),
                Err(e) => error(&e),
            },
            "ucinewgame" => engine.history = Ok(vec![Board::default()]),
            "position" => {
                engine.history = parse_position(args);
                if let Err(e) = &engine.history {
                    eprintln!("{e}");
                }
            }
            "go" => match engine.go() {
                Ok(text) => reply(&text),
                Err(e) => error(&e),
            },
            "quit" => break,
            "" | "stop" => {}
            _ => eprintln!("ignored command: {line}"),
        }
    }
}
