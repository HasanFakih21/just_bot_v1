use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::thread;

use crate::board::Board;
use crate::board::movegen::MoveGenKind;
use crate::search::data::{Report, SharedData};
use crate::search::time::{Nodes, TimeManager, TimeSettings};
use crate::threads::SearchThreads;
use crate::tools::bench::bench;
#[cfg(feature = "datagen")]
use crate::tools::datagen::generate_random_openings;
#[cfg(feature = "tuning")]
use crate::tools::parameters::{list_params, print_params_ob, set_param};
use crate::types::*;

#[derive(Default)]
pub struct UCISettings {
    pub soft_nodes: bool,
    pub frc: bool,
    pub report: Report,
}

pub fn input_loop(cli_args: String) {
    let shared = Arc::new(SharedData::default());
    let mut pool = SearchThreads::new(shared.clone(), 1);
    let mut board = Board::from_fen(STARTING_FEN).unwrap();
    let mut uci_settings = UCISettings::default();

    let rx = listen(shared.clone());

    let mut input = if !cli_args.is_empty() { cli_args } else { String::new() };

    loop {
        if input.is_empty() {
            if let Ok(s) = rx.recv() {
                input = s;
            } else {
                shared.status.stop();
                break;
            }
        }

        let (command, args) = input.split_once(" ").unwrap_or((&input, ""));

        match command.trim() {
            "position" => position(args, &mut board, uci_settings.frc),
            "uci" => uci(),
            "isready" => println!("readyok"),
            "setoption" => set_option(args, &mut uci_settings, shared.clone(), &mut pool),
            "ucinewgame" => {
                shared.history.clear();
                shared.tt.clear();
                let thread_count = pool.count();
                drop(pool);
                pool = SearchThreads::new(shared.clone(), thread_count);
            }
            "go" => {
                if let Some(m) = go(args, &mut pool, &mut board, &uci_settings) {
                    println!("bestmove {}", m.to_uci(&board));
                } else {
                    println!("bestmove 0000");
                }
            }
            "quit" => break,
            "perft" => {
                if let Ok(depth) = args.trim().parse::<usize>() {
                    crate::tools::perft::perft(depth, &mut board);
                } else {
                    eprintln!("Invalid depth: {:?}", args);
                }
            }
            "d" => println!("{}", board),
            "bench" => {
                let (total_node_count, nps) = bench();
                println!("{} nodes {} nps", total_node_count, nps);
                break;
            }
            #[cfg(feature = "datagen")]
            "genfens" => {
                genfens(args);
            }
            #[cfg(feature = "tuning")]
            "params" => print_params_ob(),
            _ => (),
        }

        if input.contains("quit") {
            break;
        }

        input.clear();
    }
}

pub fn listen(shared: Arc<SharedData>) -> Receiver<String> {
    let (tx, rx) = channel::<String>();
    let mut input_buffer = String::new();

    thread::spawn(move || {
        loop {
            if std::io::stdin().read_line(&mut input_buffer).unwrap() == 0 {
                shared.status.stop();
                break;
            };

            match input_buffer.trim() {
                "quit" => {
                    shared.status.stop();
                    break;
                }
                "stop" => {
                    shared.status.stop();
                }
                _ => (),
            }

            let _ = tx.send(input_buffer.clone());
            input_buffer.clear();
        }
    });

    rx
}

pub fn position(args: &str, board: &mut Board, frc: bool) {
    if args.trim().is_empty() {
        eprintln!("Need to provide a valid argument!");
        return;
    }

    let (command, args) = args.split_once(" ").unwrap_or((args, ""));
    let (args, moves) = args.split_once("moves").unwrap_or((args, ""));

    match command.trim() {
        "startpos" => {
            *board = Board::from_fen(STARTING_FEN).unwrap();
        }
        "fen" => {
            if args.trim().is_empty() {
                eprintln!("Please provide a fen string");
                return;
            }
            if let Ok(b) = Board::from_fen(args.trim_ascii_end()) {
                *board = b;
                board.frc = frc;
            } else {
                eprintln!("Invalid FEN: {:?}", args.trim_end());
            }
        }
        _ => eprintln!("Not a valid position argument!"),
    }

    if !moves.trim().is_empty() {
        for m_str in moves.split_ascii_whitespace() {
            let all_moves = board.generate_moves(MoveGenKind::All);
            if let Some(entry) = all_moves.iter().find(|entry| entry.mv.to_uci(board) == m_str) {
                board.make_move(entry.mv);
            } else {
                eprintln!("Illegal Move!");
                return;
            }
        }
    }
}

pub fn set_option(args: &str, uci_settings: &mut UCISettings, shared: Arc<SharedData>, pool: &mut SearchThreads) {
    let args = args.to_ascii_lowercase();
    let args: Vec<&str> = args.split_ascii_whitespace().collect();
    match args.as_slice() {
        ["name", "minimal", "value", v] => {
            let v = v.parse().unwrap_or(false);
            if v {
                uci_settings.report = Report::Minimal
            } else {
                uci_settings.report = Report::Full
            }
            println!("info string Set Minimal to {v}");
        }
        ["name", "hash", "value", amount] => {
            let amount = amount.parse::<usize>().unwrap_or(16);
            shared.tt.resize(amount);
            println!("info string Resized TT to {amount} mb");
        }
        ["name", "threads", "value", amount] => {
            let amount = amount.parse::<usize>().unwrap_or(1);
            *pool = SearchThreads::new(shared, amount);
            println!("info string Set Threads to {amount}");
        }
        ["name", "clear", "hash"] => {
            shared.tt.clear();
            println!("info string TT cleared");
        }
        ["name", "uci_chess960", "value", v] => {
            let v = v.parse().unwrap_or(false);
            uci_settings.frc = v;
            println!("info string Set UCI_Chess960 to {v}");
        }
        ["name", "softnodes", "value", v] => {
            let v = v.parse().unwrap_or(false);
            uci_settings.soft_nodes = v;
            println!("info string Set SoftNodes to {v}");
        }
        #[cfg(feature = "tuning")]
        ["name", name, "value", amount] => {
            match amount.parse::<i32>() {
                Ok(amount) => set_param(name, amount),
                Err(_) => {
                    println!("info error: invalid value '{}'", amount);
                }
            };
        }
        _ => eprintln!("Unkown option"),
    }
}

pub fn go(args: &str, pool: &mut SearchThreads, board: &mut Board, uci_settings: &UCISettings) -> Option<Move> {
    let args = args.to_ascii_lowercase();
    let args: Vec<&str> = args.split_ascii_whitespace().collect();
    let settings = parse_go(board.state.side_to_move, args.as_slice(), uci_settings.soft_nodes);
    let time = TimeManager::new(settings, board.state.full_move);

    pool.start(board, time.clone(), uci_settings.report)
}

fn parse_go(stm: Side, args: &[&str], soft_node: bool) -> TimeSettings {
    let mut settings = TimeSettings::default();
    for chunk in args.chunks(2) {
        if let [command, value] = *chunk {
            let Ok(value) = value.parse::<u64>() else {
                continue;
            };

            match command {
                "depth" if value > 0 => settings.depth = Some(value as i32),
                "movetime" if value > 0 => settings.movetime = Some(value),
                "movestogo" if value > 0 => settings.movestogo = Some(value),
                "mate" if value > 0 => settings.mate = Some(value),
                "nodes" if value > 0 => {
                    settings.nodes = if soft_node { Some(Nodes::Soft(value)) } else { Some(Nodes::Hard(value)) }
                }

                "wtime" if stm == Side::White => settings.time = Some(value),
                "winc" if stm == Side::White => settings.inc = value,
                "btime" if stm == Side::Black => settings.time = Some(value),
                "binc" if stm == Side::Black => settings.inc = value,

                _ => continue,
            }
        }
    }

    settings
}

pub fn uci() {
    println!("id name JustBot {}", env!("CARGO_PKG_VERSION"));
    println!("id author Hasan Fakih");
    println!("option name Threads type spin default 1 min 1 max 512");
    println!("option name Hash type spin default 16 min 1 max 1048576");
    println!("option name Clear Hash type button");
    println!("option name UCI_Chess960 type check default false");
    println!("option name Minimal type check default false");
    println!("option name SoftNodes type check default false");
    #[cfg(feature = "tuning")]
    list_params();
    println!("uciok");
}

#[cfg(feature = "datagen")]
pub fn genfens(args: &str) {
    let args = args.to_ascii_lowercase();
    let args: Vec<&str> = args.split_ascii_whitespace().collect();
    let mut amount = 0;
    let mut seed = 0;

    match args.as_slice() {
        [n, "seed", s, ..] => {
            amount = n.parse::<usize>().unwrap_or(0);
            seed = s.parse::<u64>().unwrap_or(0);
        }
        [n, ..] => {
            amount = n.parse::<usize>().unwrap_or(0);
        }
        _ => (),
    }

    generate_random_openings(amount, 8, seed);
}

#[cfg(test)]
pub mod tests {

    use super::*;
    use crate::types::constants::STARTING_FEN;

    #[test]
    fn test_parse_move() {
        let board = Board::from_fen(STARTING_FEN).unwrap();
        if let Ok(m) = board.parse_move("e2e4") {
            println!("bestmove {}", m.to_uci(&board));
        }
    }

    #[test]
    fn test_set_option() {
        let shared = Arc::new(SharedData::default());
        let mut pool = SearchThreads::new(shared.clone(), 1);

        set_option("name Hash value 32", &mut UCISettings::default(), shared, &mut pool);
    }
}
