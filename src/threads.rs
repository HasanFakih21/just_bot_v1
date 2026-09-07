use std::{
    collections::HashMap,
    sync::{
        Arc,
        mpsc::{Receiver, Sender},
    },
    thread::JoinHandle,
};

use crate::{
    board::{Board, movegen::MoveGenKind},
    search::{
        data::{Report, RootMove, SearchData, SharedData},
        search_runner,
        time::TimeManager,
    },
    types::{Move, Score, is_decisive, is_loss, is_win},
};

pub struct SearchThreads {
    workers: Vec<Worker>,
    shared: Arc<SharedData>,
}

#[derive(Debug, Clone)]
struct SearchParams {
    pub board: Board,
    pub root_moves: Vec<RootMove>,
    pub time: TimeManager,
    pub report: Report,
}

impl SearchThreads {
    pub fn new(shared: Arc<SharedData>, count: usize) -> Self {
        let workers = (0..count).map(|id| create_worker(Arc::clone(&shared), id)).collect();

        SearchThreads { workers, shared }
    }

    pub fn start(&mut self, board: &Board, time: TimeManager, report: Report) -> Option<Move> {
        debug_assert!(!self.workers.is_empty());
        self.shared.tt.increase_age();
        self.shared.reset_all_nodes();
        self.shared.status.run();

        let root_moves: Vec<RootMove> = board
            .generate_moves(MoveGenKind::All)
            .iter()
            .map(|e| RootMove {
                m: e.mv,
                ..Default::default()
            })
            .collect();

        let params = SearchParams {
            board: board.clone(),
            root_moves,
            time,
            report,
        };

        for w in self.workers.iter_mut() {
            w.comm
                .send(Command::Search(Box::new(params.clone())))
                .expect("Worker not found");
        }

        let mut threads = Vec::new();
        for w in self.workers.iter() {
            let Response::Search(search_result) = w.result.recv().expect("Worker not found") else {
                panic!("Should have recieved a search response here");
            };

            if !search_result.best_move.m.is_null()
                && search_result.searched_depth > 0
                && search_result.best_move.score != Score::NONE
            {
                threads.push(search_result);
            }
        }

        if threads.is_empty() {
            let move_list = board.generate_moves(MoveGenKind::All);
            if move_list.is_empty() { return None } else { return Some(move_list.get(0).mv) }
        }

        let lowest_score = threads.iter().map(|result| result.best_move.score).min().unwrap();
        let mut votes: HashMap<&Move, i64> = HashMap::new();

        let thread_weight = |result: &SearchResult| {
            (result.best_move.score as i64 - lowest_score as i64 + 10) * result.searched_depth as i64
        };

        for result in threads.iter() {
            *votes.entry(&result.best_move.m).or_default() += thread_weight(result);
        }

        let mut best_index = 0;

        for current_index in 0..threads.len() {
            let best = &threads[best_index].best_move;
            let current = &threads[current_index].best_move;

            if is_win(best.score) {
                if current.score > best.score {
                    best_index = current_index;
                }
                continue;
            }

            if is_loss(best.score) {
                if is_loss(current.score) && current.score < best.score {
                    best_index = current_index;
                }
                continue;
            }

            if is_decisive(current.score) {
                best_index = current_index;
                continue;
            }

            if votes[&current.m] > votes[&best.m] {
                best_index = current_index;
                continue;
            }

            if votes[&current.m] == votes[&best.m]
                && thread_weight(&threads[current_index]) > thread_weight(&threads[best_index])
            {
                best_index = current_index;
            }
        }

        if report != Report::None && threads[best_index].id != 0 {
            self.workers[threads[best_index].id]
                .comm
                .send(Command::PrintUCI)
                .expect("Worker {id} was supposed to print uci but couldn't");

            let Response::PrintUCI = self.workers[threads[best_index].id]
                .result
                .recv()
                .expect("Printing worker didn't respond!")
            else {
                unreachable!();
            };
        }

        Some(threads[best_index].best_move.m)
    }

    pub fn count(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for SearchThreads {
    fn drop(&mut self) {
        self.shared.status.stop();
        for w in self.workers.iter() {
            let _ = w.comm.send(Command::Quit);
        }

        for w in self.workers.drain(..) {
            let _ = w.handle.join();
        }
    }
}

fn create_worker(shared: Arc<SharedData>, id: usize) -> Worker {
    let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
    let (result_tx, result_rx) = std::sync::mpsc::channel();

    let handle = std::thread::spawn(move || {
        let mut data = SearchData::new(Arc::clone(&shared), id);

        while let Ok(command) = cmd_rx.recv() {
            match command {
                Command::Search(params) => {
                    data.board = params.board;
                    data.root_moves = params.root_moves;
                    data.time = params.time;
                    if id == 0 {
                        data.report = params.report;
                    } else {
                        data.report = Report::None;
                    }

                    search_runner(&mut data);
                    if result_tx
                        .send(Response::Search(SearchResult {
                            id: data.id,
                            best_move: data.best_move.clone().unwrap_or_default(),
                            searched_depth: data.completed_depth,
                        }))
                        .is_err()
                    {
                        break;
                    };
                }
                Command::Quit => break,
                Command::PrintUCI => {
                    data.print_uci_info();
                    if result_tx.send(Response::PrintUCI).is_err() {
                        break;
                    }
                }
            }
        }
    });

    Worker {
        handle,
        comm: cmd_tx,
        result: result_rx,
    }
}

struct SearchResult {
    id: usize,
    best_move: RootMove,
    searched_depth: i32,
}

enum Response {
    Search(SearchResult),
    PrintUCI,
}

enum Command {
    Search(Box<SearchParams>),
    PrintUCI,
    Quit,
}

struct Worker {
    handle: JoinHandle<()>,
    comm: Sender<Command>,
    result: Receiver<Response>,
}
