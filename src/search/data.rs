use std::array;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::board::Board;
use crate::nnue::Network;
use crate::search::time::{TimeManager, TimeSettings};
use crate::types::pv::PVTable;
use crate::types::stack::Stack;
use crate::types::{
    ContinuationCorrectionHistory, ContinuationHistory, CorrectionHistory, Move, NoisyHistory, PawnHistory,
    STARTING_FEN, Score, Side, is_decisive,
};
use crate::types::{QuietHistory, TranspositionTable};

#[derive(Debug)]
pub struct Status(AtomicBool);

impl Status {
    pub const RUNNING: bool = true;
    pub const STOPPED: bool = false;

    pub fn stop(&self) {
        self.0.store(Self::STOPPED, Ordering::Relaxed);
    }

    pub fn run(&self) {
        self.0.store(Self::RUNNING, Ordering::Relaxed);
    }

    pub fn get(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub struct SharedData {
    pub tt: TranspositionTable,
    pub history: SharedCorrectionHistories,
    pub status: Status,
    nodes: Box<[AlignedAtomicU64; 512]>,
}

#[derive(Debug)]
#[repr(align(64))]
struct AlignedAtomicU64(AtomicU64);

impl SharedData {
    pub fn increment_nodes(&self, id: usize) {
        self.nodes[id]
            .0
            .store(self.nodes[id].0.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
    }

    pub fn node_count(&self, id: usize) -> u64 {
        self.nodes[id].0.load(Ordering::Relaxed)
    }

    pub fn total_nodes_searched(&self) -> u64 {
        self.nodes.iter().map(|n| n.0.load(Ordering::Relaxed)).sum()
    }

    pub fn reset_all_nodes(&self) {
        for t in self.nodes.iter() {
            t.0.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for SharedData {
    fn default() -> Self {
        Self {
            history: SharedCorrectionHistories::default(),
            tt: TranspositionTable::default(),
            status: Status(AtomicBool::new(Status::RUNNING)),
            nodes: Box::new(array::from_fn(|_| AlignedAtomicU64(AtomicU64::new(0)))),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum Report {
    #[default]
    Full,
    Minimal,
    None,
}

pub struct SearchData {
    pub id: usize,
    pub best_move: Option<RootMove>,
    pub shared: Arc<SharedData>,
    pub pv: PVTable,
    pub board: Board,
    pub time: TimeManager,
    pub report: Report,
    pub stack: Box<Stack>,
    pub root_moves: Vec<RootMove>,
    pub root_depth: i32,
    pub sel_depth: i32,
    pub prev_score: i32,
    pub nmp_min_ply: i32,
    pub completed_depth: i32,

    pub quiet_history: QuietHistory,
    pub noisy_history: NoisyHistory,
    pub pawn_history: PawnHistory,
    pub conthistory: ContinuationHistory,
    pub contcorrhistory: ContinuationCorrectionHistory,

    pub network: Network,
}

impl SearchData {
    pub fn new(shared: Arc<SharedData>, id: usize) -> Self {
        SearchData {
            id,
            best_move: None,
            shared,
            pv: PVTable::new(),
            board: Board::from_fen(STARTING_FEN).unwrap(),
            time: TimeManager::new(TimeSettings::default(), 0),
            report: Report::None,
            stack: Stack::new(),
            root_moves: Vec::new(),
            root_depth: 0,
            sel_depth: 0,
            prev_score: 0,
            nmp_min_ply: 0,
            completed_depth: 0,

            quiet_history: QuietHistory::new(),
            noisy_history: NoisyHistory::new(),
            pawn_history: PawnHistory::new(),
            conthistory: ContinuationHistory::new(),
            contcorrhistory: ContinuationCorrectionHistory::new(),

            network: Network::new(),
        }
    }

    pub fn corrhistory(&self) -> &SharedCorrectionHistories {
        &self.shared.history
    }

    pub fn nodes(&self) -> u64 {
        self.shared.node_count(self.id)
    }

    pub fn reset_nodes(&self) {
        self.shared.nodes[self.id].0.store(0, Ordering::Relaxed);
    }

    pub fn nodes_per_second(&self) -> usize {
        (self.shared.total_nodes_searched() as f32 / self.time.elapsed().as_secs_f32()) as usize
    }

    pub fn update_conthistories(&mut self, m: Move, ply: isize, bonus: i32) {
        unsafe {
            for i in [1, 2, 4] {
                self.conthistory.update(
                    self.stack[ply - i].conthistory,
                    self.board.piece_at_square(m.from()),
                    m.to(),
                    bonus,
                );
            }
        }
    }

    pub fn update_correction_histories(&mut self, diff: i32, depth: i32, ply: isize) {
        let stm = self.board.state.side_to_move;
        let bonus = (157 * depth * diff / 128).clamp(-4605, 2548);
        self.corrhistory().pawn.update(stm, self.board.state.keys.pawn, bonus);
        self.corrhistory().non_pawn[Side::White].update(stm, self.board.state.keys.non_pawn[Side::White], bonus);
        self.corrhistory().non_pawn[Side::Black].update(stm, self.board.state.keys.non_pawn[Side::Black], bonus);

        unsafe {
            if !self.stack[ply - 1].m.is_null() && !self.stack[ply - 2].m.is_null() {
                self.contcorrhistory.update(
                    self.stack[ply - 2].contcorrhistory,
                    self.stack[ply - 1].piece,
                    self.stack[ply - 1].m.to(),
                    bonus,
                );
            }

            if !self.stack[ply - 1].m.is_null() && !self.stack[ply - 4].m.is_null() {
                self.contcorrhistory.update(
                    self.stack[ply - 4].contcorrhistory,
                    self.stack[ply - 1].piece,
                    self.stack[ply - 1].m.to(),
                    bonus,
                );
            }
        }
    }

    pub fn correction(&self, ply: isize) -> i32 {
        let stm = self.board.state.side_to_move;
        (self.corrhistory().pawn.get(stm, self.board.state.keys.pawn)
            + self.corrhistory().non_pawn[Side::White].get(stm, self.board.state.keys.non_pawn[Side::White])
            + self.corrhistory().non_pawn[Side::Black].get(stm, self.board.state.keys.non_pawn[Side::Black])
            + unsafe {
                if !self.stack[ply - 1].m.is_null() && !self.stack[ply - 2].m.is_null() {
                    self.contcorrhistory.get(
                        self.stack[ply - 2].contcorrhistory,
                        self.stack[ply - 1].piece,
                        self.stack[ply - 1].m.to(),
                    )
                } else {
                    0
                }
            }
            + unsafe {
                if !self.stack[ply - 1].m.is_null() && !self.stack[ply - 4].m.is_null() {
                    self.contcorrhistory.get(
                        self.stack[ply - 4].contcorrhistory,
                        self.stack[ply - 1].piece,
                        self.stack[ply - 1].m.to(),
                    )
                } else {
                    0
                }
            })
            / 64
    }

    pub fn conthistory(&self, m: Move, ply: isize, index: isize) -> i32 {
        unsafe {
            self.conthistory.get(
                self.stack[ply - index].conthistory,
                self.board.piece_at_square(m.from()),
                m.to(),
            )
        }
    }

    pub fn print_uci_info(&self) {
        let Some(root_move) = &self.best_move else {
            return;
        };

        let mut upperbound = root_move.upperbound;
        let mut lowerbound = root_move.lowerbound;

        let mut score = root_move.display_score;

        if root_move.score == -Score::INFINITY {
            score = root_move.previous_score;

            upperbound = false;
            lowerbound = false;
        }

        // Report mate score
        let mut score_print = if is_decisive(score) {
            let num_plies = Score::MATE - score.abs();
            let mate_in = score.signum() * ((num_plies + 1) / 2);
            format!("mate {}", mate_in)
        } else {
            format!("cp {}", score)
        };

        if upperbound {
            score_print.push_str(" upperbound");
        }

        if lowerbound {
            score_print.push_str(" lowerbound");
        }

        let pv_display = {
            let mut output = format!("{} ", root_move.m.to_uci(&self.board));
            for m in &root_move.pv.inner {
                output = format!("{output}{} ", m.to_uci(&self.board));
            }

            output
        };

        println!(
            "info depth {} seldepth {} time {} score {} nodes {} nps {} pv {} hashfull {}",
            root_move.searched_depth,
            root_move.sel_depth,
            self.time.elapsed().as_millis(),
            score_print,
            self.shared.total_nodes_searched(),
            self.nodes_per_second(),
            pv_display,
            self.shared.tt.hashfull(),
        );
    }

    pub fn make_move(&mut self, m: Move, ply: isize) {
        self.network.push(&self.board, m);

        let from = m.from();
        let to = m.to();
        let piece = self.board.piece_at_square(from);

        self.stack[ply].m = m;
        self.stack[ply].piece = piece;
        self.stack[ply].conthistory = self.conthistory.subtable(piece, to);
        self.stack[ply].contcorrhistory = self.contcorrhistory.subtable(piece, to);
        self.stack[ply].threats = self.board.threats();

        self.board.make_move(m);
        self.shared.tt.prefetch(self.board.hash());
        self.shared.increment_nodes(self.id);
    }

    pub fn unmake_move(&mut self) {
        self.board.unmake_move();
        self.network.pop();
    }
}

impl Default for SearchData {
    fn default() -> Self {
        Self::new(Arc::new(SharedData::default()), 0)
    }
}

#[derive(Debug, Default)]
pub struct SharedCorrectionHistories {
    pub pawn: CorrectionHistory,
    pub non_pawn: [CorrectionHistory; 2],
}

impl SharedCorrectionHistories {
    pub fn clear(&self) {
        self.pawn.clear();
        for history in self.non_pawn.iter() {
            history.clear();
        }
    }
}

#[derive(Debug, Clone)]
pub struct RootMove {
    pub m: Move,
    pub nodes: u64,
    pub pv: RootPV,
    pub score: i32,
    pub previous_score: i32,
    pub display_score: i32,
    pub upperbound: bool,
    pub lowerbound: bool,
    pub searched_depth: i32,
    pub sel_depth: i32,
}

impl Default for RootMove {
    fn default() -> Self {
        RootMove {
            m: Move::NONE,
            nodes: 0,
            pv: RootPV::default(),
            score: -Score::INFINITY,
            previous_score: -Score::INFINITY,
            display_score: -Score::INFINITY,
            upperbound: false,
            lowerbound: false,
            searched_depth: 0,
            sel_depth: 0,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RootPV {
    pub inner: Vec<Move>,
}

impl RootPV {
    pub fn commit(&mut self, line: &[Move]) {
        self.inner = Vec::from(line)
    }
}
