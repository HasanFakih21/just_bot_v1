use crate::{
    board::{movegen::MoveGenKind, see},
    search::data::SearchData,
    types::{Move, MoveEntry, MoveList, OptionPiece, stackvec::StackVec},
};

#[derive(Debug, PartialEq)]
pub enum Stage {
    HashMove,
    FirstNoisy,
    GoodNoisy,
    Quiet,
    BadNoisy,
}

#[derive(Debug)]
pub struct MovePicker {
    moves: MoveList,
    tt_move: Option<Move>,
    status: Stage,
    bad_noisy: StackVec<Move, 256>,
    bad_index: usize,
    noisy_count: usize,
}

impl MovePicker {
    pub fn new(tt_move: Option<Move>) -> MovePicker {
        Self {
            moves: MoveList::new(),
            tt_move,
            status: if tt_move.is_some() { Stage::HashMove } else { Stage::FirstNoisy },
            bad_noisy: StackVec::new(),
            bad_index: 0,
            noisy_count: 0,
        }
    }

    pub fn next(&mut self, data: &SearchData, skip_quiets: bool, ply: isize) -> Option<Move> {
        let board = &data.board;
        if self.status == Stage::HashMove {
            self.status = Stage::FirstNoisy;
            let tt_move = self.tt_move.unwrap();
            if !skip_quiets || !tt_move.kind().is_quiet() {
                return Some(tt_move);
            }
        }

        if self.status == Stage::FirstNoisy {
            board.append_moves(MoveGenKind::Noisy, &mut self.moves);
            self.remove_tt_move();
            self.score_noisy_moves(data);
            self.status = Stage::GoodNoisy;
        }

        if self.status == Stage::GoodNoisy {
            while !self.moves.is_empty() {
                let best_entry = self.best_entry();
                let threshold = -best_entry.score / 4 + 64;
                if !data.board.see(best_entry.mv, threshold) {
                    self.bad_noisy.push(best_entry.mv);
                    continue;
                }

                self.noisy_count += 1;
                return Some(best_entry.mv);
            }

            if !skip_quiets {
                self.status = Stage::Quiet;
                board.append_moves(MoveGenKind::Quiet, &mut self.moves);
                self.remove_tt_move();
                self.score_quiet_moves(data, ply);
            } else {
                self.status = Stage::BadNoisy;
            }
        }

        if self.status == Stage::Quiet && !skip_quiets {
            if !self.moves.is_empty() {
                return Some(self.best_entry().mv);
            }

            self.status = Stage::BadNoisy;
        }

        // Bad Noisy
        if self.bad_index < self.bad_noisy.len() {
            let m = self.bad_noisy.get(self.bad_index);
            self.bad_index += 1;
            return Some(m);
        }

        None
    }

    fn score_noisy_moves(&mut self, data: &SearchData) {
        let threats = data.board.threats();
        for entry in self.moves.iter_mut() {
            let mv = entry.mv;
            let mut score = 0;

            // Bonus for promotions
            if mv.kind().is_queen_promotion() {
                score += 4885;
            }

            let piece = data.board.piece_at_square(mv.from());
            let to = mv.to();
            let captured = data.board.piece_at_square(mv.capture_square()).map(|e| e.kind());
            if let OptionPiece::Some(p) = captured {
                score += see::value(p)
            }

            score += data.noisy_history.get(piece, to, captured, threats) / 8;
            entry.score = score;
        }
    }

    fn score_quiet_moves(&mut self, data: &SearchData, ply: isize) {
        let side = data.board.state.side_to_move;
        let threats = data.board.threats();

        for entry in self.moves.iter_mut() {
            let mv = entry.mv;
            let piece = data.board.piece_at_square(mv.from());
            let to = mv.to();

            let conthistory_score = 1006 * data.pawn_history.get(data.board.state.keys.pawn, piece, to) / 1024
                + 1602 * data.conthistory(mv, ply, 1) / 1024
                + 1059 * data.conthistory(mv, ply, 2) / 1024
                + 1066 * data.conthistory(mv, ply, 4) / 1024;

            entry.score = data.quiet_history.get(threats, side, mv)
                + conthistory_score
                + (9779 * data.board.is_direct_check(mv) as i32);
        }
    }

    fn best_entry(&mut self) -> MoveEntry {
        let mut best_index = 0;
        let mut best_score = i32::MIN;

        for (index, entry) in self.moves.iter().enumerate() {
            if entry.score >= best_score {
                best_score = entry.score;
                best_index = index;
            }
        }

        self.moves.remove(best_index)
    }

    fn remove_tt_move(&mut self) {
        if let Some(tt_mv) = self.tt_move
            && let Some(index) = self.moves.iter().position(|e| tt_mv == e.mv)
        {
            self.moves.remove(index);
        }
    }
}
