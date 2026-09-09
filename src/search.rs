use crate::search::{
    data::{Report, SearchData, Status},
    movepicker::MovePicker,
    time::Limit,
};
use crate::types::*;
use crate::types::{stack::Stack, stackvec::StackVec};

pub mod data;
pub mod movepicker;
pub mod time;

#[cfg(test)]
mod tests;

pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
}

pub struct PV;
pub struct Root;
pub struct NonPV;

impl NodeType for PV {
    const PV: bool = true;
    const ROOT: bool = false;
}

impl NodeType for NonPV {
    const PV: bool = false;
    const ROOT: bool = false;
}

impl NodeType for Root {
    const PV: bool = true;
    const ROOT: bool = true;
}

pub fn search_runner(data: &mut SearchData) {
    data.network.full_refresh(&data.board);
    data.pv.clear(0);
    data.time.start_clock();

    let mut delta = 24;
    let mut alpha = -Score::INFINITY;
    let mut beta = Score::INFINITY;

    let mut depth = 1;
    let mut best_score = 0;
    data.best_move = None;
    data.completed_depth = 0;
    data.root_depth = 0;
    data.sel_depth = 0;

    if data.root_moves.is_empty() {
        return;
    }

    // Time Management
    let mut move_stability = 0;

    // Iterative Deepening
    loop {
        data.stack = Stack::new();
        data.root_depth = depth;

        if data.id == 0
            && (match data.time.limit {
                Limit::Depth(limit) => depth > limit,
                _ => data.time.hard_limit(data),
            })
        {
            data.shared.status.stop();
            break;
        }

        let score = search::<Root>(data, depth, alpha, beta, 0, false);

        if data.shared.status.get() == Status::STOPPED {
            break;
        }

        // Aspiration Window
        if score <= alpha {
            // Failed Low
            alpha = (score - delta).max(-Score::INFINITY);
            beta = (alpha + delta).min(beta);
            delta += 24 * delta / 128;
            continue;
        } else if score >= beta {
            // Failed High
            alpha = (beta - delta).max(alpha);
            beta = (score + delta).min(Score::INFINITY);
            delta += 25 * delta / 128;
            continue;
        }

        data.completed_depth = depth;
        data.sel_depth = 0;
        depth += 1;

        data.root_moves.sort_by_key(|rm| std::cmp::Reverse(rm.score));
        data.root_moves.iter_mut().for_each(|rm| rm.previous_score = rm.score);
        if data.best_move.as_ref().is_some_and(|rm| rm.m == data.root_moves[0].m) {
            move_stability += 1;
        } else {
            move_stability = 0;
        }

        data.best_move = Some(data.root_moves[0].clone());
        best_score = data.root_moves[0].score;

        if data.report == Report::Full {
            data.print_uci_info();
        }

        if data.id == 0
            && let Limit::Mate(moves) = data.time.limit
            && Score::MATE - best_score.abs() <= moves as i32 * 2
        {
            data.shared.status.stop();
            break;
        }

        let multiplier = || {
            let ratio = data.root_moves[0].nodes as f32 / data.nodes() as f32;
            let node_tm = (2.977 - ratio * 2.495).max(0.553);

            let diff = (data.prev_score - best_score) as f32;
            let score_trend = (0.75 + 0.045 * diff).clamp(0.7, 1.5);
            let m_stability = (1.2000 - 0.0500 * move_stability as f32).max(0.8500);

            node_tm * score_trend * m_stability
        };

        if data.id == 0 && data.time.soft_limit(data, multiplier) {
            data.shared.status.stop();
            break;
        }

        delta = 25;
        alpha = (score - delta).max(-Score::INFINITY);
        beta = (score + delta).min(Score::INFINITY);
    }

    if matches!(data.report, Report::Minimal | Report::Full) {
        data.print_uci_info();
    }

    data.prev_score = best_score;
}

pub fn search<Node: NodeType>(
    data: &mut SearchData,
    depth: i32,
    mut alpha: i32,
    mut beta: i32,
    ply: isize,
    cutnode: bool,
) -> i32 {
    if Node::PV {
        data.sel_depth = data.sel_depth.max(ply as i32);
        if !Node::ROOT {
            data.pv.clear(ply);
        }
    }

    if data.shared.status.get() == Status::STOPPED {
        return Score::TIMEOUT;
    }

    // Horizon Node
    if depth <= 0 {
        return quiesce::<Node>(data, alpha, beta, ply);
    }

    let stm = data.board.state.side_to_move;
    let in_check = data.board.king_in_check();
    let excluded = !data.stack[ply].excluded.is_null();

    if !Node::ROOT {
        // Check for draws
        if data.board.is_draw() {
            return Score::DRAW;
        }

        // Prevent from going too deep
        if ply >= MAX_PLY as isize - 1 {
            if in_check {
                return Score::DRAW;
            } else {
                return data.network.evaluate(&data.board);
            }
        }

        // Mate Distance Pruning (MDP)
        alpha = alpha.max(-Score::MATE + ply as i32);
        beta = beta.min(Score::MATE - ply as i32 + 1);
        if alpha >= beta {
            return alpha;
        }
    }

    // Check for Time Outs
    if data.id == 0 && data.time.hard_limit(data) {
        data.shared.status.stop();
        return Score::TIMEOUT;
    }

    let mut depth = depth.min(MAX_PLY as i32 - 1);

    // Transposition Table Entries
    let tt_entry = data.shared.tt.entry(data.board.hash(), ply);
    let tt_move = tt_entry.as_ref().map(|e| e.best_move()).filter(|m| !m.is_null());
    let tt_bound = tt_entry.as_ref().map(|e| e.bound());
    let tt_score = tt_entry.as_ref().map(|e| e.score()).filter(|s| *s != Score::NONE);
    let tt_was_pv = tt_entry.as_ref().map(|e| e.is_pv()).unwrap_or(false);
    let tt_depth = tt_entry.as_ref().map(|e| e.depth());
    let tt_pv = Node::PV || tt_was_pv;

    // TT Cutoffs
    if !Node::PV
        && let Some(tt_score) = tt_score
        && tt_depth.is_some_and(|d| d >= depth)
        && (tt_score <= alpha || cutnode)
        && !excluded
        && tt_bound.is_some_and(|b| match b {
            Bound::Lower => tt_score >= beta,
            Bound::Upper => tt_score < alpha,
            Bound::Exact => true,
            Bound::None => false,
        })
    {
        if let Some(tt_move) = tt_move
            && tt_score >= beta
            && tt_move.kind().is_quiet()
        {
            let bonus = (206 * depth - 50).min(1201);
            data.quiet_history.update(data.board.threats(), stm, tt_move, bonus);
        }

        return tt_score;
    }

    // Evaluation
    let raw_eval;
    let static_eval;
    let correction = data.correction(ply);

    if in_check {
        raw_eval = Score::NONE;
        static_eval = Score::NONE;
    } else if excluded {
        raw_eval = Score::NONE;
        static_eval = data.stack[ply].eval
    } else if let Some(e) = &tt_entry
        && e.eval() != Score::NONE
    {
        raw_eval = e.eval();
        static_eval = raw_eval + correction;
    } else {
        raw_eval = data.network.evaluate(&data.board);
        static_eval = raw_eval + correction;
    };

    data.stack[ply].eval = static_eval;
    if !excluded && tt_entry.is_none() {
        data.shared.tt.add_entry(
            Move::NONE,
            Score::NONE,
            raw_eval,
            Bound::None,
            data.board.hash(),
            0,
            ply,
            tt_pv,
        );
    }

    let improvement = if in_check {
        0
    } else if data.stack[ply - 2].eval != Score::NONE {
        static_eval - data.stack[ply - 2].eval
    } else if data.stack[ply - 4].eval != Score::NONE {
        static_eval - data.stack[ply - 4].eval
    } else {
        0
    };

    let improving = improvement > 0;

    // Update Quiet History Based on the Change in Evaluation
    if !Node::ROOT
        && !in_check
        && !excluded
        && data.stack[ply - 1].m.is_null()
        && data.stack[ply - 1].m.kind().is_quiet()
        && data.stack[ply - 1].eval != Score::NONE
    {
        let bonus = (512 * (-data.stack[ply - 1].eval - static_eval) / 128).clamp(-1200, 1200);
        data.quiet_history.update(
            data.stack[ply - 1].threats,
            !stm,
            data.stack[ply - 1].m,
            (bonus.abs() - depth * 200).max(0) * bonus.signum(),
        );
    }

    if !Node::ROOT && !in_check && !excluded && data.stack[ply - 1].eval != Score::NONE {
        // Hindsight Extension
        if depth < MAX_PLY as i32
            && data.stack[ply - 1].reduction >= 3093
            && static_eval + data.stack[ply - 1].eval <= 0
        {
            depth += 1;
        }

        // Hindsight Reduction
        if depth >= 2 && data.stack[ply - 1].reduction >= 2078 && static_eval + data.stack[ply - 1].eval >= 211 {
            depth -= 1;
        }
    }

    // Razoring
    if !Node::PV
        && !in_check
        && tt_bound.is_none_or(|b| b != Bound::Lower)
        && static_eval < alpha - 242 - 254 * depth * depth
        && alpha < 2000
    {
        return quiesce::<Node>(data, alpha, beta, ply);
    }

    // Reverse Futillity Pruning (RFP)
    if !in_check
        && !Node::PV
        && !excluded
        && static_eval >= beta + 87 * depth + 6 * depth * depth - 73 * improving as i32
        && !is_decisive(beta)
        && !is_decisive(static_eval)
    {
        return ilerp::<1024>(static_eval, beta, 686);
    }

    // Null Move Pruning
    if cutnode
        && depth >= 3
        && !excluded
        && !in_check
        && ply as i32 >= data.nmp_min_ply
        && !data.board.only_king_and_pawns()
        && tt_bound.is_none_or(|b| b != Bound::Upper)
        && static_eval >= beta + (199 - 1214 * depth / 128 - 64 * improving as i32).max(0)
        && !data.stack[ply - 1].m.is_null()
    {
        let r = 6 + depth * 124 / 640;
        data.stack[ply].conthistory = data.stack.sentinel();
        data.stack[ply].contcorrhistory = data.stack.sentinel();
        data.stack[ply].m = Move::NONE;
        data.stack[ply].piece = OptionPiece::None;

        data.board.make_null_move();
        data.shared.tt.prefetch(data.board.hash());

        let score = -search::<NonPV>(data, depth - r, -beta, -beta + 1, ply + 1, false);
        data.board.unmake_move();

        if data.shared.status.get() == Status::STOPPED {
            return Score::TIMEOUT;
        }

        if score >= beta && !is_win(score) && !is_loss(score) {
            if depth <= 14 || data.nmp_min_ply > 0 {
                return score;
            }

            data.nmp_min_ply = ply as i32 + (depth - r) * 3 / 4;
            let verified_score = search::<NonPV>(data, depth - r, beta - 1, beta, ply, true);
            data.nmp_min_ply = 0;

            if data.shared.status.get() == Status::STOPPED {
                return Score::TIMEOUT;
            }

            if verified_score >= beta {
                return verified_score;
            }
        }
    }

    // Singular Extensions (SE)
    let mut extension = 0;
    if !Node::ROOT
        && !excluded
        && depth >= 5
        && tt_depth.is_some_and(|d| d >= depth - 3)
        && let Some(tt_move) = tt_move
        && let Some(tt_bound) = tt_bound
        && let Some(tt_score) = tt_score
        && !is_decisive(tt_score)
        && tt_bound != Bound::Upper
    {
        let singular_depth = (depth - 1) / 2;
        let singular_beta = tt_score - (depth + depth);

        data.stack[ply].excluded = tt_move;
        data.stack[ply].m = Move::NONE;
        // Search everything except the TT move with a null window at a reduced depth to find out if it's worth extending or not
        let singular_score = search::<NonPV>(data, singular_depth, singular_beta - 1, singular_beta, ply, cutnode);
        data.stack[ply].excluded = Move::NONE;

        if data.shared.status.get() == Status::STOPPED {
            return Score::TIMEOUT;
        }

        if singular_score < singular_beta {
            let double_margin = 10 + 150 * Node::PV as i32 + 50 * (Node::PV && !tt_was_pv) as i32;
            let triple_margin = 100 + 351 * Node::PV as i32 + 55 * (Node::PV && !tt_was_pv) as i32;
            extension = 1
                + (singular_score < singular_beta - double_margin) as i32
                + (singular_score < singular_beta - triple_margin) as i32;
        }
        // Negative Extensions
        else if tt_score >= beta || cutnode {
            extension -= 3;
        }
    }

    let mut move_count = 0;
    let mut best_score = -Score::INFINITY;
    let mut best_move: Option<Move> = None;
    // Fail-high means score is atleast this good so lower-bound/Fail-low means the score is an upper bound
    let mut bound = Bound::Upper;

    let mut move_picker = MovePicker::new(tt_move);
    let mut quiets_searched = StackVec::<Move, 32>::new();
    let mut noisies_searched = StackVec::<Move, 32>::new();
    let mut skip_quiets = false;

    while let Some(m) = move_picker.next(data, skip_quiets, ply) {
        if m == data.stack[ply].excluded {
            continue;
        }

        move_count += 1;

        let is_direct_check = data.board.is_direct_check(m);
        let is_quiet = m.kind().is_quiet();
        let history = if is_quiet {
            data.quiet_history.get(data.board.threats(), stm, m)
                + data.conthistory(m, ply, 1)
                + data.conthistory(m, ply, 2)
        } else {
            let captured = data.board.piece_at_square(m.capture_square()).map(|p| p.kind());
            data.noisy_history.get(
                data.board.piece_at_square(m.from()),
                m.to(),
                captured,
                data.board.threats(),
            )
        };

        if !Node::ROOT && !is_loss(best_score) {
            // Late Move Pruning (LMP)
            if !in_check
                && !is_direct_check
                && !is_win(beta)
                && is_quiet
                && move_count as i32 > (2976 + (1363 + 263 * improving as i32) * depth * depth) / 1024
            {
                skip_quiets = true;
                continue;
            }

            // Futility Pruning (FP)
            if !in_check
                && !is_direct_check
                && is_quiet
                && depth < 8
                && static_eval + 93 * depth + 142 + 51 * history / 1024 <= alpha
            {
                skip_quiets = true;
                continue;
            }

            // History Pruning (HP)
            if !in_check && is_quiet && depth <= 6 && history < -1481 * depth {
                continue;
            }

            // Static Exchange Evaluation Pruning (SEE Pruning)
            let threshold = (-123 * depth * depth - 44 * depth + 14).min(-35);
            if !in_check && !is_quiet && !data.board.see(m, threshold) {
                continue;
            }
        }

        let initial_nodes = data.nodes();

        // Make Move
        data.make_move(m, ply);
        let new_depth = (depth - 1) + ((move_count == 1) as i32 * extension);
        let mut score = -Score::INFINITY;

        // Late Move Reductions (LMR)
        if depth >= 2 && move_count > 1 {
            let mut r = LMR_TABLE[is_quiet as usize][depth.min(127) as usize][move_count.min(63)];
            r += 1200 * cutnode as i32;
            r -= 1200 * tt_was_pv as i32;
            r -= 800 * is_direct_check as i32;
            r += 215 * !improving as i32;
            r += 454 * (tt_score.is_some_and(|s| s <= alpha)) as i32;
            r += 303 * (tt_depth.is_some_and(|d| d < depth)) as i32;
            r -= 439 * history / 4096;

            let reduction = r / 1024;
            let reduced_depth = (new_depth - reduction).max(1) + Node::PV as i32;

            data.stack[ply].reduction = r;
            score = -search::<NonPV>(data, reduced_depth, -alpha - 1, -alpha, ply + 1, true);
            data.stack[ply].reduction = 0;

            if score > alpha && reduced_depth < new_depth {
                score = -search::<NonPV>(data, new_depth, -alpha - 1, -alpha, ply + 1, !cutnode);
            }
        } else if !Node::PV || move_count > 1 {
            score = -search::<NonPV>(data, new_depth, -alpha - 1, -alpha, ply + 1, !cutnode);
        }

        // Principal Variation Search (PVS)
        if Node::PV && (move_count == 1 || score > alpha) {
            score = -search::<PV>(data, new_depth, -beta, -alpha, ply + 1, false);
        }

        // Unmake Move
        data.unmake_move();

        if data.shared.status.get() == Status::STOPPED {
            return Score::TIMEOUT;
        }

        if Node::ROOT {
            let nodes = data.nodes();
            if let Some(root_move) = data.root_moves.iter_mut().find(|rm| rm.m == m) {
                root_move.nodes += nodes - initial_nodes;

                if move_count == 1 || score > alpha {
                    root_move.score = score;
                    root_move.display_score = score;

                    root_move.searched_depth = data.root_depth;

                    root_move.upperbound = false;
                    root_move.lowerbound = false;

                    if score <= alpha {
                        root_move.display_score = alpha;
                        root_move.upperbound = true;
                    } else if score >= beta {
                        root_move.display_score = beta;
                        root_move.lowerbound = true;
                    }

                    root_move.pv.commit(&data.pv.inner[1][..data.pv.len[1]]);
                    root_move.sel_depth = data.sel_depth;
                } else {
                    root_move.score = -Score::INFINITY;
                }
            };
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = Some(m);
                bound = Bound::Exact;

                if !Node::ROOT && Node::PV {
                    data.pv.add(m, ply);
                }

                // Cutoff
                if score >= beta {
                    bound = Bound::Lower;
                    break;
                }

                alpha = score;
            }
        }

        // Add searched quiet/noisy moves to list
        if best_move != Some(m) && move_count < 32 {
            if is_quiet {
                quiets_searched.push(m);
            } else {
                noisies_searched.push(m);
            }
        }
    }

    if move_count == 0 {
        if excluded {
            return -Score::INFINITY;
        }

        if in_check {
            return -Score::MATE + ply as i32;
        } else {
            return Score::DRAW;
        }
    }

    if let Some(m) = best_move {
        let is_quiet = m.kind().is_quiet();

        let quiet_bonus = (325 * depth).min(947) - 225;
        let quiet_malus = (289 * depth).min(948) - 235;

        let noisy_bonus = (253 * depth).min(1060) - 190;
        let noisy_malus = (298 * depth).min(938) - 271;

        let cont_bonus = (315 * depth).min(1056) - 194;
        let cont_malus = (305 * depth).min(1082) - 270;

        let threats = data.board.threats();

        if is_quiet {
            let piece = data.board.piece_at_square(m.from());
            let to = m.to();
            let pawn_key = data.board.state.keys.pawn;
            data.pawn_history.update(pawn_key, piece, to, quiet_bonus);
            data.quiet_history.update(threats, stm, m, quiet_bonus);
            data.update_conthistories(m, ply, cont_bonus);
            for quiet_move in quiets_searched.iter() {
                let piece = data.board.piece_at_square(quiet_move.from());
                let to = quiet_move.to();
                data.pawn_history.update(pawn_key, piece, to, -quiet_malus);
                data.quiet_history.update(threats, stm, *quiet_move, -quiet_malus);
                data.update_conthistories(*quiet_move, ply, -cont_malus);
            }
        } else {
            let piece = data.board.piece_at_square(m.from());
            let to = m.to();
            let captured = data.board.piece_at_square(m.capture_square()).map(|e| e.kind());
            data.noisy_history.update(piece, to, captured, threats, noisy_bonus);
        }

        for m in noisies_searched.iter() {
            let piece = data.board.piece_at_square(m.from());
            let to = m.to();
            let captured = data.board.piece_at_square(m.capture_square()).map(|e| e.kind());
            data.noisy_history.update(piece, to, captured, threats, -noisy_malus);
        }
    }

    // Prior Countermove Bonus
    if !Node::ROOT && bound == Bound::Upper && data.stack[ply - 1].m.kind().is_quiet() {
        let bonus = (122 * depth - 76).min(1194);
        data.quiet_history
            .update(data.stack[ply - 1].threats, !stm, data.stack[ply - 1].m, bonus);
    }

    if !excluded {
        data.shared.tt.add_entry(
            best_move.unwrap_or(Move::NONE),
            best_score,
            raw_eval,
            bound,
            data.board.hash(),
            depth,
            ply,
            tt_pv,
        );

        // Update Correction Histories
        if !in_check
            && best_move.is_none_or(|m| m.kind().is_quiet())
            && ((bound == Bound::Lower && best_score >= static_eval)
                || (bound == Bound::Upper && best_score <= static_eval)
                || bound == Bound::Exact)
        {
            data.update_correction_histories(best_score - static_eval, depth, ply);
        }
    }

    best_score
}

pub fn quiesce<Node: NodeType>(data: &mut SearchData, mut alpha: i32, beta: i32, ply: isize) -> i32 {
    if Node::PV {
        data.pv.clear(ply);
        data.sel_depth = data.sel_depth.max(ply as i32);
    }

    if data.board.is_draw() {
        return Score::DRAW;
    }

    if data.id == 0 && data.time.hard_limit(data) {
        data.shared.status.stop();
        return Score::TIMEOUT;
    }

    let tt_entry = data.shared.tt.entry(data.board.hash(), ply);
    let tt_bound = tt_entry.as_ref().map(|e| e.bound());
    let tt_score = tt_entry.as_ref().map(|e| e.score()).filter(|s| *s != Score::NONE);
    let tt_was_pv = tt_entry.as_ref().map(|e| e.is_pv()).unwrap_or(false);
    let tt_pv = Node::PV || tt_was_pv;

    // TT Cutoffs
    if !Node::PV
        && let Some(tt_score) = tt_score
        && tt_bound.is_some_and(|b| match b {
            Bound::Lower => tt_score >= beta,
            Bound::Upper => tt_score < alpha,
            Bound::Exact => true,
            Bound::None => false,
        })
    {
        return tt_score;
    }

    let in_check = data.board.king_in_check();

    if ply >= MAX_PLY as isize - 1 {
        if in_check {
            return Score::DRAW;
        } else {
            return data.network.evaluate(&data.board);
        }
    }

    // Evaluation
    let raw_eval;
    let static_eval;
    let mut best_score;

    if in_check {
        raw_eval = Score::NONE;
        static_eval = -Score::INFINITY;
        best_score = static_eval;
    } else if let Some(e) = &tt_entry
        && e.eval() != Score::NONE
    {
        raw_eval = e.eval();
        static_eval = raw_eval + data.correction(ply);
        best_score = static_eval;
    } else {
        raw_eval = data.network.evaluate(&data.board);
        static_eval = raw_eval + data.correction(ply);
        best_score = static_eval
    };

    if tt_entry.is_none() {
        data.shared.tt.add_entry(
            Move::NONE,
            Score::NONE,
            raw_eval,
            Bound::None,
            data.board.hash(),
            0,
            ply,
            tt_pv,
        );
    }

    // Stand Pat
    if best_score >= beta {
        if tt_entry.is_none() {
            data.shared.tt.add_entry(
                Move::NONE,
                best_score,
                raw_eval,
                Bound::Lower,
                data.board.hash(),
                0,
                ply,
                tt_pv,
            );
        }

        return best_score;
    }

    if best_score > alpha {
        alpha = best_score;
    }

    let tt_move = tt_entry.map(|e| e.best_move()).filter(|m| !m.is_null());

    let mut move_picker = MovePicker::new(tt_move);
    let mut move_count = 0;
    let mut bound = Bound::Upper;
    let mut best_move: Option<Move> = None;
    let skip_quiets = !in_check;

    while let Some(m) = move_picker.next(data, skip_quiets, ply) {
        move_count += 1;

        if !is_loss(best_score) {
            // Late Move Pruning (LMP)
            if move_count >= 3 && !data.board.is_direct_check(m) {
                break;
            }

            // Static Exchange Evaluation Pruning (SEE Pruning)
            if !data.board.see(m, -112) {
                continue;
            }
        }

        data.make_move(m, ply);
        let score = -quiesce::<Node>(data, -beta, -alpha, ply + 1);
        data.unmake_move();

        if data.shared.status.get() == Status::STOPPED {
            return Score::TIMEOUT;
        }

        if score > best_score {
            best_score = score;

            if score > alpha {
                best_move = Some(m);

                if Node::PV {
                    data.pv.add(m, ply);
                }

                // Cutoff
                if score >= beta {
                    bound = Bound::Lower;
                    break;
                }

                alpha = score;
            }
        }
    }

    if in_check && move_count == 0 {
        return -Score::MATE + ply as i32;
    }

    if best_score >= beta
        && let Some(m) = best_move
        && !m.kind().is_quiet()
    {
        // Add noisy bonus to history
        let piece = data.board.piece_at_square(m.from());
        let to = m.to();
        let captured = data.board.piece_at_square(m.capture_square()).map(|e| e.kind());
        data.noisy_history
            .update(piece, to, captured, data.board.threats(), 106);
    }

    data.shared.tt.add_entry(
        best_move.unwrap_or(Move::NONE),
        best_score,
        raw_eval,
        bound,
        data.board.hash(),
        0,
        ply,
        tt_pv,
    );

    best_score
}
