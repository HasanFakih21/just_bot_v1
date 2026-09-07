use std::time::{Duration, Instant};

use crate::{search::data::SearchData, types::MOVE_OVERHEAD};

#[derive(Debug, Clone)]
pub struct TimeManager {
    pub clock: Instant,
    pub limit: Limit,
    soft_bound: Option<Duration>,
    hard_bound: Option<Duration>,
}

#[derive(Debug, Clone)]
pub enum NodeKind {
    Soft(u64),
    Hard(u64),
}

#[derive(Debug, Clone)]
pub enum Limit {
    Infinite,
    Exact(u64),
    Nodes(NodeKind),
    Mate(u64),
    Fischer(u64, u64),
    Cyclic(u64, u64, u64),
    Depth(i32),
}

impl TimeManager {
    pub fn new(limit: Limit, full_moves: usize) -> TimeManager {
        let soft_bound;
        let hard_bound;

        match limit {
            Limit::Fischer(main, inc) => {
                let soft_scale = 0.06 - 0.05 * (-0.035 * full_moves as f64).exp();
                let hard_scale = 0.75;
                let max_time = main.saturating_sub(MOVE_OVERHEAD);

                let soft = (soft_scale * max_time as f64 + inc as f64 * 0.75) as u64;
                let hard = (hard_scale * max_time as f64 + inc as f64 * 0.75) as u64;

                soft_bound = Some(Duration::from_millis(soft.min(max_time)));
                hard_bound = Some(Duration::from_millis(hard.min(max_time)));
            }
            Limit::Cyclic(main, inc, moves) => {
                let max_time = main.saturating_sub(MOVE_OVERHEAD);
                let base = (max_time as f64 / moves as f64) + inc as f64 * 0.75;

                soft_bound = Some(Duration::from_millis(((1.0 * base) as u64).min(max_time)));
                hard_bound = Some(Duration::from_millis(((5.0 * base) as u64).min(max_time)));
            }
            Limit::Exact(main) => {
                soft_bound = Some(Duration::from_millis(main.saturating_sub(MOVE_OVERHEAD)));
                hard_bound = Some(Duration::from_millis(main.saturating_sub(MOVE_OVERHEAD)));
            }
            _ => {
                soft_bound = None;
                hard_bound = None;
            }
        }

        TimeManager {
            hard_bound,
            soft_bound,
            clock: Instant::now(),
            limit,
        }
    }

    pub fn start_clock(&mut self) {
        self.clock = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        self.clock.elapsed()
    }

    pub fn soft_limit(&self, data: &SearchData, multiplier: impl Fn() -> f32) -> bool {
        match self.limit {
            Limit::Fischer(_, _) | Limit::Cyclic(_, _, _) if let Some(limit) = self.soft_bound => {
                self.elapsed() >= Duration::from_secs_f32(limit.as_secs_f32() * multiplier())
            }
            Limit::Exact(_) if let Some(limit) = self.soft_bound => self.elapsed() >= limit,
            Limit::Nodes(NodeKind::Soft(limit) | NodeKind::Hard(limit)) => data.nodes() >= limit,
            _ => false,
        }
    }

    pub fn hard_limit(&self, data: &SearchData) -> bool {
        if data.id != 0 || data.root_depth <= 1 {
            return false;
        }

        match self.limit {
            Limit::Fischer(_, _) | Limit::Cyclic(_, _, _) | Limit::Exact(_)
                if let Some(limit) = self.hard_bound
                    && data.nodes().is_multiple_of(2048) =>
            {
                self.elapsed() >= limit
            }
            Limit::Nodes(NodeKind::Hard(limit)) => data.nodes() >= limit,
            Limit::Nodes(NodeKind::Soft(_)) => data.nodes() >= 800_000,
            _ => false,
        }
    }
}
