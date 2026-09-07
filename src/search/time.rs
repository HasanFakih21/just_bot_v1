use std::time::{Duration, Instant};

use crate::{search::data::SearchData, types::MOVE_OVERHEAD};

// Some settings don't do anything yet
#[derive(Debug, Clone, Default)]
pub struct TimeSettings {
    pub time: Option<u64>,
    pub inc: u64,
    pub movestogo: Option<u64>,
    pub depth: Option<i32>,
    pub nodes: Option<Nodes>,
    pub mate: Option<u64>,
    pub movetime: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TimeManager {
    pub clock: Instant,
    pub limits: Limits,
}

#[derive(Debug, Clone)]
pub enum Nodes {
    Soft(u64),
    Hard(u64),
}

#[derive(Debug, Clone, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    exact: bool,
    soft_bound: Option<Duration>,
    hard_bound: Option<Duration>,
    nodes: Option<Nodes>,
    mate: Option<u64>,
}

impl TimeManager {
    pub fn new(settings: TimeSettings, full_moves: usize) -> TimeManager {
        let mut limits = Limits::default();
        if let Some(remaining_time) = settings.time
            && settings.movestogo.is_none()
        {
            let soft_scale = 0.06 - 0.05 * (-0.035 * full_moves as f64).exp();
            let hard_scale = 0.75;
            let max_time = remaining_time.saturating_sub(MOVE_OVERHEAD);

            let soft = (soft_scale * max_time as f64 + settings.inc as f64 * 0.75) as u64;
            let hard = (hard_scale * max_time as f64 + settings.inc as f64 * 0.75) as u64;

            limits.soft_bound = Some(Duration::from_millis(soft.min(max_time)));
            limits.hard_bound = Some(Duration::from_millis(hard.min(max_time)));
        } else if let Some(remaining_time) = settings.time
            && let Some(moves) = settings.movestogo
        {
            let max_time = remaining_time.saturating_sub(MOVE_OVERHEAD);
            let base = (max_time as f64 / moves as f64) + settings.inc as f64 * 0.75;

            limits.soft_bound = Some(Duration::from_millis(((1.0 * base) as u64).min(max_time)));
            limits.hard_bound = Some(Duration::from_millis(((5.0 * base) as u64).min(max_time)));
        } else if let Some(movetime) = settings.movetime {
            limits.soft_bound = Some(Duration::from_millis(movetime.saturating_sub(MOVE_OVERHEAD)));
            limits.hard_bound = Some(Duration::from_millis(movetime.saturating_sub(MOVE_OVERHEAD)));
            limits.exact = true;
        }

        limits.nodes = settings.nodes;
        limits.depth = settings.depth;
        limits.mate = settings.mate;

        TimeManager {
            clock: Instant::now(),
            limits,
        }
    }

    pub fn start_clock(&mut self) {
        self.clock = Instant::now();
    }

    pub fn elapsed(&self) -> Duration {
        self.clock.elapsed()
    }

    pub fn soft_limit(&self, data: &SearchData, multiplier: impl Fn() -> f32) -> bool {
        if self.limits.exact
            && let Some(limit) = self.limits.soft_bound
        {
            self.elapsed() >= Duration::from_secs_f32(limit.as_secs_f32())
        } else if let Some(limit) = self.limits.soft_bound {
            self.elapsed() >= Duration::from_secs_f32(limit.as_secs_f32() * multiplier())
        } else if let Some(Nodes::Soft(limit) | Nodes::Hard(limit)) = &self.limits.nodes {
            data.nodes() >= *limit
        } else {
            false
        }
    }

    pub fn hard_limit(&self, data: &SearchData) -> bool {
        if data.id != 0 || data.root_depth <= 1 {
            return false;
        }

        if data.nodes().is_multiple_of(2048)
            && let Some(limit) = self.limits.hard_bound
        {
            self.elapsed() >= limit
        } else if let Some(Nodes::Hard(limit)) = &self.limits.nodes {
            data.nodes() >= *limit
        } else {
            false
        }
    }
}
