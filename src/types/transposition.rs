use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

use crate::types::{Score, from_tt, is_decisive, moves::Move};

const TT_DEFAULT_SIZE: usize = 16;
const MEGABYTE: usize = 1024 * 1024;
const MAX_AGE: u8 = 31;

const SIZE_OF_CLUSTER: usize = std::mem::size_of::<Cluster>();
const NUM_ENTRIES_PER_CLUSTER: usize = 3;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bound {
    None,
    Exact,
    Upper,
    Lower,
}

#[derive(Debug, Copy, Clone)]
pub struct Flags(u8);

impl Flags {
    pub fn new(pv: bool, bound: Bound, age: u8) -> Self {
        debug_assert!(age <= MAX_AGE);

        Flags(pv as u8 | (bound as u8) << 1 | age << 3)
    }

    pub const fn bound(&self) -> Bound {
        match (self.0 & 0b0000_0110) >> 1 {
            0 => Bound::None,
            1 => Bound::Exact,
            2 => Bound::Upper,
            3 => Bound::Lower,
            _ => unreachable!(),
        }
    }

    pub const fn pv(&self) -> bool {
        (self.0 & 1) != 0
    }

    pub const fn age(&self) -> u8 {
        self.0 >> 3
    }
}

#[derive(Debug, Clone)]
pub struct Entry {
    key: u16,        // 2 bytes
    best_move: Move, // 2 bytes
    score: i16,      // 2 bytes
    eval: i16,       // 2 bytes
    depth: u8,       // 1 byte
    flags: Flags,    // 1 byte
}

impl Entry {
    pub fn new(key: u16, best_move: Move, score: i16, eval: i16, depth: u8, flags: Flags) -> Self {
        Entry {
            key,
            best_move,
            score,
            eval,
            depth,
            flags,
        }
    }

    pub const fn relative_age(&self, tt_age: u8) -> i32 {
        ((32 + tt_age - self.flags.age()) & MAX_AGE) as i32
    }

    pub fn key(&self) -> u16 {
        self.key
    }

    pub fn bound(&self) -> Bound {
        self.flags.bound()
    }

    pub fn is_pv(&self) -> bool {
        self.flags.pv()
    }

    pub fn best_move(&self) -> Move {
        self.best_move
    }

    pub fn score(&self) -> i32 {
        self.score as i32
    }

    pub fn depth(&self) -> i32 {
        self.depth as i32
    }

    pub fn eval(&self) -> i32 {
        self.eval as i32
    }
}

#[repr(align(32))]
pub struct Cluster {
    entries: [Entry; NUM_ENTRIES_PER_CLUSTER],
}

impl Cluster {
    pub fn lookup_key(&self, key: u16) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.key() == key && (e.bound() != Bound::None || e.score() == Score::NONE))
    }
}

#[derive(Debug)]
pub struct TranspositionTable {
    clusters: AtomicPtr<Cluster>,
    len: AtomicUsize,
    age: AtomicU8,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let (len, p) = unsafe { allocate_entries(size_mb) };
        TranspositionTable {
            clusters: AtomicPtr::new(p),
            len: AtomicUsize::new(len),
            age: AtomicU8::new(0),
        }
    }

    pub fn resize(&self, size_mb: usize) {
        unsafe { deallocate_entries(self.len(), self.ptr()) }
        let (new_len, new_p) = unsafe { allocate_entries(size_mb) };
        self.len.store(new_len, Ordering::Relaxed);
        self.clusters.store(new_p, Ordering::Relaxed);
        self.age.store(0, Ordering::Relaxed);
    }

    fn ptr(&self) -> *mut Cluster {
        self.clusters.load(Ordering::Relaxed)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_entry(
        &self,
        best_move: Move,
        mut score: i32,
        eval: i32,
        bound: Bound,
        hash: u64,
        depth: i32,
        ply: isize,
        pv: bool,
    ) {
        let index = index(hash, self.len());
        debug_assert!(index < self.len());
        debug_assert!(score != Score::TIMEOUT);

        let cluster = unsafe { &mut *self.ptr().add(index) };
        let key = hash as u16;
        let tt_age = self.age();

        let replacement_index = cluster.lookup_key(key).unwrap_or_else(|| {
            let mut index = 0;
            let mut worst_quality = i32::MAX;

            for (i, entry) in cluster.entries.iter().enumerate() {
                if entry.flags.bound() == Bound::None {
                    index = i;
                    break;
                }

                let quality = entry.depth as i32 - 4 * entry.relative_age(tt_age);
                if quality < worst_quality {
                    index = i;
                    worst_quality = quality;
                }
            }

            index
        });

        let entry = &mut cluster.entries[replacement_index];
        let same_key = key == entry.key();

        // Keep the stored move if the new move is null for the same position
        if !(same_key && best_move.is_null()) {
            entry.best_move = best_move;
        }

        // Don't replace entry if this is true
        if same_key && depth + 4 + 2 * pv as i32 <= entry.depth() && entry.flags.age() == tt_age {
            return;
        }

        // Adjust mate scores
        if is_decisive(score) && score != Score::NONE {
            score += score.signum() * ply as i32;
        }

        // Replace entry
        entry.key = key;
        entry.score = score as i16;
        entry.eval = eval as i16;
        entry.depth = depth as u8;
        entry.flags = Flags::new(pv, bound, tt_age);
    }

    pub fn clear(&self) {
        self.age.store(0, Ordering::Relaxed);
        unsafe { self.ptr().write_bytes(0, self.len()) }
    }

    pub fn entry(&self, hash: u64, ply: isize) -> Option<Entry> {
        let index = index(hash, self.len());
        debug_assert!(index < self.len());

        let cluster = unsafe { &*self.ptr().add(index) };
        let index = cluster.lookup_key(hash as u16)?;
        let entry = &cluster.entries[index];

        Some(Entry {
            key: entry.key,
            best_move: entry.best_move,
            score: from_tt(entry.score, ply),
            eval: entry.eval,
            depth: entry.depth,
            flags: entry.flags,
        })
    }

    pub fn hashfull(&self) -> usize {
        let mut count = 0;
        let clusters = unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) };

        for c in clusters.iter().take(1000) {
            for e in c.entries.iter() {
                if e.flags.bound() != Bound::None && e.flags.age() == self.age() {
                    count += 1;
                }
            }
        }

        count / NUM_ENTRIES_PER_CLUSTER
    }

    pub fn age(&self) -> u8 {
        self.age.load(Ordering::Relaxed)
    }

    pub fn increase_age(&self) {
        let current_age = self.age();
        self.age.store((current_age + 1) & MAX_AGE, Ordering::Relaxed);
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn prefetch(&self, hash: u64) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            let index = index(hash, self.len());
            let ptr = self.ptr().add(index);
            _mm_prefetch::<_MM_HINT_T0>(ptr.cast());
        }

        #[cfg(not(target_arch = "x86_64"))]
        let _ = hash;
    }
}

// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
const fn index(hash: u64, len: usize) -> usize {
    (((hash as u128) * (len as u128)) >> 64) as usize
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new(TT_DEFAULT_SIZE)
    }
}

impl Drop for TranspositionTable {
    fn drop(&mut self) {
        unsafe { deallocate_entries(self.len(), self.ptr()) };
    }
}

unsafe fn allocate_entries(size_mb: usize) -> (usize, *mut Cluster) {
    let size = size_mb * MEGABYTE;
    let num_entries = size / SIZE_OF_CLUSTER;

    let layout = std::alloc::Layout::from_size_align(size, align_of::<Cluster>()).unwrap();
    let p = unsafe { std::alloc::alloc_zeroed(layout) };

    (num_entries, p.cast())
}

unsafe fn deallocate_entries(len: usize, p: *mut Cluster) {
    let size = SIZE_OF_CLUSTER * len;
    let layout = std::alloc::Layout::from_size_align(size, align_of::<Cluster>()).unwrap();

    unsafe { std::alloc::dealloc(p.cast(), layout) };
}

#[cfg(test)]
mod tests {
    use crate::types::{Bound, Flags};

    #[test]
    fn test_flags() {
        let flag = Flags::new(true, Bound::Lower, 23);
        println!("{:b}", flag.0);

        assert_eq!(flag.bound(), Bound::Lower);
        assert!(flag.pv());
        assert_eq!(23, flag.age());
    }
}
