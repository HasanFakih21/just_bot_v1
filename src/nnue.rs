use crate::{
    board::Board,
    nnue::{
        accumulator::{Accumulator, Delta, DualAccumulators},
        cache::AccumulatorCache,
    },
    types::{MAX_PLY, Move, OptionPiece, Piece, Side, Square},
};

mod accumulator;
mod cache;
mod simd {
    #[cfg(target_feature = "avx512f")]
    mod avx512;
    #[cfg(target_feature = "avx512f")]
    pub use avx512::*;

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    mod avx2;
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    pub use avx2::*;

    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    mod scalar;
    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    pub use scalar::*;
}

const HIDDEN_SIZE: usize = 1024;
const SCALE: i32 = 400;
const NUM_OUTPUT_BUCKETS: usize = 8;
const QA: i16 = 255;
const QB: i16 = 64;

#[rustfmt::skip]
const BUCKET_LAYOUT: [usize; 32] = [
    0, 0, 1, 1, 
    1, 1, 1, 1,
    1, 1, 1, 1, 
    2, 2, 2, 2,
    2, 2, 2, 2,
    2, 2, 2, 2, 
    2, 2, 2, 2,
    3, 3, 3, 3,
];

const NUM_INPUT_BUCKETS: usize = 4;

pub static MODEL: Parameters = unsafe { std::mem::transmute(*include_bytes!(env!("MODEL"))) };

pub struct Network {
    parameters: &'static Parameters,
    stack: Box<[DualAccumulators]>,
    index: usize,
    cache: AccumulatorCache,
}

impl Network {
    pub fn new() -> Self {
        Network {
            parameters: &MODEL,
            stack: vec![DualAccumulators::new(); MAX_PLY].into_boxed_slice(),
            index: 0,
            cache: AccumulatorCache::new(&MODEL),
        }
    }

    pub fn can_update(&self, pov: Side) -> Option<usize> {
        for i in (0..=self.index).rev() {
            if self.stack[i].accurate[pov] {
                return Some(i);
            }

            let Some(delta) = &self.stack[i].delta else {
                return None;
            };

            let needs_refresh = delta.piece == Piece::King
                && delta.stm == pov
                && input_context(delta.m.from() ^ (56 * (delta.stm == Side::Black) as u8))
                    != input_context(delta.m.to() ^ (56 * (delta.stm == Side::Black) as u8));

            if needs_refresh {
                return None;
            }
        }

        None
    }

    pub fn push(&mut self, board: &Board, m: Move) {
        debug_assert!(board.piece_at_square(m.from()) != OptionPiece::None);
        self.index += 1;
        self.stack[self.index].delta = Some(Delta {
            m,
            stm: board.state.side_to_move,
            piece: board.piece_at_square(m.from()).unwrap().kind(),
            captured: if m.is_capture() {
                Some(board.piece_at_square(m.capture_square()).unwrap().kind())
            } else {
                None
            },
        });
        self.stack[self.index].accurate = [false; 2];
    }

    pub fn pop(&mut self) {
        self.index -= 1;
    }

    pub fn evaluate(&mut self, board: &Board) -> i32 {
        for pov in [Side::White, Side::Black] {
            if self.stack[self.index].accurate[pov] {
                continue;
            }

            match self.can_update(pov) {
                Some(last_accurate) => {
                    // Update all the not yet updated accumulators
                    let king_square = board.king_square(pov);
                    for index in last_accurate..self.index {
                        if let Some((prev, [current, ..])) = self.stack.split_at_mut_checked(index + 1) {
                            current.update(&prev[index], board, pov, king_square, self.parameters);
                        }
                    }
                }
                None => self.stack[self.index].refresh(board, pov, self.parameters, &mut self.cache),
            }
        }

        let eval = self.output_layer(board);
        #[cfg(not(feature = "datagen"))]
        let eval = board.scale_eval(eval);
        eval
    }

    #[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
    pub fn output_layer(&self, board: &Board) -> i32 {
        const CHUNKS: usize = 16 / simd::I32_CHUNK;

        let stm = board.state.side_to_move;
        let (us, them) = (
            self.stack[self.index].values[stm].vals.as_ptr(),
            self.stack[self.index].values[!stm].vals.as_ptr(),
        );

        let bucket = output_bucket(board);
        let weights = &self.parameters.output_weights[bucket].as_ptr();

        // Initialise output.
        let mut sums = [simd::zeroed(); CHUNKS];

        unsafe {
            // Side-To-Move Accumulator -> Output.
            for i in (0..HIDDEN_SIZE).step_by(simd::I16_CHUNK) {
                let x = us.add(i);
                let w = weights.add(i);
                let v = simd::clamp_i16(*x.cast(), simd::zeroed(), simd::splat_i16(QA));
                let t = simd::mul_low_i16(v, *w.cast());
                let p = simd::madd_i16_to_i32(v, t);
                sums[0] = simd::add_i32(sums[0], p);
            }

            // Not-Side-To-Move Accumulator -> Output.
            for i in (0..HIDDEN_SIZE).step_by(simd::I16_CHUNK) {
                let x = them.add(i);
                let w = weights.add(HIDDEN_SIZE + i);
                let v = simd::clamp_i16(*x.cast(), simd::zeroed(), simd::splat_i16(QA));
                let t = simd::mul_low_i16(v, *w.cast());
                let p = simd::madd_i16_to_i32(v, t);
                sums[CHUNKS - 1] = simd::add_i32(sums[CHUNKS - 1], p);
            }
        }

        let mut output = simd::reduce_add_i32(&sums);
        output /= i32::from(QA);
        output += i32::from(self.parameters.output_bias[bucket]);
        output *= SCALE;
        output /= i32::from(QA) * i32::from(QB);
        output
    }

    #[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
    pub fn output_layer(&self, board: &Board) -> i32 {
        // Initialise output.
        let mut output = 0;
        let stm = board.state.side_to_move;
        let (us, them) = (self.stack[self.index].values[stm], self.stack[self.index].values[!stm]);

        let bucket = output_bucket(board);
        let weights = &self.parameters.output_weights[bucket];

        // Side-To-Move Accumulator -> Output.
        for (&input, &weight) in us.vals.iter().zip(&weights[..HIDDEN_SIZE]) {
            let mut y = i32::from(input).clamp(0, i32::from(QA));
            y *= y;
            output += y * i32::from(weight);
        }

        // Not-Side-To-Move Accumulator -> Output.
        for (&input, &weight) in them.vals.iter().zip(&weights[HIDDEN_SIZE..]) {
            let mut y = i32::from(input).clamp(0, i32::from(QA));
            y *= y;
            output += y * i32::from(weight);
        }

        output /= i32::from(QA);
        output += i32::from(self.parameters.output_bias[bucket]);
        output *= SCALE;
        output /= i32::from(QA) * i32::from(QB);

        output
    }

    pub fn full_refresh(&mut self, board: &Board) {
        for pov in [Side::White, Side::Black] {
            self.stack[self.index].refresh(board, pov, self.parameters, &mut self.cache);
        }
    }
}

impl Default for Network {
    fn default() -> Self {
        Self::new()
    }
}

#[repr(C)]
pub struct Parameters {
    feature_weights: [Accumulator; 768 * NUM_INPUT_BUCKETS],
    feature_bias: Accumulator,
    output_weights: [[i16; 2 * HIDDEN_SIZE]; NUM_OUTPUT_BUCKETS],
    output_bias: [i16; NUM_OUTPUT_BUCKETS],
}

#[inline]
// Input Bucket, Which Half
pub fn input_context(king_square: Square) -> (usize, bool) {
    (input_bucket(king_square), king_square.to_file() > 3)
}

#[inline]
fn input_bucket(king_square: Square) -> usize {
    let (rank, file) = king_square.to_rank_and_file();
    BUCKET_LAYOUT[rank * 4 + (file.min(7 - file))]
}

#[inline]
fn output_bucket(pos: &Board) -> usize {
    let divisor = 32usize.div_ceil(NUM_OUTPUT_BUCKETS);
    ((pos.all_occupancy().count_bits() - 2) / divisor).min(NUM_OUTPUT_BUCKETS - 1)
}

#[cfg(test)]
mod tests {

    use crate::{
        board::{Board, movegen::MoveGenKind},
        nnue::output_bucket,
        search::data::SearchData,
        types::STARTING_FEN,
    };

    #[test]
    fn test_output_bucket() {
        let data = SearchData {
            board: Board::from_fen(STARTING_FEN).unwrap(),
            ..Default::default()
        };

        let bucket = output_bucket(&data.board);
        assert_eq!(bucket, 7);
    }

    #[test]
    fn test_nnue_make_unmake() {
        let mut data = SearchData {
            board: Board::from_fen("rnbq1rk1/pp3p2/4pnpp/1p1p2N1/3P4/1P2P3/PBPbKPPP/R6R w - - 2 4").unwrap(),
            ..Default::default()
        };

        data.network.full_refresh(&data.board);
        let first_eval = data.network.evaluate(&data.board);

        println!("First Eval: {}", first_eval);
        let _ = data.board.generate_moves(MoveGenKind::All);
        let m = data.board.parse_move("e2d1").unwrap();

        // Make the move
        data.make_move(m, 0);

        println!("Second Eval: {}", data.network.evaluate(&data.board));

        // Unmake the move
        data.unmake_move();

        let final_eval = data.network.evaluate(&data.board);
        println!("Final Eval: {}", final_eval);
        assert_eq!(final_eval, first_eval);
    }
}
