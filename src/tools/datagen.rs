use std::sync::Mutex;

use crate::{
    board::{Board, movegen::MoveGenKind},
    search::data::SearchData,
    types::pseudo_rand,
};

#[derive(Debug)]
pub struct BadRandomBoard;

static SEED: Mutex<u64> = Mutex::new(0);

pub fn generate_random_openings(amount: usize, plies: isize, seed: Option<u64>) {
    if let Some(seed) = seed {
        *SEED.lock().unwrap() = seed;
    } else {
        *SEED.lock().unwrap() = rand::random();
    }

    for _ in 0..amount {
        let mut random_number = pseudo_rand(&mut SEED.lock().unwrap());
        let mut random_board = randomize_from_startpos(plies, random_number);

        // Regenerate imbalanced positions
        while random_board.is_err() {
            random_number = pseudo_rand(&mut SEED.lock().unwrap());
            random_board = randomize_from_startpos(plies, random_number);
        }

        println!("info string genfens {}", random_board.unwrap().to_fen());
    }
}

pub fn randomize_from_startpos(plies: isize, random_number: u64) -> Result<Board, BadRandomBoard> {
    let mut data = SearchData::default();
    let mut state = random_number;
    data.network.full_refresh(&data.board);

    let plies = if rand::random_bool(0.5) { plies } else { plies + 1 };

    for ply in 0..plies {
        let move_list = data.board.generate_moves(MoveGenKind::All);
        // Check if there's atleast one legal move first
        if move_list.is_empty() {
            return Err(BadRandomBoard);
        }

        let index = pseudo_rand(&mut state) % move_list.len() as u64;
        let random_move = move_list.get(index as usize).mv;
        data.make_move(random_move, ply);
    }

    // Check if eval is not too uneven
    if data.network.evaluate(&data.board).abs() > 1000 {
        return Err(BadRandomBoard);
    }

    Ok(data.board)
}

#[cfg(test)]
pub mod tests {
    use crate::tools::datagen::generate_random_openings;

    #[test]
    fn test_fengen() {
        generate_random_openings(1, 8, Some(3493));
    }
}
