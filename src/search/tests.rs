use super::*;
use crate::board::Board;

#[test]
fn test_order_moves() {
    let data = SearchData {
        board: Board::from_fen("rnbqkb1r/pp3p2/4pnpp/1p1p2N1/1Q1P4/BP2P3/P1PN1PPP/R3K2R b KQkq - 0 1").unwrap(),
        ..Default::default()
    };

    let mut move_picker = MovePicker::new(None);
    let first_move = move_picker.next(&data, false, 0).unwrap();

    assert_eq!(first_move, Move::new(Square::F8, Square::B4, MoveKind::Capture));

    let data = SearchData {
        board: Board::from_fen("rnbq1rk1/pN1p1ppp/4n2b/2p1p3/N1BP3R/2P2Q2/PP3PPP/2B1K2R w K - 0 1").unwrap(),
        ..Default::default()
    };
    let mut move_picker = MovePicker::new(None);
    let first_move = move_picker.next(&data, false, 0).unwrap();

    assert_eq!(first_move, Move::new(Square::B7, Square::D8, MoveKind::Capture));
}

#[test]
fn test_bugged_position() {
    let mut board = Board::from_fen("6k1/5pp1/7p/8/5Pn1/2R4P/B5P1/4qQ1K b - - 6 39").unwrap();
    println!("Hash: {}", board.state.keys.full);
    // Position hash: 6128121706435820836

    board = Board::from_fen("6k1/5pp1/7p/8/5Pn1/2RQ3P/B4qP1/6K1 w - - 3 38").unwrap();
    println!("Hash 2: {}", board.state.keys.full);
    // Position hash: 16381162810209017462

    board = Board::from_fen("6k1/5pp1/7p/8/5Pnq/2RQ3P/B5P1/6K1 b - - 2 37").unwrap();
    println!("Hash 3: {}", board.state.keys.full);
    // Position hash: 3246015867840709621
}
