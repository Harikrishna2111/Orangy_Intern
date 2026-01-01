import chess.pgn

def build_anand_dataset(pgn_path, out_path):
    samples = []

    with open(pgn_path, encoding="utf-8", errors="ignore") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")

            anand_white = "anand" in white.lower()
            anand_black = "anand" in black.lower()

            if not (anand_white or anand_black):
                continue

            board = game.board()
            history = []

            for move in game.mainline_moves():
                san = board.san(move)

                is_anand_turn = (
                    (board.turn and anand_white) or
                    (not board.turn and anand_black)
                )

                if is_anand_turn:
                    context = " ".join(history)
                    samples.append(f"{context} => {san}")

                history.append(san)
                board.push(move)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s + "\n")

if __name__ == "__main__":
    build_anand_dataset(
        "data/anand_games.PGN",
        "data/anand_train.txt"
    )
