"""UCI-like interface"""
import engine.game as chess
import engine.bot as bot


def position(fen: str, moves: str) -> str:
    """
    Get the FEN notation of the board after the given move is made on the given board in FEN notation.
    The UCI command 'position startpos moves d2d4' translates into position('startpos', 'd2d4')
    Instead of setting the board as in normal UCI protocols,
    this function returns the resulting board since it is stateless.

    :param fen: the FEN notation of the chess piece positions
    :param moves: string of length 4 containing the source square and the destination square (eg. 'e4g3')
    :return: the FEN notation of the new board; return the same board if the move cannot be made
    """
    if fen == "startpos":
        board = chess.Board.create_standard_board()
    else:
        board = chess.Board.from_FEN(fen)
    source_square, destination_square = moves[:2], moves[2:]
    for move in board.current_player.calculate_legal_moves():
        if move.original_coordinate == source_square and move.destination_coordinate == destination_square:
            return move.execute().to_FEN()
    return fen


def go(fen: str, depth: int) -> str:
    """
    Sends the AI to search for the best move on the current board given as FEN notation with the given search depth.

    :param fen: the FEN notation of the chess board to analyze
    :param depth: the search depth of the bot
    :return: the best move, a string of length 4 containing the starting square and destination square (eg. 'e2e4')
    """
    pass

how_java_proto_failed = "7r/4k1p1/3ppp2/p6p/P2bn2P/3b4/r7/7K b - - 3 49"
