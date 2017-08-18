import numpy as np
from typing import Type, Union, List


#########
# CONST #
#########
WHITE, BLACK = "WHITE", "BLACK"
RANKS = ['1', '2', '3', '4', '5', '6', '7', '8']  # rows
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # columns


#########
# UTILS #
#########
def algebraic_notation_to_index(algebraic_notation: str) -> int:
    """
    :param algebraic_notation:
    :return: the index corresponding to the algebraic notation (top left corner is 0, bottom right corner is 63)

    >>> algebraic_notation_to_index("a1")
    56
    >>> algebraic_notation_to_index("h1")
    63
    >>> algebraic_notation_to_index("a8")
    0
    """
    file, rank = algebraic_notation
    return (7 - RANKS.index(rank)) * 8 + FILES.index(file)


def algebraic_notation_to_vector(algebraic_notation: str) -> np.ndarray:
    pass


def get_pawn_advance_direction(color: Union['WHITE', 'BLACK']) -> int:
    return -1 if color == WHITE else 1


##########
# Player #
##########
class Player:
    def __init__(self, board: 'Board', color: Union['WHITE', 'BLACK']):
        self.board = board
        self.color: str = color
        self.king = None


#########
# Board #
#########
class Board:
    def __init__(self,
                 active_color: Union['WHITE', 'BLACK'],
                 half_move_clock: int,
                 full_move_number: int,
                 en_passant_pawn=None):
        self.board: List[Union[None, 'Piece']] = [None] * 64
        self.active_color: str = active_color
        self.half_move_clock = half_move_clock
        self.full_move_number = full_move_number
        self.en_passant_pawn: Union[None, 'Pawn'] = en_passant_pawn
        self.white_pieces, self.black_pieces, self.white_king, self.black_king = [None] * 4
        self.update()
        # TODO refactor this and move into Player class

    def __getitem__(self, key: str) -> Union[None, 'Piece']:
        return self.board[algebraic_notation_to_index(key)]

    def __setitem__(self, key: str, value: Union[None, 'Piece']):
        self.board[algebraic_notation_to_index(key)] = value

    def __repr__(self) -> str:
        """
        :return: the current board in text format

        >>> print(Board.create_standard_board())
        r n b q k b n r
        p p p p p p p p
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        P P P P P P P P
        R N B Q K B N R
        """
        return '\n'.join(
            [' '.join([str(self.board[i * 8 + j]) for j in range(8)]).replace("None", '-') for i in range(8)]
        )

    @classmethod
    def create_standard_board(cls) -> 'Board':
        board = Board(WHITE, 0, 1)
        for piece_color, first_rank, pawn_rank in zip([WHITE, BLACK], ['1', '8'], ['2', '7']):
            # first rank pieces (rook, knight, bishop, queen, king, bishop, knight, rook)
            for file, piece_type in zip(FILES, (Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook)):
                board[file + first_rank] = piece_type(board, file + first_rank, piece_color, False)
            for file in FILES:
                board[file + pawn_rank] = Pawn(board, file + pawn_rank, piece_color, False)
        board.update()
        return board

    @classmethod
    def from_FEN(cls, FEN_record: str) -> 'Board':
        pass

    def to_FEN(self) -> str:
        """
        :return: the Forsyth-Edwards notation of the current board

        >>> print(Board.create_standard_board().to_FEN())
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        """
        FEN_components = []
        # pieces on chess board
        rows = []
        for i in range(8):
            consecutive_blanks_count = 0
            row = ''
            for j in range(8):
                tile = self.board[i * 8 + j]
                if tile is None:
                    consecutive_blanks_count += 1
                else:
                    # tile is a piece
                    if consecutive_blanks_count != 0:
                        row += str(consecutive_blanks_count)
                    consecutive_blanks_count = 0
                    row += str(tile)
            if consecutive_blanks_count != 0:
                row += str(consecutive_blanks_count)
            rows.append(row)
        FEN_components.append('/'.join(rows))
        # current player color (active color)
        FEN_components.append(self.active_color[0].lower())
        # castling availability
        FEN_components.append('KQkq')  # TODO this
        # en passant pawn
        FEN_components.append(
            str(algebraic_notation_to_index(self.en_passant_pawn.piece_position)
                - 8 * get_pawn_advance_direction(self.active_color)
                )
            if self.en_passant_pawn is not None
            else '-'
        )
        # half move clock
        FEN_components.append(str(self.half_move_clock))
        # full move number
        FEN_components.append(str(self.full_move_number))
        return ' '.join(FEN_components)

    def update(self):
        """
        find and set all active white and black pieces; set the players' kings
        """
        white_pieces, black_pieces = [], []
        for tile in self.board:
            if tile is not None:
                # tile is a piece (white or black)
                white_pieces.append(tile) if tile.color == WHITE else black_pieces.append(tile)
                if isinstance(tile, King):
                    if tile.color == WHITE:
                        self.white_king = tile
                    else:
                        self.black_king = tile
        self.white_pieces = white_pieces
        self.black_pieces = black_pieces


##########
# Pieces #
##########
"""Parent class"""


class Piece:
    PIECE_VALUE = 0

    def __init__(self, board: 'Board', piece_position: str, color: Union['WHITE', 'BLACK'], is_moved: bool):
        self.board = board
        self.piece_position = piece_position
        self.color = color
        self.moved = is_moved

    @classmethod
    def move(cls, move: 'Move'):
        return cls(move.destination_coordinate, move.moved_piece.color, True)

    def __repr__(self):
        raise NotImplementedError

    def get_piece_value(self):
        return self.PIECE_VALUE

    def calculate_piece_legal_moves(self) -> List['Move']:
        raise NotImplementedError

"""Children class"""


class King(Piece):
    def __repr__(self):
        return 'K' if self.color == WHITE else 'k'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


class Queen(Piece):
    def __repr__(self):
        return 'Q' if self.color == WHITE else 'q'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


class Rook(Piece):
    def __repr__(self):
        return 'R' if self.color == WHITE else 'r'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


class Bishop(Piece):
    def __repr__(self):
        return 'B' if self.color == WHITE else 'b'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


class Knight(Piece):
    def __repr__(self):
        return 'N' if self.color == WHITE else 'n'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


class Pawn(Piece):
    def __repr__(self):
        return 'P' if self.color == WHITE else 'p'

    def calculate_piece_legal_moves(self) -> List['Move']:
        pass


#########
# Moves #
#########
class Move:
    def __init__(self, board: 'Board', moved_piece: 'Piece', destination_coordinate: str):
        self.board = board
        self.moved_piece = moved_piece
        self.destination_coordinate = destination_coordinate
