import numpy as np
from pieces import Piece
from typing import Union, Dict


class Board:
    def __getitem__(self, key) -> Union[Piece, None]:
        return self.board[self.to_index(key)]

    def __setitem__(self, key, value):
        self.board[self.to_index(key)] = value

    def __iter__(self) -> iter:
        """Iterate through the tiles of the chess board"""
        return (self.board[i] for i in range(64))

    def __repr__(self):
        return self.to_FEN()

    def __init__(self, half_move_clock: int, full_move_count: int, en_passant_pawn=None):
        self.board: Dict[int: Union[Piece, None]] = {i: None for i in range(64)}

        self.half_move_clock: int = half_move_clock
        self.full_move_clock: int = full_move_count

        self.en_passant_pawn: Piece = en_passant_pawn

    @classmethod
    def create_standard_board(cls) -> 'Board':
        board = Board(0, 1)
        # TODO set the pieces
        return board

    @classmethod
    def from_FEN(cls, FEN_string: str) -> 'Board':
        pass

    def to_FEN(self) -> str:
        """
        :return: the Forsyth-Edwards notation of the current board

        >>> Board.create_standard_board().to_FEN()
        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
        """
        pass

    @staticmethod
    def to_index(key: Union[str, int, tuple, list, np.ndarray]) -> int:
        """

        :param key:
        :return:
        """
        if isinstance(key, str):
            pass
        if isinstance(key, int):
            return key
        if isinstance(key, tuple) or isinstance(key, list) or isinstance(key, np.ndarray):
            return key[0] * 8 + key[1]
