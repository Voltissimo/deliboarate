from typing import Dict, List, Type, Union

import numpy as np

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
    """
    :param algebraic_notation:
    :return: the algebraic notation as a vector, with a8 as [0, 0] and h1 as [7, 7]

    >>> algebraic_notation_to_vector("a1")
    array([7, 0])
    >>> algebraic_notation_to_vector("h1")
    array([7, 7])
    >>> algebraic_notation_to_vector("a8")
    array([0, 0])
    """
    file, rank = algebraic_notation
    return np.array([7 - RANKS.index(rank), FILES.index(file)])


def vector_to_algebraic_notation(vector: np.ndarray) -> str:
    """
    :param vector: [i, j] with i = rows and j = cols
    :return: the corresponding algebraic notation

    >>> vector_to_algebraic_notation(np.array([0, 0]))
    'a8'
    >>> vector_to_algebraic_notation(np.array([7, 0]))
    'a1'
    >>> vector_to_algebraic_notation(np.array([7, 7]))
    'h1'
    """
    rank, file = vector
    return FILES[file] + RANKS[7 - rank]


def get_pawn_advance_direction(color: Union['WHITE', 'BLACK']) -> int:
    return -1 if color == WHITE else 1


def is_vector_coordinate_valid(vector: np.ndarray) -> bool:
    rank, file = vector
    return 0 <= rank < 8 and 0 <= file < 8


##########
# Player #
##########
class Player:
    def __init__(self,
                 board: 'Board',
                 color: Union['WHITE', 'BLACK'],
                 king: 'King',
                 active_pieces: List['Piece'],
                 king_side_castle: bool,
                 queen_side_castle: bool):
        self.board = board
        self.color: str = color
        self.active_pieces = active_pieces
        self.king = king
        self.king_side_castle_availability = king_side_castle
        self.queen_side_castle_availability = queen_side_castle
        self.moves = self.calculate_moves()

    def calculate_castling_moves(self) -> List['CastleMove']:
        castling_moves = []
        FIRST_RANK = "1" if self.color == WHITE else "8"
        if self.king_side_castle_availability:  # king and rook on first rank and not moved before
            if all([self.board[file + FIRST_RANK] is None for file in ("f", "g")]):
                for enemy_move in self.get_opponent().moves:
                    if enemy_move.destination_coordinate[0] in ("e", "f", "g") \
                            and enemy_move.destination_coordinate[1] == FIRST_RANK:
                        break
                else:
                    castling_moves.append(KingSideCastleMove(self.board, self.king))
        if self.queen_side_castle_availability:
            if all([self.board[file + FIRST_RANK] is None for file in ("b", "c", "d")]):
                for enemy_move in self.get_opponent().moves:
                    if enemy_move.destination_coordinate[0] in ("c", "d", "e") \
                            and enemy_move.destination_coordinate[1] == FIRST_RANK:
                        break
                else:
                    castling_moves.append(QueenSideCastleMove(self.board, self.king))
        return castling_moves

    def calculate_legal_moves(self) -> List['Move']:
        """
        :return: the moves this player is allowed to make

        >>> set([str(move) for move in Board.from_FEN(
        ...        "r1bqkbnr/pppp1ppp/8/8/3PP3/5n2/PPP2PPP/RNBQKB1R w KQkq - 1 4"
        ...    ).current_player.calculate_legal_moves()]) == {'gxf3', 'Qxf3', 'Ke2'}
        True
        >>> "O-O" in set([str(move) for move in Board.from_FEN(
        ...        "r1bqkb1r/ppp2ppp/3p1n2/8/3PP3/5P2/PPP2PBP/RNBQK2R w KQkq - 1 4"
        ...    ).current_player.calculate_legal_moves()])
        True
        >>> "O-O" in set([str(move) for move in Board.from_FEN(
        ...        "r1bqkb1r/pppp1ppp/5n2/8/3PP3/5PP/PPP2PB1/RNBQK2R w Qkq - 1 4"
        ...    ).current_player.calculate_legal_moves()])
        False
        >>> "O-O-O" in set([str(move) for move in Board.from_FEN(
        ...        "rnbqkbnr/8/pppppppp/8/PPPPPPPP/8/NQB5/R3KBNR w KQkq - 0 1"
        ...    ).current_player.calculate_legal_moves()])
        True
        """
        legal_moves = []
        for move in self.moves + self.calculate_castling_moves():
            transition_board = move.execute()
            if not transition_board.current_player.get_opponent().is_in_check():
                legal_moves.append(move)
        return legal_moves

    def is_in_check(self) -> bool:
        for enemy_move in self.get_opponent().moves:
            if enemy_move.destination_coordinate == self.king.piece_position:
                return True
        return False

    def is_king_escape_moves_available(self) -> bool:
        return bool(self.calculate_legal_moves())

    def is_in_stalemate(self) -> bool:
        return (not self.is_in_check()) and (not self.is_king_escape_moves_available())

    def is_in_checkmate(self) -> bool:
        return self.is_in_check() and (not self.is_king_escape_moves_available())

    def calculate_moves(self) -> List['Move']:
        moves = []
        for piece in self.active_pieces:
            moves += piece.calculate_piece_moves()
        return moves

    def get_opponent(self) -> 'Player':
        return self.board.black_player if self.color == WHITE else self.board.white_player


#########
# Board #
#########
class Board:
    def __init__(self,
                 active_color: Union['WHITE', 'BLACK'],
                 half_move_clock: int,
                 full_move_number: int,
                 en_passant_position=None):
        """
        Create an essentially empty board
        Notice: load_players() must be called again if any change is made to the board
        """
        self.board: List[Union[None, 'Piece']] = [None] * 64
        self.active_color: str = active_color
        self.half_move_clock = half_move_clock
        self.full_move_number = full_move_number
        self.en_passant_position: Union[None, str] = en_passant_position
        # these will be overridden by load_players()
        self.white_player, self.black_player = \
            Player(self, WHITE, King(self, 'a1', WHITE), [], False, False), \
            Player(self, BLACK, King(self, 'a1', BLACK), [], False, False)
        self.current_player: 'Player' = None

    def __getitem__(self, key: str) -> Union[None, 'Piece']:
        return self.board[algebraic_notation_to_index(key)]

    def __setitem__(self, key: str, value: Union[None, 'Piece']):
        self.board[algebraic_notation_to_index(key)] = value

    def __repr__(self) -> str:
        """
        :return: the current board in text format

        >>> Board.create_standard_board()
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
        """
        :return: board of the standard start

        >>> len(Board.create_standard_board().current_player.calculate_legal_moves())
        20
        """
        board = Board(WHITE, 0, 1)
        for piece_color, first_rank, pawn_rank in zip([WHITE, BLACK], ['1', '8'], ['2', '7']):
            # first rank pieces (rook, knight, bishop, queen, king, bishop, knight, rook)
            for file, piece_type in zip(FILES, (Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook)):
                if piece_type == King or piece_type == Rook:
                    board[file + first_rank] = piece_type(board, file + first_rank, piece_color)
                else:
                    board[file + first_rank] = piece_type(board, file + first_rank, piece_color)
            for file in FILES:
                board[file + pawn_rank] = Pawn(board, file + pawn_rank, piece_color)
        board.load_players(True, True, True, True)
        return board

    @classmethod
    def from_FEN(cls, FEN_record: str) -> 'Board':
        """
        :param FEN_record:
        :return: the board created from the FEN record

        >>> Board.from_FEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        r n b q k b n r
        p p p p p p p p
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        - - - - - - - -
        P P P P P P P P
        R N B Q K B N R
        """
        pieces, current_player, castling_availability_string, en_passant_position, half_move_clock, full_move_number = \
            FEN_record.split(' ')
        board = Board(
            WHITE if current_player == 'w' else BLACK,
            int(half_move_clock),
            int(full_move_number),
            en_passant_position=en_passant_position
        )
        for rank_count, rank in enumerate(pieces.split('/')):
            file_count = 0
            for file in rank:
                if str.isdigit(file):
                    file_count += int(file)
                else:
                    piece_type: str = file
                    piece_color = WHITE if file.upper() == file else BLACK
                    piece_position = vector_to_algebraic_notation(np.array([rank_count, file_count]))
                    board[piece_position] = {
                        'B': Bishop,
                        'N': Knight,
                        'P': Pawn,
                        'Q': Queen,
                        'K': King,
                        'R': Rook
                    }[piece_type.upper()](board, piece_position, piece_color)
                    file_count += 1
        board.load_players(
            'K' in castling_availability_string,
            'k' in castling_availability_string,
            'Q' in castling_availability_string,
            'q' in castling_availability_string
        )
        return board

    def to_FEN(self) -> str:
        """
        :return: the Forsyth-Edwards notation of the current board

        >>> Board.create_standard_board().to_FEN()
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
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
        castling_availability = ''
        for condition, string_to_append in zip(
                [self.white_player.king_side_castle_availability, self.white_player.queen_side_castle_availability,
                 self.black_player.king_side_castle_availability, self.black_player.queen_side_castle_availability],
                ['K', 'Q', 'k', 'q']
        ):
            if condition:
                castling_availability += string_to_append
        FEN_components.append(castling_availability if len(castling_availability) > 0 else '-')  # just to be sure
        # en passant pawn
        FEN_components.append(self.en_passant_position if self.en_passant_position is not None else '-')
        # half move clock
        FEN_components.append(str(self.half_move_clock))
        # full move number
        FEN_components.append(str(self.full_move_number))
        return ' '.join(FEN_components)

    def load_players(self,
                     white_king_side_castle: bool, white_queen_side_castle: bool,
                     black_king_side_castle: bool, black_queen_side_castle: bool):
        white_active_pieces, black_active_pieces = [], []
        white_king, black_king = None, None
        for tile in self.board:
            if tile is not None:
                piece: 'Piece' = tile
                if piece.color == WHITE:
                    white_active_pieces.append(piece)
                    if type(piece) == King:
                        white_king: 'King' = piece
                else:
                    black_active_pieces.append(piece)
                    if type(piece) == King:
                        black_king: 'King' = piece
        self.white_player = Player(self, WHITE,
                                   white_king, white_active_pieces, white_king_side_castle, white_queen_side_castle)
        self.black_player = Player(self, BLACK,
                                   black_king, black_active_pieces, black_king_side_castle, black_queen_side_castle)
        self.current_player = self.white_player if self.active_color == WHITE else self.black_player

    def create_move(self, source_square: str, destination_square: str) -> Dict[str, Union['Board', bool]]:
        """
        Create a move from the source and destination coordinates and execute it

        :param source_square: the square where the piece is located
        :param destination_square: the destination square of the move
        :return: a dictionary, {"success": bool, "board": Board}

        >>> Board.create_standard_board().create_move("d2", "d4")["board"]
        r n b q k b n r
        p p p p p p p p
        - - - - - - - -
        - - - - - - - -
        - - - P - - - -
        - - - - - - - -
        P P P - P P P P
        R N B Q K B N R
        >>> Board.create_standard_board().create_move("d2", "d4")["board"].en_passant_position
        'd3'
        >>> Board.create_standard_board().create_move("d2", "d4")["board"].create_move("d7", "d5")["board"]
        r n b q k b n r
        p p p - p p p p
        - - - - - - - -
        - - - p - - - -
        - - - P - - - -
        - - - - - - - -
        P P P - P P P P
        R N B Q K B N R
        >>> Board.create_standard_board().create_move("d2", "d1")["success"]
        False
        >>> Board.create_standard_board().create_move(
        ...        "e2", "e4"
        ...    )["board"].create_move(
        ...        "d7", "d5"
        ...    )["board"].create_move(
        ...        "e4", "d5"
        ...    )["board"].create_move(
        ...        "c7", "c5"
        ...    )["board"].create_move(
        ...        "d5", "c6"
        ...    )["success"]
        True
        """
        for move in self.current_player.calculate_legal_moves():
            if move.moved_piece.piece_position == source_square and move.destination_coordinate == destination_square:
                return {
                    "success": True,
                    "board": move.execute()
                }
        return {
            "success": False,
            "board": self.create_standard_board()
        }


##########
# Pieces #
##########


class Piece:
    PIECE_VALUE = 0

    def __init__(self, board: 'Board', piece_position: str, color: Union['WHITE', 'BLACK']):
        self.board = board
        self.piece_position = piece_position
        self.color = color

    def __eq__(self, other: 'Piece'):
        if type(self) != type(other):
            return False
        return self.color == other.color and self.piece_position == other.piece_position

    def __ne__(self, other: 'Piece'):
        return not self == other

    @classmethod
    def move(cls, move: 'Move', new_board: 'Board') -> 'Piece':
        return cls(new_board, move.destination_coordinate, move.moved_piece.color)

    def update_board(self, new_board: 'Board') -> 'Piece':
        return type(self)(new_board, self.piece_position, self.color)

    def __repr__(self):
        raise NotImplementedError

    def get_piece_value(self):
        return self.PIECE_VALUE

    def calculate_piece_moves(self) -> List['Move']:
        raise NotImplementedError  # whether or not the king is put into danger after the move is done in Player class

    def calculate_moves_through_unoccupied_squares(self,
                                                   move_vectors: List[np.ndarray],
                                                   decorator_class=None) -> List['Move']:
        moves = []
        position_vector = algebraic_notation_to_vector(self.piece_position)
        for move_vector in move_vectors:
            candidate_destination_vector = position_vector + move_vector
            if not is_vector_coordinate_valid(candidate_destination_vector):
                continue
            candidate_destination_square = vector_to_algebraic_notation(candidate_destination_vector)
            while self.board[candidate_destination_square] is None:
                candidate_destination_square = vector_to_algebraic_notation(candidate_destination_vector)
                if decorator_class is None:
                    moves.append(NormalMove(self.board, self, candidate_destination_square))
                else:
                    moves.append(decorator_class(NormalMove(self.board, self, candidate_destination_square)))
                candidate_destination_vector += move_vector
                if not is_vector_coordinate_valid(candidate_destination_vector):
                    break
                candidate_destination_square = vector_to_algebraic_notation(candidate_destination_vector)
            else:
                if not is_vector_coordinate_valid(candidate_destination_vector):
                    continue
                candidate_captured_piece: 'Piece' = self.board[candidate_destination_square]
                if candidate_captured_piece.color != self.color:
                    if decorator_class is None:
                        moves.append(
                            CaptureMove(self.board, self, candidate_destination_square, candidate_captured_piece)
                        )
                    else:
                        moves.append(
                            decorator_class(
                                CaptureMove(self.board, self, candidate_destination_square, candidate_captured_piece)
                            )
                        )
        return moves


class King(Piece):
    def __repr__(self):
        return 'K' if self.color == WHITE else 'k'

    def calculate_piece_moves(self) -> List['Move']:
        piece_position_vector = algebraic_notation_to_vector(self.piece_position)
        piece_moves = []
        for offset_vector in [np.array(vector_as_list) for vector_as_list in
                              ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]:
            candidate_destination_vector = piece_position_vector + offset_vector
            if not is_vector_coordinate_valid(candidate_destination_vector):
                continue
            candidate_destination = vector_to_algebraic_notation(candidate_destination_vector)
            if self.board[candidate_destination] is None:  # if the move leaves king in danger is checked after
                piece_moves.append(KingMove(NormalMove(self.board, self, candidate_destination)))
            else:
                candidate_captured_piece = self.board[candidate_destination]
                if candidate_captured_piece.color != self.color:
                    piece_moves.append(
                        KingMove(CaptureMove(self.board, self, candidate_destination, candidate_captured_piece))
                    )
        return piece_moves


class Queen(Piece):
    PIECE_VALUE = 9

    def __repr__(self):
        return 'Q' if self.color == WHITE else 'q'

    def calculate_piece_moves(self) -> List['Move']:
        return self.calculate_moves_through_unoccupied_squares([
            np.array(direction) for direction in
            ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))
        ])


class Rook(Piece):
    PIECE_VALUE = 5

    def __repr__(self):
        return 'R' if self.color == WHITE else 'r'

    def calculate_piece_moves(self) -> List['Move']:
        return self.calculate_moves_through_unoccupied_squares(
            [np.array(direction) for direction in
             ((1, 0), (0, -1), (-1, 0), (0, 1))],
            decorator_class=RookMove
        )


class Bishop(Piece):
    PIECE_VALUE = 3

    def __repr__(self):
        return 'B' if self.color == WHITE else 'b'

    def calculate_piece_moves(self) -> List['Move']:
        return self.calculate_moves_through_unoccupied_squares([
            np.array(direction) for direction in
            ((1, -1), (-1, -1), (-1, 1), (1, 1))
        ])


class Knight(Piece):
    PIECE_VALUE = 3

    def __repr__(self):
        return 'N' if self.color == WHITE else 'n'

    def calculate_piece_moves(self) -> List['Move']:
        piece_moves = []
        piece_position_vector = algebraic_notation_to_vector(self.piece_position)
        for move_vector in [np.array(move_vector_as_list) for move_vector_as_list
                            in ((2, 1), (2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2), (-2, 1), (-2, -1))]:
            candidate_destination_vector = piece_position_vector + move_vector
            if is_vector_coordinate_valid(candidate_destination_vector):
                candidate_destination = vector_to_algebraic_notation(candidate_destination_vector)
                if self.board[candidate_destination] is None:
                    piece_moves.append(NormalMove(self.board, self, candidate_destination))
                else:
                    candidate_captured_piece: 'Piece' = self.board[candidate_destination]
                    if candidate_captured_piece.color != self.color:
                        piece_moves.append(
                            CaptureMove(self.board, self, candidate_destination, candidate_captured_piece)
                        )
        return piece_moves


class Pawn(Piece):
    PIECE_VALUE = 1

    def __init__(self, board: 'Board', piece_position: str, color: Union['WHITE', 'BLACK']):
        super().__init__(board, piece_position, color)

    def __repr__(self):
        return 'P' if self.color == WHITE else 'p'

    def is_in_initial_rank(self, position: str) -> bool:
        return position[1] == ('2' if self.color == WHITE else '7')

    def is_in_promotion_rank(self, position: str) -> bool:
        return position[1] == ('8' if self.color == WHITE else '1')

    def calculate_piece_moves(self) -> List['Move']:
        piece_moves = []
        piece_position_vector = algebraic_notation_to_vector(self.piece_position)
        for move_vector in [np.array(move_vector_list) for move_vector_list in ((1, 0), (1, 1), (1, -1))]:
            candidate_destination_vector = piece_position_vector + move_vector * get_pawn_advance_direction(self.color)
            if not is_vector_coordinate_valid(candidate_destination_vector):
                continue
            candidate_destination_square = vector_to_algebraic_notation(candidate_destination_vector)
            if all(move_vector == np.array([1, 0])):  # normal move
                if self.board[candidate_destination_square] is None:
                    if self.is_in_promotion_rank(candidate_destination_square):  # pawn promotion
                        piece_moves.append(PawnPromotionMove(PawnMove(self.board, self, candidate_destination_square)))
                    else:
                        piece_moves.append(PawnMove(self.board, self, candidate_destination_square))
                    if self.is_in_initial_rank(self.piece_position):  # pawn jump
                        pawn_jump_candidate_square = vector_to_algebraic_notation(
                            candidate_destination_vector
                            + np.array([1, 0]) * get_pawn_advance_direction(self.color)
                        )
                        if self.board[pawn_jump_candidate_square] is None:
                            piece_moves.append(PawnJumpMove(self.board, self, pawn_jump_candidate_square))
            else:  # pawn capture move
                if self.board[candidate_destination_square] is not None:
                    candidate_captured_piece: 'Piece' = self.board[candidate_destination_square]
                    if candidate_captured_piece.color != self.color:
                        pawn_capture_move = PawnCaptureMove(
                            self.board, self, candidate_destination_square, candidate_captured_piece
                        )
                        if self.is_in_promotion_rank(candidate_destination_square):  # pawn promotion
                            pawn_capture_move = PawnPromotionMove(pawn_capture_move)
                        piece_moves.append(pawn_capture_move)
                elif candidate_destination_square == self.board.en_passant_position:  # en passant move
                    en_passant_pawn: 'Pawn' = self.board[
                        vector_to_algebraic_notation(
                            candidate_destination_vector - get_pawn_advance_direction(self.color) * np.array([1, 0])
                        )
                    ]
                    piece_moves.append(
                        PawnCaptureMove(self.board, self, candidate_destination_square, en_passant_pawn)
                    )
        return piece_moves


#########
# Moves #
#########


class Move:
    def __init__(self, board: 'Board', moved_piece: 'Piece', destination_coordinate: str):
        self.board = board
        self.moved_piece = moved_piece
        self.original_coordinate = moved_piece.piece_position
        self.destination_coordinate = destination_coordinate

    def __str__(self):
        raise NotImplementedError

    def execute(self) -> 'Board':
        raise NotImplementedError

    def calculate_move_suffix(self) -> str:
        suffix = ""
        board = self.execute()
        if board.current_player.is_in_checkmate():
            suffix = "#"
        elif board.current_player.is_in_check():
            suffix = "+"
        return suffix


class NormalMove(Move):
    def execute(self) -> 'Board':
        board = Board(self.board.current_player.get_opponent().color,
                      self.board.half_move_clock + 1,
                      (self.board.full_move_number + 1)
                      if self.board.current_player.color == BLACK else self.board.full_move_number)
        for piece in self.board.current_player.get_opponent().active_pieces:
            board[piece.piece_position] = piece.update_board(board)
        for piece in self.board.current_player.active_pieces:
            if piece == self.moved_piece:
                board[self.destination_coordinate] = self.moved_piece.move(self, board)
            else:
                board[piece.piece_position] = piece.update_board(board)
        board.load_players(
            self.board.white_player.king_side_castle_availability,
            self.board.white_player.queen_side_castle_availability,
            self.board.black_player.king_side_castle_availability,
            self.board.black_player.queen_side_castle_availability
        )
        return board

    def __str__(self) -> str:
        return str(self.moved_piece).upper() + self.destination_coordinate + self.calculate_move_suffix()


class CaptureMove(Move):
    def __init__(self, board: 'Board', moved_piece: 'Piece', destination_coordinate: str, captured_piece: 'Piece'):
        super().__init__(board, moved_piece, destination_coordinate)
        self.captured_piece = captured_piece

    def execute(self) -> 'Board':
        board = Board(self.board.current_player.get_opponent().color,
                      0,
                      (self.board.full_move_number + 1)
                      if self.board.current_player.color == BLACK else self.board.full_move_number)
        for piece in self.board.current_player.get_opponent().active_pieces:
            if piece != self.captured_piece:
                board[piece.piece_position] = piece.update_board(board)
        for piece in self.board.current_player.active_pieces:
            if piece == self.moved_piece:
                board[self.destination_coordinate] = self.moved_piece.move(self, board)
            else:
                board[piece.piece_position] = piece.update_board(board)
        board.load_players(
            self.board.white_player.king_side_castle_availability,
            self.board.white_player.queen_side_castle_availability,
            self.board.black_player.king_side_castle_availability,
            self.board.black_player.queen_side_castle_availability
        )
        return board

    def __str__(self):
        return str(self.moved_piece).upper() + 'x' + self.destination_coordinate + self.calculate_move_suffix()


class PawnMove(NormalMove):
    def execute(self) -> 'Board':
        board = super().execute()
        board.half_move_clock = 0
        return board

    def __str__(self):
        return self.destination_coordinate


class PawnJumpMove(PawnMove):
    def execute(self) -> 'Board':
        board = super().execute()
        board.half_move_clock = 0
        en_passant_square = vector_to_algebraic_notation(
            algebraic_notation_to_vector(self.destination_coordinate)
            - get_pawn_advance_direction(self.moved_piece.color) * np.array([1, 0])
        )
        board.en_passant_position = en_passant_square
        board.load_players(
            self.board.white_player.king_side_castle_availability,
            self.board.white_player.queen_side_castle_availability,
            self.board.black_player.king_side_castle_availability,
            self.board.black_player.queen_side_castle_availability
        )
        return board


class PawnCaptureMove(CaptureMove):
    def __str__(self):
        return self.moved_piece.piece_position[0] + 'x' + self.destination_coordinate + self.calculate_move_suffix()


class PawnPromotionMove(Move):
    def __init__(self,
                 decorated_move: Union['PawnMove', 'PawnCaptureMove'],
                 promotion_piece_type: Union[Type['Queen'], Type['Knight'], Type['Rook'], Type['Bishop']] = Queen):
        super().__init__(decorated_move.board, decorated_move.moved_piece, decorated_move.destination_coordinate)
        self.decorated_move = decorated_move
        self.promotion_piece_type = promotion_piece_type

    def execute(self) -> 'Board':
        board = self.decorated_move.execute()
        board[self.destination_coordinate] = \
            self.promotion_piece_type(board, self.destination_coordinate, self.moved_piece.color)
        board.load_players(
            self.board.white_player.king_side_castle_availability,
            self.board.white_player.queen_side_castle_availability,
            self.board.black_player.king_side_castle_availability,
            self.board.black_player.queen_side_castle_availability
        )
        return board

    def __str__(self):
        return str(self.decorated_move).replace("+", "").replace("#", "") + self.calculate_move_suffix()


class KingMove(Move):
    def __init__(self, decorated_move: Union['NormalMove', 'CaptureMove']):
        super().__init__(decorated_move.board, decorated_move.moved_piece, decorated_move.destination_coordinate)
        self.decorated_move = decorated_move

    def execute(self) -> 'Board':
        board = self.decorated_move.execute()
        board.load_players(
            False if self.moved_piece.color == WHITE else self.board.white_player.king_side_castle_availability,
            False if self.moved_piece.color == WHITE else self.board.white_player.queen_side_castle_availability,
            False if self.moved_piece.color == BLACK else self.board.black_player.king_side_castle_availability,
            False if self.moved_piece.color == BLACK else self.board.black_player.queen_side_castle_availability
        )
        return board

    def __str__(self):
        return str(self.decorated_move)


class RookMove(Move):
    def __init__(self, decorated_move: Union['NormalMove', 'CaptureMove']):
        super().__init__(decorated_move.board, decorated_move.moved_piece, decorated_move.destination_coordinate)
        self.decorated_move = decorated_move

    def execute(self) -> 'Board':
        board = self.decorated_move.execute()
        if self.moved_piece.color == WHITE:
            board.load_players(
                False
                if self.moved_piece.piece_position == "h1"
                else self.board.white_player.king_side_castle_availability,
                False
                if self.moved_piece.piece_position == "a1"
                else self.board.white_player.queen_side_castle_availability,
                self.board.black_player.king_side_castle_availability,
                self.board.black_player.queen_side_castle_availability
            )
        else:
            board.load_players(
                self.board.white_player.king_side_castle_availability,
                self.board.white_player.queen_side_castle_availability,
                False
                if self.moved_piece.piece_position == "h8"
                else self.board.black_player.king_side_castle_availability,
                False
                if self.moved_piece.piece_position == "a8"
                else self.board.black_player.queen_side_castle_availability
            )
        return board

    def __str__(self):
        return str(self.decorated_move)


class CastleMove(Move):
    def __init__(self, board: 'Board',
                 king: 'King', king_destination_coordinate: str,
                 rook: 'Rook', rook_destination_coordinate: str
                 ):
        super().__init__(board, king, king_destination_coordinate)
        self.king = king
        self.rook = rook
        self.rook_destination_coordinate = rook_destination_coordinate

    def execute(self) -> 'Board':
        board = Board(self.board.current_player.get_opponent().color,
                      self.board.half_move_clock + 1,
                      (self.board.full_move_number + 1)
                      if self.board.current_player.color == BLACK else self.board.full_move_number)
        for piece in self.board.current_player.get_opponent().active_pieces:
            board[piece.piece_position] = piece.update_board(board)
        for piece in self.board.current_player.active_pieces:
            if piece == self.king:
                board[self.destination_coordinate] = self.king.move(self, board)
            elif piece == self.rook:
                board[self.rook_destination_coordinate] = Rook(board, self.rook_destination_coordinate, self.rook.color)
            else:
                board[piece.piece_position] = piece.update_board(board)
        board.load_players(
            False if self.moved_piece.color == WHITE else self.board.white_player.king_side_castle_availability,
            False if self.moved_piece.color == WHITE else self.board.white_player.queen_side_castle_availability,
            False if self.moved_piece.color == BLACK else self.board.black_player.king_side_castle_availability,
            False if self.moved_piece.color == BLACK else self.board.black_player.queen_side_castle_availability
        )
        return board

    def __str__(self):
        raise NotImplementedError


class KingSideCastleMove(CastleMove):
    def __init__(self, board: 'Board', king: 'King'):
        king_destination_coordinate = "g1" if king.color == WHITE else "g8"
        rook_coordinate = "h1" if king.color == WHITE else "h8"
        rook: 'Rook' = board[rook_coordinate]
        assert type(rook) == Rook
        rook_destination_coordinate = "f1" if king.color == WHITE else "f8"
        super().__init__(board, king, king_destination_coordinate, rook, rook_destination_coordinate)

    def __str__(self):
        return "O-O"


class QueenSideCastleMove(CastleMove):
    def __init__(self, board: 'Board', king: 'King'):
        king_destination_coordinate = "c1" if king.color == WHITE else "c8"
        rook_coordinate = "h1" if king.color == WHITE else "h8"
        rook: 'Rook' = board[rook_coordinate]
        assert type(rook) == Rook
        rook_destination_coordinate = "d1" if king.color == WHITE else "d8"
        super().__init__(board, king, king_destination_coordinate, rook, rook_destination_coordinate)

    def __str__(self):
        return "O-O-O"


if __name__ == '__main__':
    pass
