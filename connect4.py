import uuid
import numpy as np
from typing import Optional, Tuple, List
from enum import Enum


class Player(Enum):
    EMPTY = 0
    PLAYER1 = 1
    PLAYER2 = 2


class Connect4:
    """Connect Four game environment."""
    
    ROWS = 6
    COLS = 7
    CONNECT = 4
    
    def __init__(self):
        """Initialize the Connect Four game."""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = Player.PLAYER1
        self.winner = None
        self.id = str(uuid.uuid4())
        self.game_over = False
        self.moves_count = 0
    
    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        self.current_player = Player.PLAYER1
        self.winner = None
        self.game_over = False
        self.moves_count = 0
        return self.board.copy()
    
    def get_valid_moves(self) -> List[int]:
        """Get list of valid column indices where a piece can be dropped."""
        return [col for col in range(self.COLS) if self.board[0, col] == Player.EMPTY.value]
    
    def is_valid_move(self, col: int) -> bool:
        """Check if a move is valid."""
        if col < 0 or col >= self.COLS:
            return False
        return self.board[0, col] == Player.EMPTY.value
    
    def make_move(self, col: int) -> Tuple[bool, Optional[Player]]:
        """
        Make a move in the specified column.
        
        Args:
            col: Column index (0-6)
            
        Returns:
            Tuple of (move_successful, winner)
        """
        if self.game_over:
            return False, self.winner
        
        if not self.is_valid_move(col):
            return False, None
        
        # Find the lowest empty row in the column
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row, col] == Player.EMPTY.value:
                self.board[row, col] = self.current_player.value
                self.moves_count += 1
                
                # Check for winner
                if self._check_winner(row, col):
                    self.winner = self.current_player
                    self.game_over = True
                    return True, self.winner
                
                # Check for draw
                if self.moves_count == self.ROWS * self.COLS:
                    self.game_over = True
                    return True, None
                
                # Switch player
                self.current_player = Player.PLAYER2 if self.current_player == Player.PLAYER1 else Player.PLAYER1
                return True, None
        
        return False, None
    
    def _check_winner(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        
        # Check horizontal
        if self._check_direction(row, col, 0, 1, player):
            return True
        
        # Check vertical
        if self._check_direction(row, col, 1, 0, player):
            return True
        
        # Check diagonal (top-left to bottom-right)
        if self._check_direction(row, col, 1, 1, player):
            return True
        
        # Check anti-diagonal (top-right to bottom-left)
        if self._check_direction(row, col, 1, -1, player):
            return True
        
        return False
    
    def _check_direction(self, row: int, col: int, row_dir: int, col_dir: int, player: int) -> bool:
        """Check if there are 4 in a row in a specific direction."""
        count = 1
        
        # Check forward direction
        r, c = row + row_dir, col + col_dir
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
            count += 1
            r += row_dir
            c += col_dir
        
        # Check backward direction
        r, c = row - row_dir, col - col_dir
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
            count += 1
            r -= row_dir
            c -= col_dir
        
        return count >= self.CONNECT
    
    def render(self) -> str:
        """Create a string representation of the board."""
        symbols = {
            Player.EMPTY.value: '.',
            Player.PLAYER1.value: 'X',
            Player.PLAYER2.value: 'O'
        }
        
        # Build the board representation
        lines = []
        for row in range(self.ROWS):
            row_str = " ".join(symbols[self.board[row, col]] for col in range(self.COLS))
            lines.append(row_str)
        lines.append("")
        
        return "\n".join(lines)
    
    def get_state(self) -> np.ndarray:
        """Get a copy of the current board state."""
        return self.board.copy()
    
    def __str__(self) -> str:
        """String representation of the game."""
        return self.render()