import unittest
import numpy as np
from connect4 import Connect4, Player


class TestConnect4(unittest.TestCase):
    """Test suite for Connect Four game environment."""
    
    def test_initialization(self):
        """Test game initialization."""
        game = Connect4()
        assert game.board.shape == (6, 7)
        assert np.all(game.board == 0)
        assert game.current_player == Player.PLAYER1
        assert game.winner is None
        assert game.game_over is False
        assert game.moves_count == 0
    
    def test_reset(self):
        """Test game reset functionality."""
        game = Connect4()
        # Make some moves
        game.make_move(0)
        game.make_move(1)
        game.make_move(2)
        
        # Reset
        board = game.reset()
        assert np.all(board == 0)
        assert np.all(game.board == 0)
        assert game.current_player == Player.PLAYER1
        assert game.winner is None
        assert game.game_over is False
        assert game.moves_count == 0
    
    def test_valid_moves(self):
        """Test getting valid moves."""
        game = Connect4()
        # Initially all columns should be valid
        valid_moves = game.get_valid_moves()
        assert valid_moves == list(range(7))
        
        # Fill a column
        for _ in range(6):
            game.make_move(0)
        
        valid_moves = game.get_valid_moves()
        assert 0 not in valid_moves
        assert valid_moves == list(range(1, 7))
    
    def test_is_valid_move(self):
        """Test move validation."""
        game = Connect4()
        
        # Valid moves
        for col in range(7):
            assert game.is_valid_move(col)
        
        # Invalid moves (out of bounds)
        assert not game.is_valid_move(-1)
        assert not game.is_valid_move(7)
        
        # Fill a column
        for _ in range(6):
            game.make_move(0)
        
        # Column 0 should now be invalid
        assert not game.is_valid_move(0)
        assert game.is_valid_move(1)
    
    def test_make_move(self):
        """Test making moves."""
        game = Connect4()
        
        # Test basic move
        success, winner = game.make_move(3)
        assert success is True
        assert winner is None
        assert game.board[5, 3] == Player.PLAYER1.value
        assert game.current_player == Player.PLAYER2
        
        # Test stacking pieces
        success, winner = game.make_move(3)
        assert success is True
        assert winner is None
        assert game.board[4, 3] == Player.PLAYER2.value
        assert game.current_player == Player.PLAYER1
        
        # Test invalid move
        success, winner = game.make_move(-1)
        assert success is False
        assert winner is None
        assert game.current_player == Player.PLAYER1  # Player shouldn't change
    
    def test_horizontal_win(self):
        """Test horizontal win detection."""
        game = Connect4()
        
        # Create a horizontal win for Player 1
        # P1 moves: 0, 1, 2, 3
        # P2 moves: 0, 1, 2
        moves = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (3, 1)]
        
        for col, player in moves[:-1]:
            success, winner = game.make_move(col)
            assert success is True
            assert winner is None
        
        # Winning move
        success, winner = game.make_move(3)
        assert success is True
        assert winner == Player.PLAYER1
        assert game.game_over is True
    
    def test_vertical_win(self):
        """Test vertical win detection."""
        game = Connect4()
        
        # Create a vertical win for Player 1
        # P1 moves: 0, 0, 0, 0
        # P2 moves: 1, 1, 1
        moves = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2), (0, 1)]
        
        winner = None
        for i, (col, expected_player) in enumerate(moves):
            success, winner = game.make_move(col)
            assert success is True
            if i < len(moves) - 1:
                assert winner is None
        
        assert winner == Player.PLAYER1
        assert game.game_over is True
    
    def test_diagonal_win(self):
        """Test diagonal win detection (bottom-left to top-right)."""
        game = Connect4()
        
        # Create a diagonal win
        # Build up the board to create a diagonal
        setup_moves = [
            0,  # P1
            1,  # P2
            1,  # P1
            2,  # P2
            2,  # P1
            3,  # P2
            2,  # P1
            3,  # P2
            3,  # P1
            6,  # P2 (distraction)
            3,  # P1 (winning move)
        ]
        
        for i, col in enumerate(setup_moves[:-1]):
            success, winner = game.make_move(col)
            assert success is True
            assert winner is None
        
        # Winning move
        success, winner = game.make_move(3)
        assert success is True
        assert winner == Player.PLAYER1
        assert game.game_over is True
    
    def test_anti_diagonal_win(self):
        """Test anti-diagonal win detection (top-left to bottom-right)."""
        game = Connect4()
        
        # Create an anti-diagonal win
        setup_moves = [
            6,  # P1
            5,  # P2
            5,  # P1
            4,  # P2
            4,  # P1
            3,  # P2
            4,  # P1
            3,  # P2
            3,  # P1
            0,  # P2 (distraction)
            3,  # P1 (winning move)
        ]
        
        for i, col in enumerate(setup_moves[:-1]):
            success, winner = game.make_move(col)
            assert success is True
            assert winner is None
        
        # Winning move
        success, winner = game.make_move(3)
        assert success is True
        assert winner == Player.PLAYER1
        assert game.game_over is True
    
    def test_draw(self):
        """Test draw detection when board is full."""
        game = Connect4()
        
        # Manually set up a board that's almost full with no winner
        # Pattern specifically designed to avoid any 4-in-a-row
        game.board = np.array([
            [1, 2, 1, 1, 2, 1, 0],  # Top row (one empty space)
            [2, 1, 2, 2, 1, 2, 2],
            [1, 2, 1, 1, 2, 1, 1],
            [2, 1, 2, 2, 1, 2, 2],
            [1, 2, 1, 1, 2, 1, 1],
            [2, 1, 2, 2, 1, 2, 2],  # Bottom row
        ])
        game.moves_count = 41
        game.current_player = Player.PLAYER1
        
        # Verify no winner exists
        assert game.winner is None
        assert not game.game_over
        
        # Make the final move
        success, winner = game.make_move(6)
        assert success is True
        assert winner is None
        assert game.game_over is True
        assert game.moves_count == 42
    
    def test_game_over_no_more_moves(self):
        """Test that no moves can be made after game is over."""
        game = Connect4()
        
        # Create a quick win
        for _ in range(3):
            game.make_move(0)  # P1
            game.make_move(1)  # P2
        game.make_move(0)  # P1 wins
        
        assert game.game_over is True
        
        # Try to make another move
        success, winner = game.make_move(2)
        assert success is False
        assert winner == Player.PLAYER1
    
    def test_render(self):
        """Test board rendering."""
        game = Connect4()
        
        # Test initial render
        output = game.render()
        assert "0 1 2 3 4 5 6" in output
        assert ". . . . . . ." in output
        assert "Current player: X" in output
        
        # Make some moves and test render
        game.make_move(3)
        game.make_move(3)
        output = game.render()
        assert "X" in output
        assert "O" in output
        
        # Test render after game over
        for _ in range(3):
            game.make_move(0)
            game.make_move(1)
        game.make_move(0)  # P1 wins
        
        output = game.render()
        assert "Game Over! X wins!" in output
    
    def test_get_state(self):
        """Test getting board state."""
        game = Connect4()
        
        # Initial state
        state = game.get_state()
        assert np.all(state == 0)
        
        # Make moves and check state
        game.make_move(0)
        game.make_move(1)
        state = game.get_state()
        assert state[5, 0] == Player.PLAYER1.value
        assert state[5, 1] == Player.PLAYER2.value
        
        # Ensure it's a copy
        state[0, 0] = 99
        assert game.board[0, 0] != 99
    
    def test_column_full(self):
        """Test behavior when a column is full."""
        game = Connect4()
        
        # Fill column 0
        for i in range(6):
            success, _ = game.make_move(0)
            assert success is True
        
        # Try to add to full column
        success, _ = game.make_move(0)
        assert success is False
        
        # Other columns should still work
        success, _ = game.make_move(1)
        assert success is True
    
    def test_alternating_players(self):
        """Test that players alternate correctly."""
        game = Connect4()
        
        assert game.current_player == Player.PLAYER1
        
        game.make_move(0)
        assert game.current_player == Player.PLAYER2
        
        game.make_move(1)
        assert game.current_player == Player.PLAYER1
        
        # Invalid move shouldn't change player
        game.make_move(-1)
        assert game.current_player == Player.PLAYER1
        
        game.make_move(2)
        assert game.current_player == Player.PLAYER2


if __name__ == "__main__":
    unittest.main(verbosity=2)