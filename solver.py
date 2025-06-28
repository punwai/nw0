import numpy as np
from typing import Tuple, Optional
from connect4 import Connect4, Player


class Connect4Solver:
    def __init__(self, max_depth: int = 3):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    
    def get_best_move(self, game: Connect4) -> int | None:
        self.nodes_evaluated = 0
        _, best_col = self._minimax(
            game, 
            self.max_depth, 
            -float('inf'), 
            float('inf'), 
            True,
            game.current_player
        )
        return best_col
    
    def _minimax(
        self, 
        game: Connect4, 
        depth: int, 
        alpha: float, 
        beta: float, 
        maximizing: bool,
        original_player: Player
    ) -> Tuple[float, Optional[int]]:
        self.nodes_evaluated += 1
        
        if depth == 0 or game.game_over:
            return self._evaluate_position(game, original_player), None
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0, None
        
        best_col = valid_moves[0]
        
        if maximizing:
            max_eval = -float('inf')
            for col in valid_moves:
                game_copy = self._copy_game(game)
                game_copy.make_move(col)
                
                eval_score, _ = self._minimax(
                    game_copy, 
                    depth - 1, 
                    alpha, 
                    beta, 
                    False,
                    original_player
                )
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_col
        else:
            min_eval = float('inf')
            for col in valid_moves:
                game_copy = self._copy_game(game)
                game_copy.make_move(col)
                
                eval_score, _ = self._minimax(
                    game_copy, 
                    depth - 1, 
                    alpha, 
                    beta, 
                    True,
                    original_player
                )
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_col
    
    def _copy_game(self, game: Connect4) -> Connect4:
        new_game = Connect4()
        new_game.board = game.board.copy()
        new_game.current_player = game.current_player
        new_game.winner = game.winner
        new_game.game_over = game.game_over
        new_game.moves_count = game.moves_count
        return new_game
    
    def _evaluate_position(self, game: Connect4, player: Player) -> float:
        if game.game_over:
            if game.winner == player:
                return 10000
            elif game.winner is not None:
                return -10000
            else:
                return 0
        
        opponent = Player.PLAYER2 if player == Player.PLAYER1 else Player.PLAYER1
        score = 0
        
        score += self._evaluate_center_column(game.board, player.value, opponent.value) * 3
        
        for row in range(Connect4.ROWS):
            for col in range(Connect4.COLS):
                score += self._evaluate_window(game, row, col, 0, 1, player.value, opponent.value)
                score += self._evaluate_window(game, row, col, 1, 0, player.value, opponent.value)
                score += self._evaluate_window(game, row, col, 1, 1, player.value, opponent.value)
                score += self._evaluate_window(game, row, col, 1, -1, player.value, opponent.value)
        
        return score
    
    def _evaluate_center_column(self, board: np.ndarray, player: int, opponent: int) -> int:
        center_col = Connect4.COLS // 2
        center_count = 0
        for row in range(Connect4.ROWS):
            if board[row, center_col] == player:
                center_count += 1
        return center_count
    
    def _evaluate_window(
        self, 
        game: Connect4, 
        row: int, 
        col: int, 
        delta_row: int, 
        delta_col: int, 
        player: int, 
        opponent: int
    ) -> int:
        window = []
        for i in range(4):
            r = row + i * delta_row
            c = col + i * delta_col
            if 0 <= r < Connect4.ROWS and 0 <= c < Connect4.COLS:
                window.append(game.board[r, c])
            else:
                return 0
        
        return self._score_window(window, player, opponent)
    
    def _score_window(self, window: list, player: int, opponent: int) -> int:
        score = 0
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(Player.EMPTY.value)
        
        if player_count == 4:
            score += 100
        elif player_count == 3 and empty_count == 1:
            score += 5
        elif player_count == 2 and empty_count == 2:
            score += 2
        
        if opponent_count == 3 and empty_count == 1:
            score -= 4
        
        return score

if __name__ == "__main__":
    game = Connect4()
    solver = Connect4Solver()
    print("1...")
    print(solver.get_best_move(game))
    print("2...")