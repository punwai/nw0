from connect4 import Connect4


def play_demo():
    """Demonstrate the Connect Four game with a simple interactive session."""
    game = Connect4()
    
    print("Welcome to Connect Four!")
    print("=" * 30)
    print()
    
    # Show initial board
    print(game)
    print()
    
    # Demo moves
    demo_moves = [
        (3, "Player 1 drops in column 3"),
        (3, "Player 2 drops in column 3"),
        (4, "Player 1 drops in column 4"),
        (4, "Player 2 drops in column 4"),
        (5, "Player 1 drops in column 5"),
        (5, "Player 2 drops in column 5"),
        (6, "Player 1 drops in column 6 for the win!"),
    ]
    
    for col, description in demo_moves:
        print(f"{description}")
        success, winner = game.make_move(col)
        
        if success:
            print(game)
            if winner:
                print(f"\n🎉 Game Over! Player {winner.value} wins!")
                break
            print()
        else:
            print("Invalid move!")
    
    print("\n" + "=" * 30)
    print("Demo completed!")


def interactive_play():
    """Play an interactive game of Connect Four."""
    game = Connect4()
    
    print("Welcome to Connect Four!")
    print("Players take turns dropping pieces into columns.")
    print("First to connect 4 pieces in a row wins!")
    print("=" * 40)
    
    while not game.game_over:
        print()
        print(game)
        print()
        
        # Get player input
        player_symbol = 'X' if game.current_player.value == 1 else 'O'
        
        try:
            col = int(input(f"Player {player_symbol}, enter column (0-6): "))
            success, winner = game.make_move(col)
            
            if not success:
                print("Invalid move! Try again.")
                continue
                
        except (ValueError, KeyboardInterrupt):
            print("\nGame interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if game.game_over:
        print()
        print(game)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_play()
    else:
        play_demo()