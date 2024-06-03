from training_backend.game_controller import GameController

game = GameController(debug=False)
game.startGame()

print(game.pacman.position.asTuple())