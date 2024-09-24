# Name: Dylan Pellegrin
# Description: Main window loop

from tkinter import Tk
from game import Game

window = Tk()
window.title("Romm Adventure... Reloaded")
game = Game(window)
game.play()
window.mainloop()