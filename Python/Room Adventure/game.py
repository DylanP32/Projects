# Description: Game loop and GUI constructor

from tkinter import (
    # Widgets 
    Frame, Label, Text, PhotoImage, Entry,

    # Constants
    X, Y, BOTH,
    BOTTOM, RIGHT, LEFT,
    DISABLED, NORMAL, END,

    # Type Hint Stuff
    Tk 
)
from room import Room
import os
from item import Item

class Game(Frame):

    # some constants
    EXIT_ACTION = ["quit", "exit", "bye", "adios"]
    WIDTH = 800
    HEIGHT = 600

    # statuses
    class Status:
        DEFAULT = "I don't understand. Try verb noun. Valid verbs are go, look, take, use."
        DEAD = "You fell out of the window."
        WIN = "You found all the items"
        BAD_EXIT = "Invalid exit."
        ROOM_CHANGED = "Room changed."
        GRABBED = "Item grabbed."
        BAD_GRABBABLE = "I can't grab that."
        BAD_ITEM = "I don't see that."
        KEY_USED = "You unlocked the closet and went inside."
        EGGS_USED = "You broke the tv. That was rude."
        VACUUM_USED = "The rug has been vacuumed. While vacuuming you saw a ring in the rug."
        BAD_USEABLE = "I can't use that."
        NOT_HERE = "I can't use that here. Maybe in a different room."
    
    def __init__(self, parent: Tk):
        self.inventory = []
        Frame.__init__(self, parent)
        self.pack(fill = BOTH, expand = 1) #geometry management for Game instances
    
    def setup_game(self) -> None:
        
        # create rooms
        r1 = Room("Room 1", os.path.join("images", "room1.gif"))
        r2 = Room("Room 2", os.path.join("images", "room2.gif"))
        r3 = Room("Room 3", os.path.join("images", "room3.gif"))
        r3c = Room("Room 3 closet", os.path.join("images", "room3.gif"))
        r4 = Room("Room 4", os.path.join("images", "room4.gif"))
        r5 = Room("Room 5", os.path.join("images", "room5.gif"))
        r6 = Room("Room 6", os.path.join("images", "room6.gif"))
        r7 = Room("Room 7", os.path.join("images", "room7.gif"))
        r8 = Room("Room 8", os.path.join("images", "room8.gif"))
        
        
        # create exits
        r1.add_exit("east",r2)
        r1.add_exit("south",r3)
        r1.add_exit("up",r5)
        
        r2.add_exit("west",r1)
        r2.add_exit("south",r4)
        r2.add_exit("up",r6)
        
        r3.add_exit("east",r4)
        r3.add_exit("north",r1)
        r3.add_exit("up",r7)
        r3.add_exit("", r3c)
        r3c.add_exit("back", r3)
        
        r4.add_exit("north",r2)
        r4.add_exit("west",r3)
        r4.add_exit("up",r8)
        
        r5.add_exit("east",r6)
        r5.add_exit("south",r7)
        r5.add_exit("down",r1)
        r5.add_exit("window", None)
        
        r6.add_exit("west",r5)
        r6.add_exit("south",r8)
        r6.add_exit("down",r2)
        
        r7.add_exit("north",r5)
        r7.add_exit("east",r8)
        r7.add_exit("down",r3)
        
        r8.add_exit("north",r6)
        r8.add_exit("west",r7)
        r8.add_exit("down",r4)
        
        # add items
        r1.add_item(Item("chair", "It is made of wicker and no one\n  is sitting on it.", False, False))
        r1.add_item(Item("table", "It is made of oak. A golden\n  key rests on it.", False, False))
        
        r2.add_item(Item("rug", "It is nice and Indian.\n  It also needs to be vacuumed.", False, False))
        r2.add_item(Item("fireplace", "It is full of ashes.", False, False))
        
        r3.add_item(Item("bookshelves", "They are empty. Go figure.", False, False))
        r3.add_item(Item("statue", "There is nothing special about it.", False, False))
        r3.add_item(Item("desk", "The statue is resting on it. So is a book.", False, False))
        r3.add_item(Item("closet", "The door is locked. you may need a key.", False, False))
        
        r4.add_item(Item("brewrig", "Gourd is brewing some sort of oatmeal stout on\n  the brew rig. A 6-pack is resting beside it.", False, False))
        r4.add_item(Item("barrels", "Old oak barrels.", False, False))
        r4.add_item(Item("kegs", "5 shiny silver metal kegs.", False, False))
        
        r5.add_item(Item("window", "You see a pretty view through the glass from\n  afar. Maybe you should 'go' to it and get \n  a better look.", False, False))
        r5.add_item(Item("nothing", "I told you theres nothing in this room.\n  Did you not believe me?", False, False))
        r5.add_item(Item("something", "I'm just kidding, there's still nothing.", False, False))
        
        r6.add_item(Item("refrigerator", "You open the fridge and there\n  is only a carton of eggs.", False, False))
        r6.add_item(Item("cabinets", "Cabinets lining the walls. You find nothing.", False, False))
        r6.add_item(Item("oven", "State of the art oven with a stove on top.", False, False))
        r6.add_item(Item("microwave", "Bright red microwave", False, False))
        
        r7.add_item(Item("bed", "King sized bed with fancy purple comforter and\n  many matching pillows", False, False))
        r7.add_item(Item("dresser", "Big wooden dresser. You expect to find clothes\n  but all you find is a flower.", False, False))
        r7.add_item(Item("tv", "Giant 4k TV sitting upon the dresser.\n  Definitely expensive, wouldn't want to throw\n  anything at it...", False, False))
        r7.add_item(Item("desk", "Has 3 monitors set up on it. Must be the desk\n  of a programmer.", False, False))
        
        r8.add_item(Item("flag_wall", "The owner of this house has too many flags.\n  He wouldn't mind if you take a flag.", False, False))
        r8.add_item(Item("couches", "Must be in the man cave.", False, False))
        r8.add_item(Item("movie_screen", "Great place to watch Saints games.", False, False))

        # add grabbables
        r1.add_item(Item("key", None, True, True, "table", "closet"))
        
        # ring is not grabbable until rug is vacuumed
        r2.add_item(Item("ring", None, False, False))
        
        r3.add_item(Item("book", None, True, False, "desk"))
        
        # vacuum is not grabbable until closet is unlocked
        r3c.add_item(Item("vacuum", "A vacuum used for cleaning rugs.", False, True, None, "rug"))
        
        r4.add_item(Item("6-pack", None, True, False, "brewrig"))
        
        r6.add_item(Item("eggs", None, True, True, "refrigerator", "tv"))
        
        r7.add_item(Item("flower", None, True, False, "dresser"))
        
        r8.add_item(Item("flag", None, True, False, "flag_wall"))
        
        # set current room
        self.current_room = r1
    
    def setup_gui(self) -> None:
        
        # the input element
        self.player_input = Entry(self, bg="white", fg="black")
        self.player_input.bind("<Return>", self.process_input)
        self.player_input.pack(side=BOTTOM, fill=X)
        self.player_input.focus()
        
        # image element
        img = None
        img_width = Game.WIDTH // 2
        self.image_container = Label(
            self,
            width=img_width,
            image=img
        
        )
        self.image_container.image = img
        self.image_container.pack(side=LEFT, fill=Y)
        self.image_container.pack_propagate(False)
        
        # the info element
        text_container_width = Game.WIDTH // 2
        text_container = Frame(self, width=text_container_width)
        self.text = Text(
            text_container,
            bg="lightgray",
            fg="black",
            state=DISABLED
        )
        self.text.pack(fill=Y, expand=1)
        text_container.pack(side=RIGHT, fill=Y)
        text_container.pack_propagate(False)
    
    def set_image(self, status=None):
        if self.current_room == None and status == Game.Status.DEAD:
            img = PhotoImage(file=os.path.join("images", "skull.gif"))
        # checks if eggs were used and displays end game image
        if self.current_room == None and status == Game.Status.EGGS_USED:
            img = PhotoImage(file=os.path.join("images", "skull.gif"))
        # checks if the player won and displays win image
        elif self.current_room == None and status == Game.Status.WIN:
            img = PhotoImage(file=os.path.join("images", "Win.gif"))
        else:
            img = PhotoImage(file=self.current_room.image)
        
        self.image_container.config(image=img)
        self.image_container.image=img
    
    def set_status(self, status):
        self.text.config(state=NORMAL)
        self.text.delete(1.0, END)
        if self.current_room == None:
            self.text.insert(END, status)
        else:
            content = f"{self.current_room}\n"
            content += f"  You are carrying: "
            for item in self.inventory:
                content += f"{item.name} "
            content += f"\n\n  {status}"
            self.text.insert(END, content)
        
        self.text.config(state=DISABLED)
    
    def clear_entry(self):
        self.player_input.delete(0, END)
        
    
    def handle_go(self, direction):
        status = Game.Status.BAD_EXIT
        
        if direction in self.current_room.exits:
            self.current_room = self.current_room.exits[direction]
            status = Game.Status.ROOM_CHANGED
        
        self.set_status(status)
        self.set_image()
    
    def handle_look(self, item: Item):
        status = Game.Status.BAD_ITEM
        
        # makes sure item is not a grabbable (grabbables are not displayed)
        if item in self.current_room.items and item.isGrabbable == False:
            status = item.description
        
        # vaccum is the only grabbable displayed
        elif item in self.current_room.items and item.name == "vacuum":
            status = item.description
        
        self.set_status(status)
    
    def handle_take(self, item: Item):
        status = Game.Status.BAD_GRABBABLE
        
        # makes sure item is grabbable,
        # then adds them to inventory and removes them from room
        if item in self.current_room.items and item.isGrabbable == True:
            
            self.inventory.append(item)
            self.current_room.delete_item(item)
            status = Game.Status.GRABBED
            
            # changes descriptions of the items where grabbables were taken from
            for i in self.current_room.items:
                if item.whereItsFound == i.name:
                    
                    match item.whereItsFound:
                        case "table":
                            i.description = " It is made of oak. Nothing rests on it."
                        case "desk":
                            i.description = " The statue is resting on it."
                        case "brewrig":
                            i.description = " Gourd is brewing some sort of\n  oatmeal stout on the brew rig."
                        case "refrigerator":
                            i.description = " You open the fridge and\n  it is empty."
                        case "dresser":
                            i.description = " Big wooden dresser. You expect to find\n  clothes but it's empty"
                        case "flag_wall":
                            i.description = " The owner of this house has too many flags."
        
        # checks if all non useable (besides eggs), grabbables are in inventory. 
        # Once the correct items are in the inventory, causes win screen to start
        if len(self.inventory) >= 5:
            grabbables = 0
            
            for item in self.inventory:
                if (item.isGrabbable == True and item.isUseable == False) or item.name == "eggs":
                    grabbables += 1
                
                if grabbables == 6:
                    self.current_room = None
                    status = Game.Status.WIN
        
        self.set_status(status)
    
    # use function. Allows interactions between items to allow for items to be found.
    def handle_use(self, item: Item):
        status = Game.Status.BAD_USEABLE
        
        if item.isUseable == True:
            status = Game.Status.NOT_HERE
            
            # checks what useable is used on what item
            for i in self.current_room.items:
                if item.usedOn == i.name:
                    
                    match item.name:
                        # handles key being used
                        case "key":
                            # sends player to room 3 closet
                            self.handle_go("")
                            
                            # makes vacuum grabbable
                            for j in self.current_room.items:
                                if j.name == "vacuum":
                                    j.isGrabbable = True
                            
                            # changes description of closet once key used
                            i.description == "The door is locked. The key won't work anymore..."
                            status = Game.Status.KEY_USED
                        
                        # handles eggs being used
                        case "eggs":
                            
                            # ends game when eggs used
                            self.current_room = None
                            status = Game.Status.EGGS_USED
                            self.set_image(status)
                        
                        # handles vacuum being used
                        case "vacuum":
                            # makes ring grabbable once vacuum is used
                            for k in self.current_room.items:
                                if k.name == "ring":
                                    k.isGrabbable = True
                            
                            # changes description of rug once vacuum is used
                            i.description == "rug", "It is nice and Indian. recently vacuumed"
                            status = Game.Status.VACUUM_USED
                    
                    # removes useable once used
                    self.inventory.remove(item)
                    break
        
        self.set_status(status)
    
    def handle_default(self):
        self.set_status(Game.Status.DEFAULT)
        self.clear_entry
    
    def play(self):
        self.setup_game()
        self.setup_gui()
        self.set_image()
        self.set_status("")
    
    def process_input(self, event):
        # get the input from the entry element
        action = self.player_input.get()
        action = action.lower()
        
        # stop the game if applicable
        if action in Game.EXIT_ACTION:
            exit()
        
        # clear entry if None
        if self.current_room == None:
            self.clear_entry()
            return
        
        # sanitize the input
        words = action.split()
        
        if len(words) != 2:
            self.handle_default()
            return
        
        verb = words[0]
        noun = words[1]
        
        # checks all items to see if their name matches "noun"
        # if item name (i.name) == "noun" then noun = that item
        for i in self.current_room.items:
            if i.name == noun:
                noun = i
                break
        for i in self.inventory:
            if i.name == noun:
                noun = i
                break
        
        # handle the appropiate verb
        match verb:
            case "go": self.handle_go(noun)
            case "look": self.handle_look(noun)
            case "take": self.handle_take(noun)
            case "use": self.handle_use(noun)
        
        self.clear_entry()
        




