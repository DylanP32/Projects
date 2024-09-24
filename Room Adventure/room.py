from item import Item
# Name: Dylan Pellegrin
# Description: The Room Class

class Room(Item):

    def __init__(self, name: str, image_filepath: str) -> None:
        """
        A Room in the Room Adventure game.

        name: str - a name for the room
            ex: 'Room 1' or 'Living Room'
        image_filepath: str - the relative filepath to the image
            ex: `images/room1.gif` on Linux/Potato/Mac
            ex: `images\\room1.gif` on Windows 
            Fr:  os.path.join("images", "room1.gif")
        """
        self.name = name
        self.image = image_filepath
        self.exits: dict[str,'Room'] = {} # key is the direction, value is Room 
        self.items = []
        

    def add_exit(self, location: str, room: 'Room | None') -> None:
        """
        Adds an exit to the room.

        location: str - a direction such as 'north', 'south', 'east', etc.
        room: Room | None - a room object or None. In the case of losing/death 
            sequence, use None.
        """
        self.exits[location] = room

    def add_item(self, item) -> None:
        self.items.append(item)
    
    
    def delete_item(self, item) -> None:
        self.items.remove(item)
    
    def __str__(self) -> str:

        result = f"\n  You are in {self.name}\n"

        # handle the items
        result += "  You see: "
        for item in self.items:
            result += f"{item} "
        result += "\n"

        # handle the exits
        result += "  Exits: "
        for exit_ in self.exits.keys():
            result += exit_ + " "
        result += "\n"

        return result
