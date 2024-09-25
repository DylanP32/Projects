# Description: The Item Class

# Item class. Has an assortment of attributes that allow for different interactions within the game.
class Item:
    def __init__(self, name: str, description:str, grabbable: bool, useable: bool, whereItsFound=None, usedOn=None):
        
        # a string name for each item
        # widely used for identification of items as well as displaying on GUI
        self.name = name
        
        # a string explanation of each item
        # displayed as a status for "look"
        self.description = description
        
        # a boolean that determines whether the
        # player can or cannot pick up an item
        self.isGrabbable = grabbable
        
        # a string that determines what item a grabbable can be found 
        # allows for the item descriptions to be changed once the grabbable is taken
        self.whereItsFound = whereItsFound
        
        # a boolean that determines whether
        # the player can or cannot use an item
        self.isUseable = useable
        
        # a string that determines what item a useable 
        # can be used on
        self.usedOn = usedOn
    
    # magic function so an Item can be printed
    def __str__(self) -> str:
        return f"{self.name}"
