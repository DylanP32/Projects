// Dylan Pellegrin
// Room Game part 3
// 5/23/24

// Changes made:
//   -Added 4 rooms with at least 2 items and a grabbable each
//   -Improved command line interface
//   -Allowed for items to be removed items from the room lists
//   -Added space to inventory to make room for all items


import java.util.Scanner;


public class RoomAdventure {

    private static Room currentRoom;
    private static String[] inventory = {null, null, null, null, null, null, null, null};
    private static String status;

    final private static String DEFAULT_STATUS = "Sorry, I don't understand. Try [verb] [noun].";
    
    public static void main(String[] args){
        System.out.println("\nWelcome to the Room game, your objective is to explore the rooms and discover different items.");
        System.out.println("You have 3 commands: go, look, take.");
        System.out.println("========================================================");
        setupGame();

        while (true){
            // Print info about game.

            System.out.println(currentRoom.toString());
            String Inventory = "\nInventory: ";
            for (int i=0; i < inventory.length; i++){
                if (inventory[i] != null){
                    Inventory += inventory[i] + ", ";
                }
            }
            System.out.println(Inventory);
            System.out.println("\nWhat would you like to do? ");
            
            // take input
            Scanner s = new Scanner(System.in);
            String input = s.nextLine(); // wait here for input
            System.out.println("\n========================================================");
            // process input
            String[] words = input.split(" ");

            if (words.length != 2){
                status = DEFAULT_STATUS;
            }

            String verb = words[0];
            String noun = words[1];

            switch (verb){
                case "go":
                    handleGo(noun);
                    break;
                case "look":
                    handleLook(noun);
                    break;
                case "take":
                    handleTake(noun);
                    break;
                default: status = DEFAULT_STATUS;
            }

            System.out.println("\n" + status);
            
        }
    }

    private static void handleGo(String noun){
        String[] exitDirections = currentRoom.getExitDirections();
        Room[] exitDestinations = currentRoom.getExitDestinations();
        status = "I don't see that exit.";
        for (int i = 0; i < exitDirections.length; i++){
            if (noun.equals(exitDirections[i])){
                currentRoom = exitDestinations[i];
                status = "Changed Room.";
            }
        }
    }

    private static void handleLook(String noun){
        String[] items = currentRoom.getItems();
        String[] itemDescriptions = currentRoom.getItemDescriptions();
        status = "I don't see that item.";
        for (int i = 0; i < items.length; i++){
            if (noun.equals(items[i])){
                status = itemDescriptions[i];
            }
        }
    }

    private static void handleTake(String noun){
        String[] grabbables = currentRoom.getGrabbables();
        status = "I can't grab that.";
        for (int i=0; i < grabbables.length; i++){
            if (noun.equals(grabbables[i])){
                for (int j = 0; j < inventory.length; j++){
                    if (inventory[j] == null){
                        inventory[j] = noun;
                        grabbables[i] = null;
                        status = "Added it to the inventory";
                        
                    }
                }
            }
        }
    }

    public static void setupGame(){
        Room room1 = new Room("Room 1");
        Room room2 = new Room("Room 2");
        Room room3 = new Room("Room 3");
        Room room4 = new Room("Room 4");
        Room room5 = new Room("Room 5");
        Room room6 = new Room("Room 6");
        Room room7 = new Room("Room 7");
        Room room8 = new Room("Room 8");


        // Setup Room 1
        String[] room1ExitDirections = {"east", "south", "up"};
        Room[] room1ExitDestinations = {room2, room3, room5};

        String[] room1Items = {"chair", "stool"};
        String[] room1ItemDescriptions = {
            "It is a chair.", 
            "It's like a chair. There is a key on it."
        };

        String[] room1Grabbables = {"key"};

        room1.setExitDirections(room1ExitDirections);
        room1.setExitDestinations(room1ExitDestinations);
        room1.setItems(room1Items);
        room1.setItemDescriptions(room1ItemDescriptions);
        room1.setGrabbables(room1Grabbables);


        // Setup Room 2
        String[] room2ExitDirections = {"west", "south", "up"};
        Room[] room2ExitDestinations = {room1, room4, room6};

        String[] room2Items = {"rug", "fireplace"};
        String[] room2ItemDescriptions = {
            "It's like a chair but flat. There is a ring on it.", 
            "It's hot."
        };

        String[] room2Grabbables = {"ring"};

        room2.setExitDirections(room2ExitDirections);
        room2.setExitDestinations(room2ExitDestinations);
        room2.setItems(room2Items);
        room2.setItemDescriptions(room2ItemDescriptions);
        room2.setGrabbables(room2Grabbables);


        // Setup Room 3
        String[] room3ExitDirections = {"north", "east", "up"};
        Room[] room3ExitDestinations = {room1, room4, room7};

        String[] room3Items = {"statue", "bookshelf"};
        String[] room3ItemDescriptions = {
            "It's the lady of the mist. A full sized replica.", 
            "There is one book on it."
        };

        String[] room3Grabbables = {"book"};

        room3.setExitDirections(room3ExitDirections);
        room3.setExitDestinations(room3ExitDestinations);
        room3.setItems(room3Items);
        room3.setItemDescriptions(room3ItemDescriptions);
        room3.setGrabbables(room3Grabbables);


        // Setup Room 4
        String[] room4ExitDirections = {"north", "west", "up"};
        Room[] room4ExitDestinations = {room2, room3, room8};

        String[] room4Items = {"barrels", "kegs"};
        String[] room4ItemDescriptions = {
            "Old oak barrels. A 6-pack is resting beside it.",
            "5 shiny silver metal kegs."
        };

        String[] room4Grabbables = {"6-pack"};

        room4.setExitDirections(room4ExitDirections);
        room4.setExitDestinations(room4ExitDestinations);
        room4.setItems(room4Items);
        room4.setItemDescriptions(room4ItemDescriptions);
        room4.setGrabbables(room4Grabbables);


        // Setup Room 5
        String[] room5ExitDirections = {"east", "south", "down"};
        Room[] room5ExitDestinations = {room6, room7, room1};

        String[] room5Items = {"washing_machine", "dryer"};
        String[] room5ItemDescriptions = {
            "Big white washing machine. Must be in the laundry room", 
            "Big white dryer. There is a shirt hanging out the front of it"
        };

        String[] room5Grabbables = {"shirt"};

        room5.setExitDirections(room5ExitDirections);
        room5.setExitDestinations(room5ExitDestinations);
        room5.setItems(room5Items);
        room5.setItemDescriptions(room5ItemDescriptions);
        room5.setGrabbables(room5Grabbables);


        // Setup Room 6
        String[] room6ExitDirections = {"west", "south", "down"};
        Room[] room6ExitDestinations = {room5, room8, room2};

        String[] room6Items = {"refrigerator", "cabinets", "oven"};
        String[] room6ItemDescriptions = {
            "You open the fridge and there is only a carton of eggs.",
            "Cabinets lining the walls. You find nothing.",
            "State of the art oven with a stove on top."
        };

        String[] room6Grabbables = {"eggs"};

        room6.setExitDirections(room6ExitDirections);
        room6.setExitDestinations(room6ExitDestinations);
        room6.setItems(room6Items);
        room6.setItemDescriptions(room6ItemDescriptions);
        room6.setGrabbables(room6Grabbables);


        // Setup Room 7
        String[] room7ExitDirections = {"north", "east", "down"};
        Room[] room7ExitDestinations = {room5, room8, room3};

        String[] room7Items = {"bed", "desk", "dresser"};
        String[] room7ItemDescriptions = {
            "King sized bed with fancy purple comforter and many matching pillows",
            "Has 3 monitors set up on it. Must be the desk of a programmer.",
            "Big wooden dresser. You expect to find clothes but all you find is a flower."
            };

        String[] room7Grabbables = {"flower"};

        room7.setExitDirections(room7ExitDirections);
        room7.setExitDestinations(room7ExitDestinations);
        room7.setItems(room7Items);
        room7.setItemDescriptions(room7ItemDescriptions);
        room7.setGrabbables(room7Grabbables);


        // Setup Room 8
        String[] room8ExitDirections = {"north", "west", "down"};
        Room[] room8ExitDestinations = {room5, room7, room4};

        String[] room8Items = {"flag_wall", "couches", "movie_screen"};
        String[] room8ItemDescriptions = {
            "The owner of this house has too many flags. He wouldn't mind if you take a flag.",
            "Must be in the man cave.",
            "Great place to watch Saints games."
        };

        String[] room8Grabbables = {"flag"};

        room8.setExitDirections(room8ExitDirections);
        room8.setExitDestinations(room8ExitDestinations);
        room8.setItems(room8Items);
        room8.setItemDescriptions(room8ItemDescriptions);
        room8.setGrabbables(room8Grabbables);

        currentRoom = room1;
    }


}

class Room{

    private String name;
    private String[] exitDirections; // north, south, east, west, up, down
    private Room[] exitDestinations;
    private String[] items;
    private String[] itemDescriptions;
    private String[] grabbables;

    public Room(String name){
        this.name = name;
    }

    public void setExitDirections(String[] exitDirections){
        this.exitDirections = exitDirections;
    }

    public String[] getExitDirections(){
        return exitDirections;
    }

    public void setExitDestinations(Room[] exitDestinations){
        this.exitDestinations = exitDestinations;
    }

    public Room[] getExitDestinations(){
        return exitDestinations;
    }

    public void setItems(String[] items){
        this.items = items;
    }

    public String[] getItems(){
        return items;
    }

    public void setItemDescriptions(String[] itemDescriptions){
        this.itemDescriptions = itemDescriptions;
    }

    public String[] getItemDescriptions(){
        return itemDescriptions;
    }

    public void setGrabbables(String[] grabbables){
        this.grabbables = grabbables;
    }

    public String[] getGrabbables(){
        return grabbables;
    }


    public String toString(){
        String result = "\n";

        result += "Location: You are in " + name;

        // add items to the output
        result += "\n\n   You See: ";
        for (int i = 0; i < items.length; i++){
            result += items[i] + ", ";
        }

        // add exits to the output
        result += "\n   Exits: ";
        for (int i = 0; i < exitDirections.length; i++){
            result += exitDirections[i] + ", ";
        }

        return result;
    }

}
