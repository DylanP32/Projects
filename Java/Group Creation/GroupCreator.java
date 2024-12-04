import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;

// Dylan Pellegrin
// 10/21/24

class GroupCreator 
{
    
    //*********************************************************************************
    // ADD FILE NAME HERE IN THE QUOTATIONS
    static final String filePath = "compatability_withnames.csv";  // Path to CSV file
    //*********************************************************************************

    static int numOfPeople; // Total number of people
    static int pplPerGroup; // How many people per group
    static int numOfGroups; // How many groups there will be
    static int bestTotalScore = 0; // Keeps track of the best group score 
    static long startTime; // Keeps track of the 


    public static void main(String[] args)
    {
        // Call readFile to fill and create the 2d array with all names and compatability ratings
        String[][] fileChart = readFile();

        //Prints the matrix of the CSV file
        // System.out.println();
        // for (String[] row : fileChart)
        // {
        //     for (String val : row)
        //     {
        //         System.out.print(val + ", ");
        //     }
        //     System.out.println();
        // }

        // Intro user input
        pplPerGroup = intro();
        numOfGroups = numOfPeople/pplPerGroup; // Set numOfGroups
        
        // Run time tacker
        startTime = System.currentTimeMillis();
        

        // Main loop to create groups
        List<List<List<Integer>>> finalCombos = new ArrayList<>();
        finalCombos = createGroups(fileChart, finalCombos);

        // Pick the best combo and print it out
        List<List<Integer>> bestCombination = getBestCombo(finalCombos);
        printFinalCombo(bestCombination, fileChart);
    }


    // createGroups takes the csv file put into a 2d array and once groups are made adds them to a List finalCombos
    private static List<List<List<Integer>>> createGroups(String[][] fileChart, List<List<List<Integer>>> finalCombos)
    {
        // Call group1Combos to get the 10 best combinations for group 1
        List<List<Integer>> group1Combos = createGroup1BestCombos(fileChart);

        // Iterate through each group 1 combo and create combinations for each
        for(List<Integer> combination : group1Combos)
        {
            // Create a List of each grouping for this combination starting with group 1
            List<List<Integer>> groups = new ArrayList<>();
            groups.add(combination);

            List<Integer> arr = new ArrayList<>(); // Creates a temp Array of intgers 1 to numOfPeople
            for(int i = 1; i <= numOfPeople; i++) // Represents each person
            {
                arr.add(i);
            }

            // For each remaining group create combinations
            for(int i = 1; i < numOfGroups; i++)
            {
                // If a person has already been put in a group, take them out from the selection
                for(List<Integer> group : groups)
                {
                    for(int j = 0; j < pplPerGroup; j++)
                    {
                        for(int k = 0; k < arr.size(); k++)
                        {
                            if(group.get(j) == arr.get(k))
                            {
                                
                                arr.remove(k);
                                k--;
                            }
                        }
                    }
                }

                // If this is the last group, score the remaining people and add it to groups
                if(i == numOfGroups-1)
                {
                    // Score last group
                    int tempSum = 0;
                    for(int p1 = 0; p1 < arr.size(); p1++)
                    {
                        for(int p2 = 0; p2 < arr.size(); p2++)
                        {
                            if(!(p1 == p2))
                            {
                                tempSum += Integer.parseInt(fileChart[arr.get(p1)][arr.get(p2)]);
                            }
                        }
                    }
                    arr.add(tempSum);
                    groups.add(arr);
                }
                else 
                {
                // Otherwise, create combinations for group 2 to numOfGroups, analyze their scores,
                // and keep the best one 
                List<List<Integer>> otherGroups = createCombinations(arr);
                analyzeComboScores(fileChart, otherGroups, i);
                groups.add(otherGroups.get(0));
                }
            }
            //add each combination to the finalCombos
            finalCombos.add(groups);
        }
        // returns the combinations of groupings
        return finalCombos;
    }


    // createGroup1BestCombos goes through each possible combination of groupings
    // chooses the top 10 and returns those to make the rest of the combinations
    // Takes in fileChart for the purposes of scoring
    private static List<List<Integer>> createGroup1BestCombos(String[][] fileChart)
    {

        // Create an integer array of 1 to numOfPeople to be used to represent each person
        List<Integer> arr = new ArrayList<>();
        for(int i = 1; i <= numOfPeople; i++)
        {
            arr.add(i);
        }

        // Creates all possible combinations for group 1
        // and puts them in a List of Lists of people (represented by their order in the chart)
        List<List<Integer>> group1Combos = createCombinations(arr);

        // Score each combination and keep the top 10
        analyzeComboScores(fileChart, group1Combos, 0);

        return group1Combos;
    }


    // createCombinations creates the results list
    // then calls backtrack to fill said list
    // Returns the list for further use
    private static List<List<Integer>> createCombinations(List<Integer> arr)
    {
        List<List<Integer>> result = new ArrayList<>();
        backtrack(result, new ArrayList<>(), arr, 0, pplPerGroup);

        // Prints combinations created
        // for (List<Integer> combination : result)
        // {
        //     System.out.println(combination);
        // }

        return result;
    }


    // backtrack causes a recursive iteration which goes through all combinations of groupings
    // of a set of integers and puts each into a temp list of integers.
    // These temp lists are compiled into the final "results" list created in createCombinations
    private static void backtrack(List<List<Integer>> result, List<Integer> tempList, List<Integer> arr, int start, int pplPerGroup)
    {
        if (pplPerGroup == 0)
        {
            result.add(new ArrayList<>(tempList));
            return;
        }

        for (int i = start; i < arr.size(); i++)
        {
            tempList.add(arr.get(i));
            backtrack(result, tempList, arr, i + 1, pplPerGroup - 1);
            tempList.remove(tempList.size() - 1);
        }
    }


    // analyzeComboScores takes a List of Lists of all the different combinations of people
    // and adds up the score of the combination and adds it to the pplPerGroup + 1 index
    // analyzeComboScores then removes every combo except the top 10 scores for group 1 and the 
    // highest score for every group after
    private static void analyzeComboScores(String[][] fileChart, List<List<Integer>> combos, int groupNum)
    {
        // p1 = person 1
        // p2 = person 2

        int tempSum = 0; // Holds the sum of a group score
        boolean haterFound = false; // Represents if a rating that is less than 0 is found

        // Iterate through each combination
        for(int combination = 0; combination < combos.size(); combination++)
        {
            tempSum = 0; // Restart the group score for each group

            // Iterate through each person and adds up their rating for each other group member excluding themselves
            // Ratings are added to tempSum
            for(int p1 = 0; p1 < combos.get(combination).size(); p1++)
            {
                for(int p2 = 0; p2 < combos.get(combination).size(); p2++)
                {
                    if(!(p1 == p2))
                    {
                        // If a hater is found break the for loop and delete this combination
                        if(Integer.parseInt(fileChart[combos.get(combination).get(p1)][combos.get(combination).get(p2)]) < 0)
                        {
                            haterFound = true;
                            combos.remove(combination);
                            combination--;
                            break;
                        }
                        // Otherwise tally up the sum
                        tempSum += Integer.parseInt(fileChart[combos.get(combination).get(p1)][combos.get(combination).get(p2)]);
                    }
                }
                if(haterFound)
                {
                    break;
                }
            }
            if(haterFound)
            {
                haterFound = false;
            }
            else // if there is no haters, add the score to the group
            {
                combos.get(combination).add(tempSum);
            }
        }

        // Sort each combination of a group based on their last index which holds their score in descending order
        Collections.sort(combos, Collections.reverseOrder(Comparator.comparing(combo -> combo.get(pplPerGroup))));

        // if this is group 1
        if(groupNum == 0)
        {
            // Keep only the top 10 combinations
            combos.subList(10, combos.size()).clear();
        }
        else // Otherwise keep the top score of all combinations and clear the rest
        {
            List<Integer> temp = combos.get(0);
            combos.clear();
            combos.add(temp);
        }
        
        // Prints the resulting combos
        // System.out.println(combos)
    }


    // getBestCombo goes through the resulting list of the ten best combinations (finalCombos) and adds up 
    // each total score and chooses the highest scoring one, returning it as a List of groups
    public static List<List<Integer>> getBestCombo(List<List<List<Integer>>> finalCombos)
    {
        // Prints the final 10 combos
        // System.out.println(finalCombos)

        int tempScore = 0; // Holds the score of each combination temporarily
        int combinationCounter = 0; // Keeps track of which combination is being evaluated
        
        // Holds the combo with the best total score
        List<List<Integer>> bestTotalCombo = new ArrayList<>();

        // Iterate through each of the final combinations
        for(List<List<Integer>> combination : finalCombos)
        {
            // Iterate through each group in a combination and add up their score based on the last index
            for(List<Integer> group : combination)
            {
                tempScore += group.get(pplPerGroup);
            }

            // basae case, the first group is the assumed best score
            if(combinationCounter == 0)
            {
                bestTotalScore = tempScore;
                bestTotalCombo = combination;
            }
            else if(tempScore > bestTotalScore) // Otherwise, if the tempScore of the new group is better than the last, make the new group the best group.
            {
                bestTotalScore = tempScore;
                bestTotalCombo = combination;
            }
            tempScore = 0;
            combinationCounter++;
        }
        finalCombos.clear(); //Clear all combos
        
        // Prints the final combo
        // System.out.println(bestTotalCombo)

        // Return the final groupings
        return bestTotalCombo;
    }


    private static void printFinalCombo(List<List<Integer>> bestCombination, String[][] fileChart)
    {
        System.out.println("=========================================================================================================================");
        int counter = 1; // Keeps track of group
        System.out.println("Final Combination with the total score of " + bestTotalScore + ":\n");
        // Iterate through each group and print it
        for(List<Integer> group : bestCombination)
        {
            System.out.print("\tGroup " + (counter) + ": ");
            // Only run through the first n-1 indexes so the group score is not printed
            for(int person = 0; person < pplPerGroup; person++)
            {
                System.out.print(fileChart[0][group.get(person)] + ", ");
            }
            // Print group score acording to last index which still houses group score
            System.out.println("\n\tGroup score: " + group.get(pplPerGroup));
            System.out.println();
            counter++;
        }

        // Print the run time
        long endTime = System.currentTimeMillis();
        long executionTime = endTime - startTime;
        System.out.println("Run time: " + executionTime + " milliseconds");
    }


//************************************************ USER INPUT ************************************************


    // intro prompts the user for how many people they want per group
    // Implements range checking to validate that inputs are correct
    private static int intro()
    {
        System.out.println("\n====================================================================================");
        System.out.println("Group Creator");
        System.out.println("====================================================================================");
        System.out.println("Disclaimer: Input for the amount of people per group must be factor of the total" +
                            "\nnumber of people that is not 1, half of, or the actual total number of people");
        System.out.println("====================================================================================\n");
        
        // Get user input for the amount of people per group and checking for a valid input
        boolean validInput = false;
        while(!(validInput))
        {

            Scanner s = new Scanner(System.in);
            System.out.println("Input the number of people per group: ");
            String perGroupInput = s.nextLine(); // Wait here for input

            try {
                // Try making people per group input into an int
                int tempNum = Integer.parseInt(perGroupInput);

                // Checks if the people per group is not 1, half of or the total numOfPeople
                // as well as if amount of groups is even and a factor of total people
                if((tempNum != 1) && (tempNum != numOfPeople/2) && (tempNum != numOfPeople) && (numOfPeople % tempNum == 0))
                {
                    validInput = true; // if inputs pass checks, kill the loop and process combos
                }
                else
                {
                    // Throw an error if this is false
                    int x = 1/0;
                }
            } catch (Exception e) {

                // Reminds user of parameters for input and continues
                System.out.println("\n====================================================================================");
                System.out.println("The input provided is not valid.");
                System.out.println("The amount of people per group must be factor of the total number of" +
                                    "\npeople that is not 1, half of, or the actual total number of people");
                System.out.println("====================================================================================\n");
                continue;
            }
            // Tracks the amount of people per group and closes scanner
            pplPerGroup = Integer.parseInt(perGroupInput);
            s.close();
        }
        return pplPerGroup;
    }


    // readFile takes a csv file and inputs it into a 2d array of Strings
    private static String[][] readFile()
    {
        // Create the 2d array that will store all names as well as ratings.
        // Size is determined by the BufferedReader.
        String[][] chart = null;

        try (BufferedReader br = new BufferedReader(new FileReader(filePath)))
        {
            String line; // Stores each line from the file
            int i = 0; // Row counter
            int size = 0; // Holds the size of each row in the chart

            // Read the lines into the 2d array until the end of the file
            while ((line = br.readLine()) != null)
            {
                // Split the line into columns based on the delimiter, storing the row into a String array
                String[] row = line.split(",");

                if (i == 0)
                    {
                        // Instantiate chart with the size of the line
                        size = row.length;
                        chart = new String[size][size];

                        // Set the numOfPeople to size -1 because the chart has an extra column 
                        numOfPeople = size - 1;
                        
                    }

                // for the length of the row, add each index to the 2d array
                for (int j = 0; j < size; j++)
                { 
                    // Try setting the ratings to an int
                    // If not possible then the file is not correct
                    if(!(i == 0 || j == 0))
                    {
                        int temp = Integer.parseInt(row[j]);
                    }
                    chart[i][j] = row[j];
                }
                i++;
            }
        }
        catch (Exception e) {
            System.out.println("\n====================================================================================");
            System.out.println("There is problems with the file provided. Make sure that the file name is correct," +
                               "\nthat the file is in the same directory as the GroupCreator.java, and that the" +
                               "\ninformation within the CSV file is formatted where the first row and column are all" +
                               "\nnames and everything else is integer ratings. Lastly, be sure that the number of" +
                               "\npeople being split into groups is not a prime number.");
            System.out.println("\n====================================================================================");
            e.printStackTrace(); // Handle exceptions (e.g., file not found)
            System.exit(0);
        }
           return chart;
    }
}