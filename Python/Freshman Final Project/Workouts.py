import tkinter as tk
from Page import Page, ResultsPage, workoutReps, workoutWeights

# The Workout 1 class. Root class that initializes all other pages.
# Also displays the first page.
class Workout1(Page):
    def __init__(self, root=None):
        
        # creates root page (page 1).
        self.root = root
        self.frame = tk.Frame(self.root, bg = "#749CBB")
        self.frame.pack()
        
        # initializes the page GUI.
        Page.__init__(self, self.frame, self.root, 1, "One")
        
        # initializes workout pages 1 and 2.
        self.page_2 = Workout2(master=self.root, app=self)
        self.page_3 = Workout3(master=self.root, app=self)
        self.page_4 = WorkoutResults(master=self.root, app=self)
    
    # Starts page 1.
    def main_page(self):
        self.frame.pack()
    
    # Starts page 2. Makes sure there in valid inputs and workout was completed.
    def make_next_page(self):
        if self.valid_inputs and self.play_clicked and self.stop_clicked == True:
            self.frame.pack_forget()
            self.page_2.start_page()
        
        elif self.valid_inputs == False:
            self.invalidEntry.set("Invalid Entry. Please input the weight \nbeing lifted and reps lifted")
            Invalid_Entry = tk.Label(self.frame2, textvariable = self.invalidEntry, bg = "white", font = ("texgyreadventor-regular", 10))
            Invalid_Entry.grid(row=2, column=0, columnspan= 2, sticky="")
            
        elif self.play_clicked or self.stop_clicked == False:
            self.workout_not_complete.set("Workout not complete.")
            Complete_Workout = tk.Label(self.frame3, textvariable = self.workout_not_complete, bg = "white", font = ("texgyreadventor-regular", 15))
            Complete_Workout.grid(row=2, column=0, columnspan=2, pady=(5,0), sticky="")


# The Workout 2 class. Displays second page.
class Workout2(Page):
    def __init__(self, master=None, app=None):
        
        # Creates page 2.
        self.master = master
        self.app = app
        self.frame = tk.Frame(self.master, bg = "#749CBB")
        
        # Initializes the page GUI.
        Page.__init__(self, self.frame, self.master, 2, "Two")
    
    # Starts page 2.
    def start_page(self):
        self.frame.pack()
    
    # Starts page 3.
    def make_next_page(self):
        if self.valid_inputs and self.play_clicked and self.stop_clicked == True:
            self.frame.pack_forget()
            self.app.page_3.start_page()
        
        elif self.valid_inputs == False:
            self.invalidEntry.set("Invalid Entry. Please input the weight \nbeing lifted and reps lifted")
            Invalid_Entry = tk.Label(self.frame2, textvariable = self.invalidEntry, bg = "white", font = ("texgyreadventor-regular", 10))
            Invalid_Entry.grid(row=2, column=0, columnspan=2, sticky="")
            
        elif self.play_clicked or self.stop_clicked == False:
            self.workout_not_complete.set("Workout not complete.")
            Complete_Workout = tk.Label(self.frame3, textvariable = self.workout_not_complete, bg = "white", font = ("texgyreadventor-regular", 15))
            Complete_Workout.grid(row=2, column=0, columnspan=2, pady=(5,0), sticky="")


# The Workout 3 class. Displays third page
class Workout3(Page):
    def __init__(self, master=None, app=None):
        
        # Creates page 3.
        self.master = master
        self.app = app
        self.frame = tk.Frame(self.master, bg = "#749CBB")
        Page.__init__(self, self.frame, self.master, 3, "Three")
    
    # Starts page 3.
    def start_page(self):
        self.frame.pack()
    
    # Starts page 1.
    def make_next_page(self):
        if self.valid_inputs and self.play_clicked and self.stop_clicked == True:
            self.frame.pack_forget()
            self.app.page_4.start_page()
        
        elif self.valid_inputs == False:
            self.invalidEntry.set("Invalid Entry. Please input the weight \nbeing lifted and reps lifted")
            Invalid_Entry = tk.Label(self.frame2, textvariable = self.invalidEntry, bg = "white", font = ("texgyreadventor-regular", 10))
            Invalid_Entry.grid(row=2, column=0, columnspan=2, sticky="")
            
        elif self.play_clicked or self.stop_clicked == False:
            self.workout_not_complete.set("Workout not complete.")
            Complete_Workout = tk.Label(self.frame3, textvariable = self.workout_not_complete, bg = "white", font = ("texgyreadventor-regular", 15))
            Complete_Workout.grid(row=2, column=0, columnspan=2, pady=(5,0), sticky="")


class WorkoutResults(ResultsPage):
    def __init__(self, master=None, app=None):
        self.master = master
        self.app = app

    
    # Starts results page
    def start_page(self):
        self.frame = tk.Frame(self.master, bg = "#749CBB")
        ResultsPage.__init__(self, self.frame, self.master, workoutWeights, workoutReps)
        self.frame.pack()
    
    # # Resets app
    # def start_over(self):
    #     self.frame.pack_forget()
    #     self.app.main_page()

######################## MAIN ###########################

# Main page loop.
if __name__ == '__main__':
    root = tk.Tk()
    
    # Generates the GUI.
    app = Workout1(root)
    
    # Display the GUI.
    root.mainloop()

