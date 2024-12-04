from tkinter import *
# the main GUI
class MainGUI(Frame):
    # the constructor
    def __init__(self, parent):
        Frame.__init__(self, parent, bg="white")
        self.setupGUI()
    # sets up the GUI
    def setupGUI(self):
        #setup the label
        self.display = Label(self, text="", anchor=E,
        bg="white", height=1, font=("Arial", 50))
        # put it in the top row, spanning across all four
        # columns; and expand it on all four sides
        self.display.grid(row=0, column=0, columnspan=4,
        sticky=E+W+N+S)
        # configure the rows and columns of the Frame to adjust
        # to the window
        # there are 7 rows (0 through 6)
        for row in range(7):
            Grid.rowconfigure(self, row, weight=1)
        # there are 4 columns (0 through 3)
        for col in range(4):
            Grid.columnconfigure(self, col, weight=1)

        #setup the lpr button
        img = PhotoImage(file="images/lpr.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("("))
        button.image = img
        button.grid(row=1,column=0,sticky=N+S+E+W)
        
        #setup the rpr button
        img = PhotoImage(file="images/rpr.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process(")"))
        button.image = img
        button.grid(row=1,column=1,sticky=N+S+E+W)
        
        #setup the ac button
        img = PhotoImage(file="images/clr.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("AC"))
        button.image = img
        button.grid(row=1,column=2,sticky=N+S+E+W)
        
        #setup the back button
        img = PhotoImage(file="images/bak.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("BACK"))
        button.image = img
        button.grid(row=1,column=3,sticky=N+S+E+W)
        
        #setup the 7 button
        img = PhotoImage(file="images/7.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("7"))
        button.image = img
        button.grid(row=2,column=0,sticky=N+S+E+W)
        
        #setup the 8 button
        img = PhotoImage(file="images/8.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("8"))
        button.image = img
        button.grid(row=2,column=1,sticky=N+S+E+W)
        
        #setup the 9 button
        img = PhotoImage(file="images/9.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("9"))
        button.image = img
        button.grid(row=2,column=2,sticky=N+S+E+W)
        
        #setup the / button
        img = PhotoImage(file="images/div.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("/"))
        button.image = img
        button.grid(row=2,column=3,sticky=N+S+E+W)
        
        #setup the 4 button
        img = PhotoImage(file="images/4.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("4"))
        button.image = img
        button.grid(row=3,column=0,sticky=N+S+E+W)
        
        #setup the 5 button
        img = PhotoImage(file="images/5.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("5"))
        button.image = img
        button.grid(row=3,column=1,sticky=N+S+E+W)
        
        #setup the 6 button
        img = PhotoImage(file="images/6.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("6"))
        button.image = img
        button.grid(row=3,column=2,sticky=N+S+E+W)
        
        #setup the * button
        img = PhotoImage(file="images/mul.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("*"))
        button.image = img
        button.grid(row=3,column=3,sticky=N+S+E+W)
        
        #setup the 1 button
        img = PhotoImage(file="images/1.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("1"))
        button.image = img
        button.grid(row=4,column=0,sticky=N+S+E+W)
        
        #setup the 2 button
        img = PhotoImage(file="images/2.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("2"))
        button.image = img
        button.grid(row=4,column=1,sticky=N+S+E+W)
        
        #setup the 3 button
        img = PhotoImage(file="images/3.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("3"))
        button.image = img
        button.grid(row=4,column=2,sticky=N+S+E+W)
        
        #setup the - button
        img = PhotoImage(file="images/sub.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("-"))
        button.image = img
        button.grid(row=4,column=3,sticky=N+S+E+W)
        
        #setup the 0 button
        img = PhotoImage(file="images/0.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("0"))
        button.image = img
        button.grid(row=5,column=0,sticky=N+S+E+W)
        
        #setup the . button
        img = PhotoImage(file="images/dot.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("."))
        button.image = img
        button.grid(row=5,column=1,sticky=N+S+E+W)
        
        #setup the + button
        img = PhotoImage(file="images/add.gif")
        button =Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("+"))
        button.image = img
        button.grid(row=5,column=3,sticky=N+S+E+W)
        
        #setup the wide = button
        img = PhotoImage(file="images/eql-wide.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("="))
        button.image = img
        button.grid(row=6,column=0,columnspan=2,sticky=N+S+E+W)
        
        #setup the pow button
        img = PhotoImage(file="images/pow.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("**"))
        button.image = img
        button.grid(row=6,column=2,sticky=N+S+E+W)
        
        #setup the mod button
        img = PhotoImage(file="images/mod.gif")
        button = Button(self,bg="white",image=img,borderwidth=0,highlightthickness=0,activebackground="white", command=lambda: self.process("%"))
        button.image = img
        button.grid(row=6,column=3,sticky=N+S+E+W)
        
        self.pack(fill=BOTH, expand=1)
        
    # processes button presses
    def process(self, button):
        # AC clears the display
        if (button == "AC"):
            # clear the display
            self.display["text"] = ""
        # = starts an evaluation of whatever is on the display
        elif (button == "="):
            # get the expression in the display
            expr = self.display["text"]
            # the evaluation may return an error!
            try:
                # evaluate the expression
                result = str(eval(expr))
                # truncuates a 14+ character answer to an 11 character answer
                if len(result) > 14:
                    result = result[:-(len(result)-11)] + "..."
                # store the result to the display
                self.display["text"] = result
            # handle if an error occurs during evaluation
            except:
                # note the error in the display
                self.display["text"] = "ERROR"
        # BACK deletes the last character of the display
        elif (button == "BACK"):
            self.display["text"] = self.display["text"][:-1]
        else:
            # clears dsiplay if button is pressed while displaying error message
            if (self.display["text"] == "ERROR"):
                self.display["text"] = ""
            # check character length
            # does not allow for more than 14 characters to be input
            if (len(self.display["text"]) == 14):
                pass
            else:
                # otherwise, just tack on the appropriate
                # operand/operator
                self.display["text"] += button

##############################
# the main part of the program
##############################

# create the window
window = Tk()
# set the window title
window.title("The Reckoner")
# generate the GUI
p = MainGUI(window)
# display the GUI and wait for user interaction
window.mainloop()