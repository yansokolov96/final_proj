import threading
import tkinter as tk
import time

from tkinter import ttk, messagebox
from PIL import ImageTk, Image
from PIL.Image import Resampling

import script_0_delete_random_links as DeleteLinks
import script_1_preprocess as PreProcessClass
import script_2_pre_train as PreTrain
import script_3_fine_tuning as FineTuning
import script_4_evaluation_plots as EvaluationPlot
import script_5_load_run_data as RunScript
import script_7_avg as Average
import os
import testFile
# Create the root window
# from PIL.Image import Resampling



root = tk.Tk()
root.title("GRAPH-BERT")


# Create a left panel and configure its appearance
left_panel = tk.Frame(root, bg="#6699CC", width=200, height=400)
left_panel.pack(side="left", fill="both", expand=True)

lblHeader = tk.Label(left_panel, text="\t      Graph - BERT\t\t", bg="#2A537D", fg="#142639", activebackground="#003366", activeforeground="#FFFFFF", pady=10,font=("Arial", 12, "bold"))
lblHeader.pack(fill="x")

def parse_int(text_box):
    try:
        return int(text_box.get("1.0", tk.END).strip())
    except ValueError:
        return 10

def dyeAllButtonsWhite():
    button1.configure(fg = 'white')
    button2.configure(fg = 'white')
    button3.configure(fg = 'white')
    button4.configure(fg = 'white')
    button5.configure(fg = 'white')
    openFileManagerBtn.configure(fg = 'white')


# Create some buttons and add them to the left panel
def button1_clicked():
    dyeAllButtonsWhite()
    update_progress(0,100)
    button1.configure(fg = 'Yellow')
    picture1.pack_forget()
    picture2.pack_forget()
    picture3.pack_forget()

    if not isNumberOfRunsValid():
        messagebox.showinfo("Alert", "Invalid input. Please enter a valid number of rows")
        return
    else: numberOfRuns = int(numberOfRunsTB.get(1.0, tk.END).strip())


    if not isPrecentageIsValid():
        messagebox.showinfo("Alert", "Invalid input. Number of nodes to delete must be a number between 1 and 90")
        deleteNodesTB.delete(1.0, tk.END)
        deleteNodesTB.insert(tk.END, "10")
        return
    else: percentage = int(deleteNodesTB.get(1.0, tk.END).strip())


    deleteNodesTB.delete(1.0,tk.END)
    deleteNodesTB.insert(tk.END, percentage)
    deleteNodesTB.pack()

    for i in range(1,numberOfRuns+1):
        DeleteLinks.runScript(percentage)
        PreProcessClass.runScript(update_progress, text_box,percentage)
        button1.config(state='disabled')
        button2.config(state='normal')
        button2_clicked(i)
    Average.runScript(numberOfRuns)
    picture1.pack_forget()
    picture2.pack_forget()
    picture3.pack_forget()
    addImage("./ImageResults/ConvMatrix.png", picture1, 800, 400)



def update_progress(current, total):
    progress['value'] = int(current / total * 100)
    progress.update()

# Create a progress bar
progress = ttk.Progressbar(root, orient='horizontal', length=510, mode='determinate')
progress.pack()

# panel = tk.Frame(left_panel,width = 10,height = 10)
# panel.pack()

label = tk.Label(left_panel, text="Delete Number Of Rows(%): ",bg="#6699CC")
label.pack(fill="x")

deleteNodesTB = tk.Text(left_panel, bg="white", fg="Black", width=3, height=1,padx = 10, pady = 5,bd = 1,font = ("Arial", 12))
deleteNodesTB.insert(tk.END, 10)
deleteNodesTB.tag_configure("center", justify='center')
deleteNodesTB.tag_add("center", 1.0, "end")
deleteNodesTB.pack()


label2 = tk.Label(left_panel, text="Number Of Runs: ",bg="#6699CC")
label2.pack(fill="x")

numberOfRunsTB = tk.Text(left_panel, bg="white", fg="Black", width=3, height=1,padx = 10, pady = 5,bd = 1,font = ("Arial", 12))
numberOfRunsTB.insert(tk.END, 1)
numberOfRunsTB.tag_configure("center", justify='center')
numberOfRunsTB.tag_add("center", 1.0, "end")
numberOfRunsTB.pack()

button1 = tk.Button(left_panel, text="PreProcessing", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=button1_clicked, pady=10)
button1.pack(fill="x")



def button2_clicked(i):
    dyeAllButtonsWhite()
    button2.configure(fg = 'Yellow')
    PreTrain.runScript(update_progress, text_box)
    button2.config(state='disabled')
    button3.config(state='normal')
    button3_clicked(i)


button2 = tk.Button(left_panel, text="PreTrain", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=button2_clicked, pady=10)
button2.pack(fill="x")

def button3_clicked(i):
    dyeAllButtonsWhite()
    button3.configure(fg = 'Yellow')
    FineTuning.runScript(update_progress, text_box)
    button3.config(state='disabled')
    button4.config(state='normal')
    button5.config(state='normal')
    button1.config(state='normal')
    runAverageConv(i)

button3 = tk.Button(left_panel, text="FineTuning", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=button3_clicked, pady=10)
button3.pack(fill="x")


def button4_clicked():
    dyeAllButtonsWhite()
    button4.configure(fg = 'Yellow')
    EvaluationPlot.runScript(text_box)
    x = 250
    y = 230
    addImage("EvaluationPlot1.png",picture1,x,y)
    addImage("EvaluationPlot2.png",picture2,x,y)
    addImage("EvaluationPlot3.png",picture3,x,y)

button4 = tk.Button(left_panel, text="Show Network Results", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=button4_clicked, pady=10)
button4.pack(fill="x")

def runAverageConv(i):
    dyeAllButtonsWhite()
    button5.configure(fg = 'Yellow')
    RunScript.runScript(i)
    addImage("./ImageResults/Model3D" + str(i) + ".png",picture1,400,250)
    addImage("./ImageResults/Matrix" + str(i) + ".png", picture2, 400, 250)
    picture3.pack_forget()

def showAvgMatrix():
    dyeAllButtonsWhite()
    button5.configure(fg='Yellow')
    if(os.path.exists("ImageResults/ConvMatrix.png")):
        picture1.pack_forget()
        picture2.pack_forget()
        picture3.pack_forget()
        addImage("./ImageResults/ConvMatrix.png", picture1, 760, 350)
    else: messagebox.showinfo("Alert", "Convolutional Matrix does not exist")

button5 = tk.Button(left_panel, text="Show Average Convolutional Matrix", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=showAvgMatrix, pady=10)
button5.pack(fill="x")


def openFileManagerClicked():
    dyeAllButtonsWhite()
    openFileManagerBtn.configure(fg='Yellow')
    project_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(project_dir, "ImageResults")
    os.startfile(results_folder)

openFileManagerBtn = tk.Button(left_panel, text="Show Run Results", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=openFileManagerClicked, pady=10)
openFileManagerBtn.pack(fill="x")


# button6 = tk.Button(left_panel, text="Load & Run", bg="#336699", fg="#FFFFFF", activebackground="#003366", activeforeground="#FFFFFF", command=button6_clicked, pady=10)
# button6.pack(fill="x",side = tk.BOTTOM)

# Create a right panel and configure its appearance
right_panel = tk.Frame(root, bg="#59799A", width=300, height=300)
right_panel.pack(side="right", fill="both", expand=True)




# Create a text box in the right panel
text_box = tk.Text(right_panel, bg="#83ABD4", fg="Black", width=40, height=30,padx = 20, pady = 20,bd = 0,font = ("Arial", 12, "bold"))
text_box.pack()

button2.config(state='disabled')
button3.config(state='disabled')
# button4.config(state='disabled')
# button5.config(state='disabled')


picture1 = tk.Label(right_panel)
picture2 = tk.Label(right_panel)
picture3 = tk.Label(right_panel)




# new_image = None

def addImage(name,picture,sizeX,sizeY):
    img = Image.open(name)
    resized_image = img.resize((sizeX, sizeY), resample=Resampling.BICUBIC)
    new_image = ImageTk.PhotoImage(resized_image)
    picture.configure(image=new_image)
    picture.image = new_image  # store a reference to the image object
    picture.pack(side='left')


def isEvaluationExist():
    return os.path.exists("EvaluationPlot1.png") and os.path.exists(
        "EvaluationPlot2.png") and os.path.exists

def isModelExist():
    return os.path.exists("./ImageResults/Model3D.png") and os.path.exists("./ImageResults/Matrix.png")

def isPrecentageIsValid():
     valid = deleteNodesTB.get(1.0, tk.END).strip().isdigit()
     if valid:
          percentage = int(deleteNodesTB.get(1.0, tk.END).strip())
     else: return False
     return percentage > 0 and percentage <= 90

def isNumberOfRunsValid():
    valid =  numberOfRunsTB.get(1.0, tk.END).strip().isdigit()
    if valid:
          value = int(numberOfRunsTB.get(1.0, tk.END).strip())
    else: return False
    return value > 0


# if isEvaluationExist():
#     button4.config(state='normal')
#     text_box.insert(tk.END, "Evaluation Plots are already existed from the last run, you can display them on the screen by clicking on 'Network Results'\n")






root.mainloop()
