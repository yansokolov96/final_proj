# This code is a script that uses the WL node coloring method to color the nodes in a graph dataset, and then runs multiple other methods on the colored graph dataset. The script uses the updateGUI function to update a GUI with the progress of the script and the results.
# It starts by removing old evaluation plots and model. It then loads the dataset 'cora' and sets the seed for random number generators for reproducibility.
# It then runs the WL node coloring method on the dataset, which is implemented in the MethodWLNodeColoring class, and saves the results in the './result/WL/' folder.
# It then runs multiple other methods on the colored graph dataset, updating the GUI with the progress and results of each method.
# It also has a parameter precentage which is passed to the function "runScript" as an input and it is used to delete a certain percentage of nodes from the dataset, but the script does not make use of this parameter in this case.

import numpy as np
import torch
import os
import tkinter as tk

from code.DatasetLoader import DatasetLoader
from code.MethodWLNodeColoring import MethodWLNodeColoring
from code.MethodGraphBatching import MethodGraphBatching
from code.MethodHopDistance import MethodHopDistance
from code.ResultSaving import ResultSaving
from code.Settings import Settings


#GUI UPDATE
progressPercentage = 0

def updateGUI(Progress_Bar_Precentage,GUI_Text,GUI,update_progress=None):
    global GUI_Total_Text,progressPercentage
    GUI_Total_Text =  GUI_Text + "\n\n"
    # GUI.delete(1.0, tk.END)
    GUI.insert(tk.END, GUI_Total_Text)
    progressPercentage += Progress_Bar_Precentage
    GUI.see(tk.END)

    if progressPercentage > 100: progressPercentage = 100

    if update_progress:
        update_progress(progressPercentage, 100)  # Update progress to 1%

def runScript(update_progress=None,GUI = None,precentage = 10):

    updateGUI(1,str(precentage) + "% Nodes To Delete\n\n" + "------------------------Pre Processing------------------------",GUI,update_progress)

    # remove old results
    try:
        os.remove("EvaluationPlot1.png")
        os.remove("EvaluationPlot2.png")
        os.remove("EvaluationPlot3.png")
        os.remove("Matrix.png")
        os.remove("Model3D.png")
    except OSError as e:
        print("Could not remove old evaluations and model")

    GUI.insert(tk.END, "\nOld Evaluations and old Model have been removed.\n")

    # sets an environment variable "KMP_DUPLICATE_LIB_OK" to "True". This is used to resolve a problem that can occur when using multiple threads on macOS and Linux with the Intel MKL library.
    # When this environment variable is set to "True", it allows the program to run correctly even if multiple versions of the same library are present in the system. This is done to prevent the program from crashing or producing unexpected results.
    # This variable is used to make sure that the program will run smoothly even if there are multiple versions of the library present in the system.
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


    #---- 'cora' , 'citeseer', 'pubmed' ----

    dataset_name = 'cora'

    np.random.seed(1)
    torch.manual_seed(1)

    # nclass: The number of classes in the dataset. for instance a class could be birds,fish,mammals (not related to the project)
    # nfeature: The number of features in the dataset. a feature in cars could be a model, price and etc.
    # ngraph: The number of graph in the dataset. a graph could be for instance in social networks a network of family, network of friends ands etc.

    #---- cora-small is for debuging only ----
    if dataset_name == 'cora-small':
        nclass = 7
        nfeature = 1433
        ngraph = 10
    elif dataset_name == 'cora':
        nclass = 7
        nfeature = 1433
        ngraph = 2708
    elif dataset_name == 'citeseer':
        nclass = 6
        nfeature = 3703
        ngraph = 3312
    elif dataset_name == 'pubmed':
        nclass = 3
        nfeature = 500
        ngraph = 19717


    #---- Step 1: WL based graph coloring ----
    if 1:
        print('************ Start ************')
        print('WL, dataset: ' + dataset_name)
        updateGUI(1, "************ Start ************\n\n" + 'WL, dataset: ' + dataset_name, GUI, update_progress)
        # ---- objection initialization setction ---------------
        data_obj = DatasetLoader()
        data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
        data_obj.dataset_name = dataset_name

        method_obj = MethodWLNodeColoring()

        result_obj = ResultSaving()
        result_obj.result_destination_folder_path = './result/WL/'
        result_obj.result_destination_file_name = dataset_name

        setting_obj = Settings()

        evaluate_obj = None
        # ------------------------------------------------------

        # ---- running section ---------------------------------
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.load_run_save_evaluate()
        # ------------------------------------------------------

        print('************ Finish ************')
        updateGUI(1, "************ Finish ************", GUI, update_progress)
    #------------------------------------

    updateGUI(10,"Step 1: WL based graph coloring ✅",GUI,update_progress)


    #---- Step 2: intimacy calculation and subgraph batching ----
    if 1:
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:#, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            print('************ Start ************')
            print('Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k))
            updateGUI(4, '************ Start ************\n\n' + 'Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k), GUI, update_progress)
            # ---- objection initialization setction ---------------
            data_obj = DatasetLoader()
            data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
            data_obj.dataset_name = dataset_name
            data_obj.compute_s = True

            method_obj = MethodGraphBatching()
            method_obj.k = k

            result_obj = ResultSaving()
            result_obj.result_destination_folder_path = './result/Batch/'
            result_obj.result_destination_file_name = dataset_name + '_' + str(k)

            setting_obj = Settings()

            evaluate_obj = None
            # ------------------------------------------------------

            # ---- running section ---------------------------------
            setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
            setting_obj.load_run_save_evaluate()

            # updateGUI(4, 'Subgraph Batching, dataset: ' + dataset_name + ', k: ' + str(k), GUI, update_progress)

            # ------------------------------------------------------
            updateGUI(0, '************ Finish ************', GUI, update_progress)
            print('************ Finish ************')
    #------------------------------------
    updateGUI(10, "Step 2: intimacy calculation and subgraph batching ✅", GUI, update_progress)

     # Hop distance is a measure of the distance between two nodes in a graph. It is defined as the number of edges that need to be traversed to go from one node to the other. The hop distance between two nodes can also be thought of as the length of the shortest path between the two nodes in terms of the number of edges.
    #
    # For example, in a simple graph with three nodes A, B, and C connected by edges, the hop distance between node A and node C is 2, because to go from A to C, we need to traverse two edges: A-B-C.

     #---- Step 3: Shortest path: hop distance among nodes ----
    if 1:
        for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print('************ Start ************')
            updateGUI(4, '************ Start ************\n\n' + 'HopDistance, dataset: ' + dataset_name + ', k: ' + str(k), GUI, update_progress)
            print('HopDistance, dataset: ' + dataset_name + ', k: ' + str(k))
            # ---- objection initialization setction ---------------
            data_obj = DatasetLoader()
            data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
            data_obj.dataset_name = dataset_name

            method_obj = MethodHopDistance()
            method_obj.k = k
            method_obj.dataset_name = dataset_name

            result_obj = ResultSaving()
            result_obj.result_destination_folder_path = './result/Hop/'
            result_obj.result_destination_file_name = 'hop_' + dataset_name + '_' + str(k)

            setting_obj = Settings()

            evaluate_obj = None
            # ------------------------------------------------------

            # ---- running section ---------------------------------
            setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
            setting_obj.load_run_save_evaluate()
            # ------------------------------------------------------
            print('************ Finish ************')
            updateGUI(0,'************ Finish ************\n', GUI, update_progress)


    #------------------------------------
    updateGUI(100,"Shortest path: hop distance among nodes\nPre Processing ✅",GUI,update_progress)
