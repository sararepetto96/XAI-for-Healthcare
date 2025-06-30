from heapq import nlargest, nsmallest
import json
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
import os
from typing import Tuple, List, Dict

import numpy as np
from captum.attr._utils import visualization
from matplotlib import pyplot as plt

from .ExplainableModels import ExplainableModel
from .MedDataset import MedDataset

import pandas as pd

from torchvision.transforms import ToPILImage

from medmnistc.corruptions.registry import CORRUPTIONS_DS

from tqdm import tqdm

from .agreement import batch_agreement, image_convolution_agreement, image_density_agreement

import seaborn as sns


models = ["MedVit", #convolutional+transformer model
          "densenet121", "resnet50", "vgg16", "convnext_base", #convolutional models
          "custom_deit_base_patch16_224", "custom_vit_base_patch16_224","custom_pit_b_224"] #transformer models



def print_agreements(agreement_measure:str,
                    print_mode:str, 
                    table_mean:pd.DataFrame, table_var:pd.DataFrame, 
                    model_name:str, data_name:str, data_split:str, agreement_type:str):
    
    if agreement_measure == "density":
        name = r"Convolution Density Agreement ($\rho$)"
    elif agreement_measure == "l2":
        name = r"L2 distance ($\ell_2$)"
    elif agreement_measure == "SSIM":
        name = r"Structural Similarity Index (SSIM)"
    else:
        raise ValueError(f"Unknown agreement measure: {agreement_measure}")
    
    mean_title = f"Mean {name} for {data_name} {data_split}"
    var_title = f"Variance {name} for {data_name} {data_split}"
    
    if print_mode == "latex":
        table_mean.fillna(" ", inplace = True)
        table_var.fillna(" ", inplace = True)
        print()
        print(table_mean.to_latex(float_format="%.4f", caption=mean_title, position="hbt!"))
        print()
        print(table_var.to_latex(float_format="%.4f", caption=var_title, position="hbt!"))
    elif print_mode == "print":
        print(mean_title)
        print(table_mean)
        
        print(var_title)
        print(table_var)
    elif print_mode == "csv" or print_mode == "heatmap":
        
        folder = f"{agreement_type}"
        os.makedirs(folder, exist_ok=True)
        
        if agreement_measure == "density" or agreement_measure == "SSIM":
            table_mean.fillna(1.0, inplace = True)
            vmax = 1.0
        elif agreement_measure == "l2":
            table_mean.fillna(0.0, inplace = True)
            vmax = table_mean.to_numpy().max()
        else:
            raise ValueError(f"Unknown agreement measure: {agreement_measure}")
        
        table_var.fillna(0.0, inplace = True)
        
        if print_mode == "csv":
            
            table_mean.to_csv(f"{agreement_type}/mean_{agreement_measure}_{model_name}_{data_name}_{data_split}.csv", float_format="%.4f")
            table_var.to_csv(f"{agreement_type}/var_{agreement_measure}_{model_name}_{data_name}_{data_split}.csv", float_format="%.4f")
        
        if print_mode == "heatmap":
            fig, ax = plt.subplots(figsize=(10,5))
            sns.heatmap(table_mean, annot=True, ax=ax, vmax=vmax, vmin=0.0)
            ax.set_title(mean_title)
            
            if agreement_type == "explanation_agreement":
                ax.set_xlabel("Target XAI technique")
                ax.set_ylabel("Source XAI technique")
            elif "corruption_agreement" in agreement_type:
                
                ax.set_ylabel("XAI technique")
                #remove x axis thick labels
                ax.set_xlabel("")
                ax.set_xticklabels([])
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
                
            fig.tight_layout()
            fig.savefig(f"{agreement_type}/mean_{agreement_measure}_{model_name}_{data_name}_{data_split}.png")
            
            fig, ax = plt.subplots(figsize=(10,5))
            sns.heatmap(table_var, annot=True, ax=ax,vmax=1.0, vmin=0.0)
            ax.set_title(var_title)
                
            if agreement_type == "explanation_agreement":
                ax.set_xlabel("Target XAI technique")
                ax.set_ylabel("Source XAI technique")
                
            elif "corruption_agreement" in agreement_type:
                
                ax.set_ylabel("XAI technique")
                #remove x axis thick labels
                ax.set_xlabel("")
                ax.set_xticklabels([])
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
            
            fig.tight_layout()
            fig.savefig(f"{agreement_type}/var_{agreement_measure}_{model_name}_{data_name}_{data_split}.png")


        
def corruption_agreement( model_name:str, 
                        data_name: str, data_name_corrupted:str, data_split: str, n_classes:int, 
                        best_to_show: int = 5, worst_to_show: int = 5,
                        print_mode: str = "print" ):
    
    agreement_type = f"corruption_agreement"
    
    explainable_model = ExplainableModel(model_name, data_name, n_classes)
    
    print("Calculating attributions...", flush=True)
    attributions = explainable_model.calculate_all_attributions(data_name=data_name, data_split=data_split)
    
    print("Calculating attributions for corrupted test...", flush=True)
    attributions_corrupted = explainable_model.calculate_all_attributions(data_name=data_name_corrupted, data_split=data_split)
    
    def internal(agreement_measure:str):
    
        pairs = list(zip(attributions, attributions_corrupted))
        names = [name for (name, _) in attributions]
        
        print(f"{agreement_type}/mean_{agreement_measure}_{model_name}_{data_name_corrupted}_{data_split}.csv")
        
        table_mean=pd.Series(index=names, dtype=float)
        table_var=pd.Series(index=names, dtype=float)
        
        if os.path.exists(f"{agreement_type}/mean_{agreement_measure}_{model_name}_{data_name_corrupted}_{data_split}.csv"):
            print(f"Agreement already calculated for {agreement_measure} on {model_name} {data_name_corrupted} {data_split}.", flush=True)
            
            #load from csv
            table_mean_loaded = pd.read_csv(f"{agreement_type}/mean_{agreement_measure}_{model_name}_{data_name_corrupted}_{data_split}.csv", index_col=0, header=0)
            table_var_loaded = pd.read_csv(f"{agreement_type}/var_{agreement_measure}_{model_name}_{data_name_corrupted}_{data_split}.csv", index_col=0, header=0)
        
            for name in names:
                if name in table_mean_loaded.index and name in table_var_loaded.index:
                    table_mean.loc[name] = table_mean_loaded.loc[name].iloc[0]
                    table_var.loc[name] = table_var_loaded.loc[name].iloc[0]
            
        best_dict = {}
        worst_dict = {}
        
        for (name_1, att_1), (name_2, att_2) in pairs:
            if att_1 is None or att_2 is None:
                continue
            
            print(f"Calculating batch agreement for ({name_1}, {name_2})... ", flush=True)
            
            #check if mean and var already calculated
            if not pd.isna(float(table_mean.loc[name_1])) and not pd.isna(float(table_var.loc[name_1])):
                print(f"Agreement already calculated for ({name_1}, {name_2}).", flush=True)
            else:    
                mean, var, agreement_per_image = batch_agreement(attributions_test_1=att_1, 
                                                                attributions_test_2=att_2, 
                                                                agreement_function=agreement_measure)
                table_mean.loc[name_1] = mean
                table_var.loc[name_1] = var
                
                if best_to_show > 0:
                    bests = nlargest(best_to_show, agreement_per_image, key=agreement_per_image.get)
                    best_dict[name_1] = bests
                    
                if worst_to_show > 0:
                    worsts = nsmallest(worst_to_show, agreement_per_image, key=agreement_per_image.get)
                    worst_dict[name_1] = worsts
            
            print(f"({name_1}, {name_2}): mean: {float(table_mean.loc[name_1]):.4f}, var: {float(table_var.loc[name_1]):.4f}", flush=True)
            
            
        print_agreements(agreement_measure,
                        print_mode, 
                        table_mean, table_var, 
                        model_name, data_name_corrupted, data_split, agreement_type)
        
        return best_dict, worst_dict
    
    best_worst = {}
    
    for measure in ["density", "l2", "SSIM"]:
        print(f"Calculating {measure} agreement...", flush=True)
        best, worst = internal(measure)
        
        best_worst[measure] = {"best": best, "worst": worst}
    
    return best_worst

def full_corruption_agreement():
    
    best_worst = {}
    
    for model_name in models:
        
        best_worst[model_name] = {}
    
        for (data_name, n_classes) in [("dermamnist",7), ("octmnist",4), ("pneumoniamnist",2)]:
            
            for corruption in ["contrast_down", "jpeg_compression", "speckle_noise"]:
        
                best_worst[model_name][data_name] = {}
                print(f"Calculating corruption agreement for {model_name} on {data_name} with corruption {corruption}...", flush=True)
                
                best_worst_run = corruption_agreement(model_name = model_name, n_classes=n_classes,
                                                data_name=data_name+"_224", data_split="test",
                                                data_name_corrupted=data_name+"_corrupted_"+corruption+"_224",
                                                best_to_show=0, worst_to_show=0, print_mode="csv")
                
                best_worst[model_name][data_name] = best_worst_run
    
    return best_worst

def full_corruption_accuracy():
    
    accuracies = {}
    
    for model_name in models:
    
        for (data_name, n_classes) in [("dermamnist",7), ("octmnist",4), ("pneumoniamnist",2)]:
            
            for corruption in ["contrast_down", "jpeg_compression", "speckle_noise"]:
        
                print(f"Calculating corruption accuracy for {model_name} on {data_name} with corruption {corruption}...", flush=True)
                
                accuracy, _ = ExplainableModel(model_name=model_name, 
                                 train_data_name=data_name+"_224", 
                                 n_classes=n_classes).test(data_name=data_name+"_corrupted_"+corruption+"_224", 
                                                           data_split="test", batch_size=256)
                print(f"Accuracy: {accuracy:.4f}", flush=True)
                accuracies[f"{model_name}_{data_name}_{corruption}"] = accuracy
    
    with open("corruption_accuracy.json", "w") as f:
        json.dump(accuracies, f, indent=4)
    print("Corruption accuracies saved to corruption_accuracy.json")