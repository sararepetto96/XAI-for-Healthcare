from typing import Tuple, Dict

from matplotlib import pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial.distance import jensenshannon

from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from scipy.signal import correlate2d
from pytorch_msssim import ssim
from .Metrics import consistency
from tqdm import tqdm

def attributions_check(attributions_a: np.ndarray, attributions_b: np.ndarray):
    assert attributions_a.shape == attributions_b.shape, "attributions should have same shape"
    assert attributions_a.ndim == 3, "attributions should be 3D numpy arrays (3 channels images)"
    assert attributions_a.shape[0] == 3, "attributions should have 3 channels"
    assert attributions_a.shape[1] == 224 and attributions_a.shape[2] == 224, \
        "attributions should have 224x224 resolution"  #224x224 resolution

def attributions_preprocessing(attributions: np.ndarray,
                                consider_before_aggregation: str,
                                aggregate_channels: str,
                                consider_after_aggregation: bool,
                                return_abs:bool,
                                min_max_normalization:bool) -> np.ndarray:

    assert consider_before_aggregation in ["both", "pos", "neg"], "Invalid consider before aggregation parameter. Choose between both, pos or neg"
    assert consider_after_aggregation in ["both", "pos", "neg"], "Invalid consider after aggregation parameter. Choose between both, pos or neg"
    assert aggregate_channels in ["abs", "sum", "mean", "no"], "Invalid aggregation method. Choose between abs, sum, mean or no"
    
    if consider_before_aggregation == "pos": #considering only positive attributions
        attributions = np.where(attributions<0, 0, attributions)
    elif consider_before_aggregation == "neg": #considering only negative attributions
        attributions = np.where(attributions>0, 0, attributions)
    elif consider_before_aggregation == "both": #considering both positive and negative attributions
        pass
    
    if aggregate_channels == "abs":
        attributions = np.sum(np.abs(attributions), axis=0)
    elif aggregate_channels == "sum":
        attributions = np.sum(attributions, axis=0)
    elif aggregate_channels == "mean":
        attributions = np.mean(attributions, axis=0)
    elif aggregate_channels == "no":
        pass
    
    if consider_after_aggregation == "pos": #considering only positive attributions
        attributions = np.where(attributions<0, 0, attributions)
    elif consider_after_aggregation == "neg": #considering only negative attributions
        attributions = np.where(attributions>0, 0, attributions)
    elif consider_after_aggregation == "both": #considering both positive and negative attributions
        pass
    
    if return_abs:
        attributions = np.abs(attributions)
    
    if min_max_normalization:
        attributions = (attributions - np.min(attributions)) / (np.max(attributions) - np.min(attributions) + 1e-8)
    
    return attributions
    
    
def top_k_attributions_indexes(attributions_a: np.ndarray, attributions_b: np.ndarray, top_k_percentage: float,
                                consider: str,
                                aggregate_channels: str) -> Tuple[np.ndarray,np.ndarray]:
    
    assert 0 < top_k_percentage <= 1, "percentage should be between 0 and 1"  #percentage
    
    attributions_check(attributions_a, attributions_b)
    
    attributions_a = attributions_preprocessing(attributions_a, 
                                                consider, aggregate_channels, True)
    attributions_b = attributions_preprocessing(attributions_b, 
                                                consider, aggregate_channels, True)
    
    attributions_a = attributions_a.squeeze()
    attributions_b = attributions_b.squeeze()

    top_k = int(attributions_a.flatten().shape[0] * top_k_percentage)

    a = np.argpartition(attributions_a.flatten(), -top_k)[-top_k:]  #top k indices
    b = np.argpartition(attributions_b.flatten(), -top_k)[-top_k:]  #top k indices
    
    return a,b

def convolve2D(image: np.ndarray, kernelDim:Tuple[int,int], stride:int, padding:int = 0, kernel_function:str = "OR")-> np.ndarray:
    
    assert kernel_function in ["OR", "MEAN"], "Invalid kernel function. Choose between OR or MEAN"
    
    assert len(image.shape) == 2, "Input image should be 2D"
    assert image.shape[0] == image.shape[1], "Input image should be square"
    assert image.shape[0]==224, "Input image should have 224x224 resolution"
    
    assert len(kernelDim) == 2, "Kernel dimensions should be 2D"
    assert stride > 0, "Stride should be greater than 0"
    assert padding >= 0, "Padding should be greater or equal to 0"
    
    # Add padding
    if padding > 0:
        #image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant', constant_values=0)
        
        image = np.concatenate([np.zeros((padding, image.shape[1]), dtype=image.dtype),
                                image, 
                                np.zeros((padding, image.shape[1]), dtype=image.dtype)], axis=0)
        
        image = np.concatenate([np.zeros((image.shape[0], padding), dtype=image.dtype),
                                image,
                                np.zeros((image.shape[0], padding), dtype=image.dtype)], axis=1)

    # Get the shape of the input image
    H, W = image.shape

    # Kernel dimensions
    kH, kW = kernelDim

    # Compute the output shape
    xOutput = (H - kH) // stride + 1
    yOutput = (W - kW) // stride + 1

    # Use as_strided to create sliding windows
    shape = (xOutput, yOutput, kH, kW)
    strides_image = image.strides
    strides = (stride * strides_image[0], stride * strides_image[1]) + strides_image
    windows = as_strided(image, shape=shape, strides=strides)
    
    if kernel_function == "OR":
        # Apply the kernel function (in this case, checking if any pixel in the window is 1)
        output:np.ndarray = np.any(windows, axis=(2, 3)).astype(np.uint8)
        
        #rescale the output to the original image size
        result = np.kron(output, np.ones((stride, stride), dtype=np.uint8))
        
        return result
    
    elif kernel_function == "MEAN":
        
        # Apply the kernel function (in this case, mean of the window)
        output:np.ndarray = np.mean(windows, axis=(2, 3)).astype(np.float32)
        
        #rescale the output to the original image size
        result = np.kron(output, np.ones((stride, stride), dtype=np.float32))
        
        return result
    else:
        raise ValueError("Invalid kernel function. Choose between OR or MEAN")

def calculate_importance_masks(aggregate_channels: str, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    if aggregate_channels == "no":
        mask_a = np.zeros((3,224,224), dtype=np.uint8).flatten()
        mask_b = np.zeros((3,224,224), dtype=np.uint8).flatten()
        
        mask_a[a] = 1
        mask_b[b] = 1
        
        mask_a = mask_a.reshape((3,224,224))
        mask_b = mask_b.reshape((3,224,224))
    else:
        mask_a = np.zeros((224,224), dtype=np.uint8).flatten()
        mask_b = np.zeros((224,224), dtype=np.uint8).flatten()
    
        mask_a[a] = 1
        mask_b[b] = 1
        
        mask_a = mask_a.reshape((224,224))
        mask_b = mask_b.reshape((224,224))
    
    return mask_a, mask_b

def image_convolution_agreement(attributions_a: np.ndarray, attributions_b: np.ndarray, top_k_percentage: float,
                    consider: str = "both",
                    aggregate_channels: str = "sum",
                    kernel_size: Tuple[int,int]= (1,1),
                    stride:int = 1, padding:int=0) -> Tuple[float, np.ndarray, np.ndarray]:

    top_a, top_b = top_k_attributions_indexes(attributions_a,attributions_b,top_k_percentage,consider,aggregate_channels)

    importance_mask_a, importance_mask_b = calculate_importance_masks(aggregate_channels, top_a, top_b)
    
    if aggregate_channels == "no":
        convolved_importance_mask_a = np.concatenate([ convolve2D(importance_mask_a[channel], kernel_size, stride, padding) for channel in [0, 1, 2] ], dim=0)
        convolved_importance_mask_b = np.concatenate([ convolve2D(importance_mask_b[channel], kernel_size, stride, padding) for channel in [0, 1, 2] ], dim=0)
        
    else:
        convolved_importance_mask_a = convolve2D(importance_mask_a, kernel_size, stride, padding)
        convolved_importance_mask_b = convolve2D(importance_mask_b, kernel_size, stride, padding)
    
    mega_pixel_mask_indexes_a = np.flatnonzero(convolved_importance_mask_a)
    mega_pixel_mask_indexes_b = np.flatnonzero(convolved_importance_mask_b)
    
    #Jaccard similarity
    jaccard = np.intersect1d(mega_pixel_mask_indexes_a, mega_pixel_mask_indexes_b).size / np.union1d(mega_pixel_mask_indexes_a, mega_pixel_mask_indexes_b).size

    return jaccard, convolved_importance_mask_a, convolved_importance_mask_b 

def density_creation(attributions: np.ndarray, epsilon:int, kernel_size: Tuple[int,int], stride:int, padding:int) -> np.ndarray:
    
    #convolve attributions: mean of the window
    convolved = convolve2D(attributions, kernel_size, stride, padding, kernel_function="MEAN")
    
    #create densities
    sum_ = np.sum(convolved, dtype=np.double)
    
    # (avoid division by zero)
    if sum_ == 0:
        convolved += epsilon
        sum_ = np.sum(convolved, dtype=np.double)
    
    density = convolved / sum_
    
    return density

def normalized_cross_correlation(I1:np.ndarray, I2:np.ndarray)->float:
    # Ensure inputs are numpy arrays
    I1 = np.array(I1, dtype=np.float64)
    I2 = np.array(I2, dtype=np.float64)
    
    # Compute the mean of each image
    mean_I1 = np.mean(I1)
    mean_I2 = np.mean(I2)
    
    # Center the images by subtracting the mean
    I1_centered = I1 #- mean_I1
    I2_centered = I2 #- mean_I2
    
    # Compute the cross-correlation using scipy's correlate2d
    correlation = correlate2d(I1_centered, I2_centered, mode='valid')
    
    # Compute the norms (standard deviations of the centered arrays)
    norm_I1 = np.sqrt(np.sum(I1_centered**2))
    norm_I2 = np.sqrt(np.sum(I2_centered**2))
    
    # Normalize the cross-correlation result
    ncc_value = correlation / (norm_I1 * norm_I2)
    
    return ncc_value[0][0]

def density_agreement(density_a: np.ndarray, density_b: np.ndarray, function_to_use:str="jensenshannon") -> float:
    
    if function_to_use == "jensenshannon":
        return 1-jensenshannon(density_a.flatten(), density_b.flatten())
    
    elif function_to_use == "wasserstein_1d":
        
        values = np.arange(1, (224*224)+1)
        
        return ((224*224) - wasserstein_distance(values, values, density_a.flatten(), density_b.flatten()))/(224*224)
    
    elif function_to_use == "wasserstein_2d":
        values = np.arange(1, (224)+1)
        support = [[v1,v2] for v1 in values for v2 in values]
        
        return wasserstein_distance_nd(support, support, density_a.flatten(), density_b.flatten())
    
    elif function_to_use == "cosine":
        return np.dot(density_a.flatten(), density_b.flatten()) / (np.linalg.norm(density_a.flatten()) * np.linalg.norm(density_b.flatten()))
    
    elif function_to_use == "normalized_cross_correlation":
        return (normalized_cross_correlation(density_a, density_b) +1) /2
    
    else:
        raise ValueError("Invalid function to use. Choose between jensenshannon, wasserstein_1d, wasserstein_2d, cosine or normalized_cross_correlation")

def image_density_agreement(attributions_a: np.ndarray, attributions_b: np.ndarray,
                        kernel_size: Tuple[int,int] = (3,3), stride:int=1, padding:int=0) -> float:
    
    epsilon = 1e-8
    
    density_a = density_creation(attributions_a, epsilon, kernel_size, stride, padding)
    density_b = density_creation(attributions_b, epsilon, kernel_size, stride, padding)
    
    da = density_agreement(density_a, density_b)
    if np.isnan(da):
        da = 1.0
    
    return float(da), density_a, density_b

def l2_distance(attributions_a: np.ndarray, attributions_b: np.ndarray) -> float:
    
    # Compute the L2 distance
    l2_distance = np.linalg.norm((attributions_a.flatten() - attributions_b.flatten()), ord=2)
    
    return float(l2_distance)

def SSIM(attributions_a: np.ndarray, attributions_b: np.ndarray) -> float:
    
    attributions_a = torch.from_numpy(attributions_a).unsqueeze(0).unsqueeze(0)
    attributions_b = torch.from_numpy(attributions_b).unsqueeze(0).unsqueeze(0)
    
    # Compute the SSIM
    ssim_value = ssim(attributions_a, attributions_b,data_range=1, size_average=False)
    
    return ssim_value.item()

def agreement_selector(attributions_a: np.ndarray, attributions_b: np.ndarray,
                        agreement_function:str, #agreement function to use: "density", "l2", "SSIM")
                        consider_before_aggregation: str="both", #consider before aggregation: "both", "pos", "neg"
                        aggregate_channels: str="sum", #aggregation method: "abs", "sum", "mean", "no"
                        consider_after_aggregation: str="pos", #consider after aggregation: "both", "pos", "neg"
                        kernel_size:Tuple[int, int]=(3,3), #kernel size for convolution
                        stride:int=1, #stride for convolution
                        padding:int=0 #padding for convolution
                        )-> float: #agreement value
    
    if aggregate_channels == "no":
        raise NotImplementedError("Not implemented yet")
    
    attributions_check(attributions_a, attributions_b)
    
    attributions_a = attributions_preprocessing(attributions_a, 
                                                consider_before_aggregation=consider_before_aggregation, aggregate_channels=aggregate_channels, 
                                                consider_after_aggregation=consider_after_aggregation, return_abs=False, min_max_normalization=True)
    attributions_b = attributions_preprocessing(attributions_b, 
                                                consider_before_aggregation=consider_before_aggregation, aggregate_channels=aggregate_channels, 
                                                consider_after_aggregation=consider_after_aggregation, return_abs=False, min_max_normalization=True)
    
    if agreement_function == "density":
        value, _, _ = image_density_agreement(attributions_a=attributions_a,
                                            attributions_b=attributions_b,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding)
        
    elif agreement_function == "l2":
        value = l2_distance(attributions_a=attributions_a,
                            attributions_b=attributions_b)
        
    elif agreement_function == "SSIM":
        value = SSIM(attributions_a=attributions_a,
                    attributions_b=attributions_b)
    else:
        raise ValueError("Invalid agreement function.")
    
    return value

def batch_agreement(attributions_test_1: Dict[str, np.ndarray], attributions_test_2: Dict[str, np.ndarray],
                    
                    agreement_function:str, #agreement function to use: "density", "l2", "SSIM"
                    consider_before_aggregation: str="both", #consider before aggregation: "both", "pos", "neg"
                    aggregate_channels: str="sum", #aggregation method: "abs", "sum", "mean", "no"
                    consider_after_aggregation: str="pos", #consider after aggregation: "both", "pos", "neg"
                    
                    # Density only parameters:
                    kernel_size:Tuple[int, int]=(3,3), #kernel size for convolution
                    stride:int=1, #stride for convolution
                    padding:int=0 #padding for convolution
                    )-> tuple[float, float, Dict[str, float]]: #mean agreement, variance agreement, agreement per image
    
    assert len(attributions_test_1) == len(attributions_test_2), "attributions should have same len"
    
    assert agreement_function in ["density", "l2", "SSIM"], "Invalid agreement function. Choose between convolution, density oe consistency"
    
    #attributions_test_1 = np.copy(attributions_test_1).item()
    #attributions_test_2 = np.copy(attributions_test_2).item()
    
    agreement_per_image = {}
    for image1, image2 in tqdm(zip(attributions_test_1.keys(), attributions_test_2.keys()), total=len(attributions_test_1), desc="Calculating agreement"):
        
        agreement_per_image[image1] = agreement_selector(attributions_a=attributions_test_1[image1],
                                        attributions_b=attributions_test_2[image2],
                                        agreement_function=agreement_function,
                                        consider_before_aggregation=consider_before_aggregation,
                                        aggregate_channels=aggregate_channels,
                                        consider_after_aggregation=consider_after_aggregation,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding)

    return float(np.average(list(agreement_per_image.values()))), float(np.var(list(agreement_per_image.values()))), agreement_per_image