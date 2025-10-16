import copy
from functools import partial
import json
import numbers
import random
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
import gc
from captum import attr
from .custom_XAI import Saliency_with_grad, GuidedBackProp_with_grad, GradCAM_with_grad,SmoothGrad,ScoreCAM_with_grad, GradCAM_plus_plus_with_grad,FinerCAM
from typing import Optional, Tuple, List, Dict
from adv_lib.utils.visdom_logger import VisdomLogger
from adv_lib.utils.projections import clamp_
import os
from timm.models import create_model
from multiprocessing import get_context

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from . import MedViT,ViT
from .MedDataset import MedDataset

from PIL.Image import Image

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .agreement import image_density_agreement, SSIM, l2_distance

import time

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, FinerWeightedTarget
from lxt.efficient import monkey_patch, monkey_patch_zennit
from lxt.efficient.patches import patch_method
import matplotlib.pyplot as plt

from lxt.efficient.patches import non_linear_forward, layer_norm_forward

from .patches import detach_qk

from torcheval.metrics.functional import multiclass_auroc, binary_auroc, multiclass_f1_score, binary_f1_score

class Normalizer(torch.nn.Module):
    def __init__(self,device: str ="cuda"):
        super(Normalizer, self).__init__()
        mean_const, std_const = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        mean = torch.as_tensor(mean_const)[None, :, None, None]
        std = torch.as_tensor(std_const)[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.device = device

    def forward(self, x):
        return x.sub(self.mean.to(self.device)).div(self.std.to(self.device))

class NormalizedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        super(NormalizedModel, self).__init__()
        self.model = model
        self.normalizer = Normalizer(device=device)

    def forward(self, x):
        x = self.normalizer(x)
        return self.model(x)
class ExplainableModel():
    
    @staticmethod
    def get_feature_mask(sub_cube_size=4) -> torch.Tensor:

        base = torch.ones(size=(sub_cube_size,), dtype=torch.long)

        if 224 % sub_cube_size != 0:
            raise ValueError("sub_cube_size must divide 224")

        sub_cubes_per_stride = 224 // sub_cube_size  #16

        j = 0
        stride_tensors = []
        for _ in range(sub_cubes_per_stride):

            base_tensors = []
            for _ in range(sub_cubes_per_stride):
                base_tensors.append(base * j)
                j += 1

            row_tensor = torch.concatenate(base_tensors, dim=0)

            stride_tensor = torch.cat([row_tensor[None, :] for _ in range(sub_cube_size)], dim=0)

            stride_tensors.append(stride_tensor)

        channel_tensor = torch.cat(stride_tensors, dim=0)

        feature_mask = torch.cat([channel_tensor[None, :, :] for _ in range(3)], dim=0)[None, :, :, :]

        return feature_mask
    
    def __call__(self, inp: Image, argmax:bool=True)-> torch.Tensor:
        
        t = ExplainableModel.build_transform()
        
        self.model.eval().cuda()
        
        with torch.inference_mode():
            out = self.model(t(inp).unsqueeze(0).cuda())
        
        self.model.eval().cpu()
        
        if argmax:
            return torch.argmax(torch.nn.Softmax(dim=0)(out[0]), dim=0).detach().cpu()
        else:
            return out.detach().cpu()
    
    def __get_attribution_path(self, data_name:str, data_split:str) -> str:
        return f"{self.model_name}_attributions/{self.train_data_name}/{data_name}_{data_split}/"
    
    def __save_attributions(self, algorithm: str, data_name:str, data_split:str, attributions: Dict[str, np.ndarray]):

        attributions_path = self.__get_attribution_path(data_name, data_split)
        os.makedirs(attributions_path, exist_ok=True)
        
        np.savez_compressed(os.path.join(attributions_path, f"{algorithm}.npz"), **attributions)

    def __load_attributions(self, algorithm: str, data_name:str, data_split:str) -> Dict[str, np.ndarray] | None:
        
        attributions_path = self.__get_attribution_path(data_name, data_split)

        path = os.path.join(attributions_path, f"{algorithm}.npz")
        if not os.path.exists(path):
            return None, attributions_path

        loaded_data = np.load(path)

        return {key: loaded_data[key] for key in loaded_data}, attributions_path
    
    @staticmethod
    def build_transform(input_size = 224):
        resize_im = input_size > 32

        t = []
        if resize_im:
            size = int((256 / 224) * input_size)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(input_size))

        t.append(transforms.ToTensor())
        #t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)
    
    def CAM_like_procedure(self, cam_method, input_tensor: torch.Tensor, target_classes: torch.Tensor, reshape_transform = None):
        
        if self.model_name == "MedVit":
            target_layers = self.__get_target_layer_GradCAM_MedViT()
        elif self.model_name == "vgg16":
            target_layers = self.model.model.features[-1]
        elif self.model_name == "densenet121":
            target_layers = self.model.model.features[-1]
        elif self.model_name == "resnet50":
            target_layers = self.model.model.layer4[-1]
        elif self.model_name == "convnext_base":
            target_layers = self.model.model.stages[-1].blocks[-1].conv_dw
        elif self.model_name == "custom_vit_base_patch16_224":
            target_layers = self.model.model.blocks[-1].norm1
        elif self.model_name == "custom_deit_base_patch16_224":
            target_layers = self.model.model.blocks[-1].norm1
            
        elif self.model_name == "custom_pit_b_224":
        
            target_layers = self.model.model.transformers[-1].blocks[-1].norm1
        else:
            raise ValueError(f"CAM not implemented for model {self.model_name}")
        
        #print(target_layers)

        if target_layers is None:
            raise ValueError(f"CAM target layer not found for model {self.model_name}")
        
        if cam_method is FinerCAM:
            target_classes = [FinerWeightedTarget(c, [i for i in range(self.n_classes) if i != c], alpha=1) for c in target_classes.cpu().tolist()]
        else:
            target_classes = [ClassifierOutputTarget(c) for c in target_classes.cpu().tolist()] 
        
        
        cam = cam_method(model=self.model, target_layers=[target_layers],reshape_transform=reshape_transform)
        attributions = cam(input_tensor.requires_grad_(), targets=target_classes, eigen_smooth=False)
        
        
        cam.activations_and_grads.release() 
        
        del cam
        gc.collect()
        torch.cuda.empty_cache()
        
        # Expand to (BATCH, 3, 224, 224) with zeros
        zeros = torch.zeros(attributions.size(0), 3, attributions.size(1), attributions.size(1), 
                            device=attributions.device, dtype=attributions.dtype)
        
        zeros[:, 0] = attributions
        attributions = zeros
        
        return attributions

    def __init__(self, model_name:str, train_data_name: str, n_classes: int):

        model_path = f"../models_training/{model_name}/output/{train_data_name}/checkpoint_best.pth"
        self.model_name = model_name
        self.train_data_name = train_data_name
        self.n_classes = n_classes
        
        self.__load_model(model_path, n_classes)
        
        self.available_algorithms = [
                                    "IntegratedGradients", 
                                    "Saliency", 
                                    "DeepLift", 
                                    "InputXGradient", 
                                    "GradCAM",
                                    "SmoothGrad",
                                    'GradCAM_plusplus',
                                    'FinerCAM',
                                    
                                    "lxt",
                                    'LibraGRAD'
                                        ]
        
        self.is_model_lxt_patched = False
        
        if self.model_name in ["MedVit", "custom_deit_base_patch16_224", "custom_vit_base_patch16_224","custom_pit_b_224"]:
            self.is_transformer = True
        else:
            self.is_transformer = False
        
    def __load_model(self, model_path: str, n_classes: int):
        
        print(f"loading:{model_path}")
        
        if self.model_name == "MedVit":
            model_name = "MedViT_large"
        else:
            model_name = self.model_name

        self.model : torch.nn.Module = create_model(model_name, num_classes=n_classes)  
            
        checkpoint_model = torch.load(model_path)["model"]
        
        self.model.load_state_dict(checkpoint_model, strict=True, assign=True)
        #self.model.eval()  # Switch back to evaluation mode
        
        self.model = NormalizedModel(self.model)
        self.model = self.model.eval()



    def test(self, data_name: str, data_split: str = "test", batch_size: int = 512):
        # ---- Load data ----
        data = ExplainableModel.load_data(data_name, data_split)
        loader = DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)

        # ---- Set model to eval ----
        self.model.eval().cuda()

        # ---- Dataset-specific configs ----
        num_classes_map = {'dermamnist_224': 7, 'octmnist_224': 4, 'pneumoniamnist_224': 2}
        auroc_fns = {'dermamnist_224': multiclass_auroc, 'octmnist_224': multiclass_auroc, 'pneumoniamnist_224': binary_auroc}
        f1_fns = {'dermamnist_224': multiclass_f1_score, 'octmnist_224': multiclass_f1_score, 'pneumoniamnist_224': binary_f1_score}

        num_classes = num_classes_map[data_name]
        auroc_fn = auroc_fns[data_name]
        f1_fn = f1_fns[data_name]

        # ---- Inference loop ----
        all_logits = []
        all_preds = []
        all_labels = []
        correct = 0
        total = 0

        with torch.inference_mode():
            for inputs, labels, _ in tqdm(loader, desc=f"Testing {data_name}"):
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1) if num_classes > 2 else torch.sigmoid(outputs)

                _, predicted = torch.max(outputs.data, dim=1)

                all_logits.append(probs.cpu())
                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                torch.cuda.empty_cache()

        # ---- Compute metrics ----
        accuracy = 100 * correct / total
        all_logits = torch.cat(all_logits, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # AUROC (requires probabilities)
        auroc = auroc_fn(
            **({"input": all_logits} if num_classes > 2 else {"input": all_preds}),
            target=all_labels,
            **({"num_classes": num_classes} if num_classes > 2 else {}),
            **({"average": "macro"} if num_classes > 2 else {})
        )

        # F1 Score (requires predicted classes)
        f1_score = f1_fn(
            input=all_preds,
            target=all_labels,
            **({"num_classes": num_classes} if num_classes > 2 else {}),
            **({"average": "macro"} if num_classes > 2 else {})
        )

        self.model.cpu()
    
        return accuracy, len(data), auroc, f1_score

    
    def __get_target_layer_GradCAM_MedViT(self) -> torch.nn.Module:
        
        
        for (i, m) in enumerate(self.model.model.modules()):
            
            if i == 1450:  #last LTB
                #print(m)                
                return m.conv

    def applyXAI(self, algorithm: str, input_tensor: torch.Tensor,
                target_classes: torch.Tensor, post_processing=False) -> torch.Tensor:
        
        assert algorithm in self.available_algorithms, f"Invalid algorithm {algorithm}, choose from {self.available_algorithms}"
        
        if algorithm == "lxt" or algorithm == "LibraGRAD":
            assert self.is_transformer, "lxt/LibraGRAD algorithm is only available for Transformers models!"
            assert (self.is_model_lxt_patched), "lxt/LibraGRAD algorithm requires the model to be patched with lxt monkey patching, please call patch_model_lxt() before using it"
        else:
            assert (not self.is_model_lxt_patched), f"{algorithm} doesn't need to be patched, please reinitialize object"
        
        input_tensor=input_tensor.cuda()
        target_classes=target_classes.cuda()
        
        if algorithm == "IntegratedGradients":

            method = attr.IntegratedGradients(self.model)
            attributions = method.attribute(input_tensor, target=target_classes, internal_batch_size=6)
            
        elif algorithm == "Saliency":

            method = Saliency_with_grad(self.model)
            attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
        
        elif algorithm == "DeepLift":

            method = attr.DeepLift(self.model)
            attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)
        
        elif algorithm == "InputXGradient":
            
            method = attr.InputXGradient(self.model)
            attributions = method.attribute(input_tensor.requires_grad_(), target=target_classes)

        elif algorithm == "GradCAM":
            
            attributions = self.CAM_like_procedure(GradCAM_with_grad, input_tensor, target_classes)
        
        elif algorithm == "lxt":
            
            self.model.zero_grad()
            input_tensor.requires_grad_().retain_grad()  # Retain gradients for input tensor: this is necessary for not leaf tensors
            
            output = self.model(input_tensor)
            
            for i in range(len(target_classes)):
                output[i,target_classes[i]].backward(retain_graph=True)
                
            # Convert grads to tensor
            grads = input_tensor.grad.clone()

            # Get relevance at *ANY LAYER* in your model. Simply multiply the gradient * activation!
            # here for the input embeddings:
            attributions = (grads * input_tensor)

        elif algorithm == "SmoothGrad":
            method = SmoothGrad(self.model)
            attributions = method.generate(x=input_tensor.requires_grad_(),index = target_classes)

        elif algorithm == "ScoreCAM":
            
            attributions = self.CAM_like_procedure(ScoreCAM_with_grad, input_tensor, target_classes)

        elif algorithm == "GradCAM_plusplus":
            
            attributions = self.CAM_like_procedure(GradCAM_plus_plus_with_grad, input_tensor, target_classes)

        elif algorithm == "FinerCAM":
            
            attributions = self.CAM_like_procedure(FinerCAM, input_tensor, target_classes)
        
        elif algorithm == "LibraGRAD":
            
            def reshape_transform(tensor, height=14, width=14):
                result = tensor[:, 1 :  , :].reshape(tensor.size(0),
                    height, width, tensor.size(2))

                # Bring the channels to the first dimension,
                # like in CNNs.
                result = result.transpose(2, 3).transpose(1, 2)
                return result
            
            if self.model_name == "MedVit":
                attributions = self.CAM_like_procedure(GradCAM_plus_plus_with_grad, input_tensor, target_classes)
                
            elif self.model_name =='custom_pit_b_224':
                attributions = self.CAM_like_procedure(GradCAM_plus_plus_with_grad, input_tensor, target_classes,
                                                       reshape_transform=lambda tensor : reshape_transform(tensor,height=8,width=8))
                
            else:
                attributions = self.CAM_like_procedure(GradCAM_plus_plus_with_grad, input_tensor, target_classes,
                                                       reshape_transform=lambda tensor : reshape_transform(tensor,height=14,width=14))
            
        else:
            raise ValueError(f"Invalid algorithm {algorithm}")

        if post_processing:
            
            #sum the attributions along the channels
            attributions = attributions.sum(dim=1, keepdim=True)
            
            #exclude negative attributions
            attributions[attributions<0]=0
            
            #min-max normalization
            attributions=attributions - attributions.min(dim=1)[0].min(dim=1)[0].min(dim=1)[0].reshape(-1,1,1,1)
            attributions = attributions/(attributions.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].reshape(-1,1,1,1) + 1e-8)
            
        else:
            attributions = attributions.cpu().detach()

        return attributions

    @staticmethod
    def visualize(fig: Figure, ax: Axes,
                    max_value:int|None=None, min_value:int|None=None,
                    image: np.ndarray | None = None,
                    attributions: np.ndarray | None = None,
                    density: np.ndarray | None = None) -> None:
        
        if attributions is not None and density is not None:
            raise ValueError("Only one of attributions or density must be provided")
        
        to_display = None
        if attributions is not None:
            assert attributions.shape[0]==attributions.shape[1] and attributions.shape[2]==3, "Invalid attributions shape, must be (H, W, 3)"
            attributions = np.abs(np.sum(attributions, axis=2))
            to_display = attributions
            
        if density is not None:
            assert density.shape[0]==density.shape[1], "Invalid density shape, must be (H, W)"
            to_display = density
        
        if image is not None:
            assert len(image.shape) == 3 and image.shape[0]==image.shape[1] and image.shape[2]==3, "Invalid image shape, must be (H, W, 3)"
            
            if to_display is None:
                ax.imshow(image)
            else:
                ax.imshow(np.mean(image, axis=2), cmap="gray")
        
        ax.axis("off")
        axis_separator = make_axes_locatable(ax)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        
        if to_display is not None:
            if max_value is None or min_value is None:
                raise ValueError("max_value and min_value must be provided when displaying attributions or density")
            heat_map = ax.imshow(
                to_display, #vmax=max_value, vmin=min_value,
                cmap="Blues", alpha=0.5
            )
            fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
            
            
    def patch_model(self, algorithm:str, dict_tuning: Optional[Dict[str, float]] = None):
        
        if algorithm != "lxt" and algorithm != "LibraGRAD":
            return
        
        if self.is_model_lxt_patched:
            print("Model already patched, skipping patching")
            return
        
        assert self.is_transformer, "lxt/LibraGRAD algorithm is only available for Transformers models!"
        
        self.model.cpu()

        monkey_patch_dict = {
            #non linear activations
            nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
            nn.Sigmoid: partial(patch_method, non_linear_forward, keep_original=True),
            
            #layer normalization
            nn.LayerNorm: partial(patch_method, layer_norm_forward),
            
            #Vit-like models specific patch
            ViT.CustomAttention : partial(patch_method, detach_qk, method_name="detach_qk" ,keep_original=True),
            
            #MedVit specific patch
            MedViT.E_MHSA: partial(patch_method, detach_qk, method_name="detach_qk", keep_original=True),
            MedViT.h_swish: partial(patch_method, non_linear_forward, keep_original=True),
        }
            
        
        monkey_patch(self.model, patch_map=monkey_patch_dict, verbose=True)
        
        if algorithm == "lxt":
            monkey_patch_zennit(verbose=True)
            
            # Define rules for the Conv2d and Linear layers using 'zennit'
            # LayerMapComposite maps specific layer types to specific LRP rule implementations
            from zennit.composites import LayerMapComposite
            import zennit.rules as z_rules
            
            if dict_tuning is None:
                # Default tuning parameters for different models
                if self.model_name == "MedVit":
                    dict_tuning = {
                        "conv_gamma": 0.25,
                        "lin_gamma": 0.25,
                        "pool_epsilon": 0.000001,
                        "norm_epsilon": 0.000001
                    }
                elif self.model_name in ["custom_deit_base_patch16_224", "custom_pit_b_224"]:
                    dict_tuning = {
                        "conv_gamma": 0.25,
                        "lin_gamma": 0.1,
                    }
                elif self.model_name == "custom_vit_base_patch16_224":
                    dict_tuning = {
                        "conv_gamma": 0.75,
                        "lin_gamma": 1.0,
                    }
                else:
                    raise ValueError(f"No default dict_tuning for {self.model_name} for patching, please provide dict_tuning")
            
            rules=[
                (torch.nn.Conv2d, z_rules.Gamma(gamma=dict_tuning["conv_gamma"])),
                (torch.nn.Linear, z_rules.Gamma(gamma=dict_tuning["lin_gamma"])),
            ]
            
            if self.model_name == "MedVit":
                rules += [(torch.nn.BatchNorm2d, z_rules.Epsilon(epsilon=dict_tuning["norm_epsilon"])),
                            (torch.nn.BatchNorm1d, z_rules.Epsilon(epsilon=dict_tuning["norm_epsilon"])),
                            (torch.nn.AvgPool1d, z_rules.Epsilon(epsilon=dict_tuning["pool_epsilon"])),
                            (torch.nn.AdaptiveAvgPool2d, z_rules.Epsilon(epsilon=dict_tuning["pool_epsilon"])),]
            
            #Need for tuning...
            zennit_comp = LayerMapComposite(rules)

            # Register the composite rules with the model
            zennit_comp.register(self.model)
        
        
        self.is_model_lxt_patched = True

    def explain_image(self, algorithm: str, image:Image | torch.Tensor, image_class:int, tuning_dict: Optional[Dict[str, float]] = None, to_transform=False) -> np.ndarray:
        
        if isinstance(image, torch.Tensor):
            assert image.shape == (3, 224, 224), "Invalid image shape, must be (3, 224, 224)"
        elif isinstance(image, Image):
            assert image.size == (224, 224), "Invalid image size, must be (224, 224)"
        else:
            raise ValueError("Invalid image type", type(image))
        
        if to_transform:
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage(mode="RGB")(image)
            t = self.build_transform()
            image_tensor = t(image).unsqueeze(0)
        else:
            image_tensor = image.unsqueeze(0)

        torch.cuda.empty_cache()
        
        self.patch_model(algorithm=algorithm, dict_tuning=tuning_dict)
        
        self.model.cuda()

        image_attribution = self.applyXAI(algorithm, image_tensor.cuda(), torch.Tensor([image_class]).int(), post_processing=True)
        #image_attribution = torch.zeros((1, 3, 224, 224), device="cuda")
        
        torch.cuda.empty_cache()
        self.model.cpu()
        
        if algorithm == "lxt" and self.is_model_lxt_patched:
            print("MODEL PATCHED, PLEASE USE ONLY LXT ALGORITHM AFTER PATCHING")

        if isinstance(image_attribution, torch.Tensor):
            image_attribution = image_attribution.cpu().squeeze().detach().numpy()
        
        return image_attribution
    
    @staticmethod
    def load_data(data_name:str, data_split:str = "test", to_transform:bool = True) -> MedDataset | None:
        data_path = f"../models_training/data/{data_name}/{data_split}"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path {data_path} does not exist!")
        
        if to_transform:
            return MedDataset(data_path, transform=ExplainableModel.build_transform())
        else:
            return MedDataset(data_path, transform=lambda x: transforms.ToTensor()(x))

    def explain_dataset(self, algorithm: str, data_name: str, data_split: str, batch_size: int,
                        new_process: bool = False) -> Dict[str, np.ndarray]:
        
        attributions, attributions_path = self.__load_attributions(algorithm, data_name, data_split)
        if attributions is not None:
            print(f"Attributions for {algorithm} already exist. ({attributions_path})", flush=True)
            return attributions

        data = ExplainableModel.load_data(data_name, data_split)
        
        self.patch_model(algorithm=algorithm)
        
        torch.cuda.empty_cache()
        self.model = self.model.cuda()

        med_loader = DataLoader(data, batch_size=batch_size, pin_memory=True, 
                                num_workers = 8, 
                                shuffle=False)
        attributions_tot = dict()

        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        restart_every = 5  # batches

        for i, (image_tensor, labels, names) in enumerate(tqdm(med_loader, desc=f"Explaining batch of images using {algorithm}")):
            if new_process:
                # Restart pool every N batches
                if i % restart_every == 0:
                    if pool:
                        pool.close()
                        pool.join()
                    pool = ctx.Pool(1)

                # Define wrapper function if needed externally
                r = pool.apply_async(self.applyXAI, args=(algorithm, image_tensor, torch.Tensor(labels)))
                attributions = r.get()
            else:
                attributions = self.applyXAI(algorithm, image_tensor, torch.Tensor(labels))

            for j in range(attributions.shape[0]):
                attributions_tot[names[j]] = attributions[j].numpy().astype(np.float32)

            del image_tensor, labels, names, attributions
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.1)  # helps prevent IPC buildup

        if pool:
            pool.close()
            pool.join()

        self.model.cpu()
        self.__save_attributions(algorithm, data_name, data_split, attributions_tot)
        
        if algorithm == "lxt" and self.is_model_lxt_patched:
            print("MODEL PATCHED, PLEASE USE ONLY LXT ALGORITHM AFTER PATCHING")
        
        return attributions_tot

    
    def calculate_all_attributions(self, data_name:str, data_split:str):
        
        # list of attributions to calculate with batch size optimal for each algorithm (cluster)
        def convolutional_attributions():
            return [
                
                ("IntegratedGradients", self.explain_dataset("IntegratedGradients", data_name, data_split, batch_size=16)),
                ("Saliency", self.explain_dataset("Saliency", data_name, data_split, batch_size=8, new_process=True)),
                ("DeepLift", self.explain_dataset("DeepLift", data_name, data_split, batch_size=4)),
                ("InputXGradient", self.explain_dataset("InputXGradient", data_name, data_split, batch_size=16)),
                
                ("GradCAM", self.explain_dataset("GradCAM", data_name, data_split, batch_size=4, new_process=True)),
                ("SmoothGrad", self.explain_dataset("SmoothGrad", data_name, data_split, batch_size=2, new_process=False)),
                ("GradCAM_plusplus", self.explain_dataset("GradCAM_plusplus", data_name, data_split, batch_size=4, new_process=True)),
                ("FinerCAM", self.explain_dataset("FinerCAM", data_name, data_split, batch_size=8, new_process=True)),
            ]
        
        def transformer_attributions():
            to_return=[
                ("lxt", self.explain_dataset("lxt", data_name, data_split, batch_size=4, new_process=False)),
                ("LibraGRAD", self.explain_dataset("LibraGRAD", data_name, data_split, batch_size=4, new_process=False)),
            ]
            
            return [x for x in to_return if x is not None]
        
        if self.is_transformer:
            if self.model_name == "MedVit":
                list_attributions = convolutional_attributions() + transformer_attributions()
            else:
                list_attributions = transformer_attributions()
        else:
            list_attributions = convolutional_attributions()         
        
        return list_attributions
    
    @staticmethod
    def load_adversarial_examples(algorithm:str, model_name: str, train_data_name: str, data_name: str, ε: int) -> Dict[str, np.ndarray]:
        folder_name = f"adversarial_examples/{model_name}/{train_data_name}/{data_name}_test"
        adv_example_file = f"{folder_name}/{algorithm}_{ε}.npz"
        
        if not os.path.exists(adv_example_file):
            raise FileNotFoundError(f"Adversarial examples file {adv_example_file} does not exist!")
        
        loaded_data = np.load(adv_example_file)
        return {key: loaded_data[key] for key in loaded_data}

    @staticmethod
    def load_adversarial_explanations(algorithm:str, model_name: str, train_data_name: str, data_name: str, ε: int) -> Dict[str, np.ndarray]:
        folder_name = f"adversarial_explanations/{model_name}/{train_data_name}/{data_name}_test"
        adv_expl_file = f"{folder_name}/{algorithm}_{ε}.npz"
        
        if not os.path.exists(adv_expl_file):
            raise FileNotFoundError(f"Adversarial explanations file {adv_expl_file} does not exist!")
        
        loaded_data = np.load(adv_expl_file)
        return {key: loaded_data[key] for key in loaded_data}
    
        
    def attack(self,
            data_name:str,
            algorithm: str,
            ε: int,
            expl_loss_function : str="topk",
            loss_function : str="ce",
            n_steps: int = 100,
            lr : int = 0.1,
            batch_size : int = 8,
            new_process: bool = False,
            dataset_subset_size=100,
            ) -> None:
        #save metrics only if default parameters are used
        default_params = (expl_loss_function == "topk" and loss_function == "ce" and n_steps == 100 and lr == 0.1 and dataset_subset_size == 100)
        
        if default_params:
            file_name = f"attack/{self.model_name}/{self.train_data_name}/{data_name}_test/"
            os.makedirs(file_name, exist_ok=True)
            
            file_name = file_name+f"{algorithm}_{ε}.json"
            if os.path.exists(file_name):
                print(f"File {file_name} already exists, skipping attack", flush=True)
                return
            
        #if adversarial examples and adversarial explanations already exist, skip attack
        adv_example_file = f"adversarial_examples/{self.model_name}/{self.train_data_name}/{data_name}_test/{algorithm}_{ε}.npz"
        adv_expl_file = f"adversarial_explanations/{self.model_name}/{self.train_data_name}/{data_name}_test/{algorithm}_{ε}.npz"
        
        if (not default_params) and os.path.exists(adv_example_file) and os.path.exists(adv_expl_file):
            print(f"Adversarial examples for {algorithm} already exist, skipping attack", flush=True)
            print(f"Adversarial explanations for {algorithm} already exist, skipping attack", flush=True)
            return
        
        
        if algorithm not in self.available_algorithms:
            raise ValueError(f"Invalid algorithm {algorithm}, choose from {self.available_algorithms}")
        
        self.patch_model(algorithm=algorithm)
    
        lst_L_2_exp = []
        lst_SSIM = []
        lst_agreement = []
        
        total_accuracy_original = 0
        total_accuracy_perturbed = 0
        
        batch_cross_entropy = []
        batch_explanation_loss = []
        batch_loss = []
        
        adversarial_examples = {}
        adversarial_explanations = {}

        samples = 0
        epsilon = ε
        ε = ε/255
        
        data = ExplainableModel.load_data(data_name, "test", to_transform=True)
        
        set_all_seeds(0)
        
        subset_indices = torch.randperm(len(data))[:dataset_subset_size]
        subset = Subset(data , subset_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=False)
        
        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        
        restart_every = 1  # batches
        
        if algorithm == "GuidedBackprop":
            restart_every = 1  # batches
        if algorithm == "Saliency":
            restart_every = 8
        if algorithm == "GradCAM":
            restart_every = 1
        if algorithm == "SmoothGrad":
            restart_every = 5
        if algorithm == "Grad_roolout":
            restart_every = 5
    
        for i, (inputs, labels, names) in enumerate(dataloader):
                
                torch.cuda.empty_cache()
                
                if new_process:
                    
                    # Restart pool every N batches
                    if i % restart_every == 0:
                        if pool:
                            pool.close()
                            pool.join()
                        pool = ctx.Pool(1)
                    
                    #Done to avoid problems with torch memory leaks
                    #https://github.com/pytorch/pytorch/issues/51978
                    #these two lines will spawn a new process and run the function in it and then return the result to the main process
                    #this way the memory is freed forcefully after the process is done
                    r=pool.apply_async(pgd_linf, args=(i, len(dataloader),inputs,labels,names,algorithm,ε,self,n_steps,lr,expl_loss_function,loss_function))
                    metrics, adv_inputs, adversarial_expl = r.get()
                else:
                    metrics, adv_inputs, adversarial_expl  = pgd_linf(i, len(dataloader), inputs, labels, names, algorithm, ε, self, n_steps, lr,
                                                                        expl_loss_function, loss_function)
                
                torch.cuda.empty_cache()
                    
                samples += 1
                
                lst_L_2_exp += metrics['l2']
                lst_agreement += metrics['agreement']
                lst_SSIM += metrics['SSIM']
                
                total_accuracy_original += metrics['accuracy_original']
                total_accuracy_perturbed += metrics['accuracy_perturbed']
                
                batch_cross_entropy += metrics['cross_entropy']
                batch_explanation_loss += metrics['explanation_loss']
                batch_loss += metrics['epoch_loss']
                
                # create a tensor of adversarial inputs
                for adv_input, adv_expl, name in zip(adv_inputs, adversarial_expl, names):
                    adv_input = adv_input.numpy()
                    adv_expl = adv_expl.numpy()
                    adversarial_examples[name] = adv_input.astype(np.float32)
                    adversarial_explanations[name] = adv_expl.astype(np.float32)

                self.model.zero_grad()

                del inputs, labels, names
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(0.5)  # helps prevent IPC buildup
        
        if pool:
            pool.close()
            pool.join()
                
        total_accuracy_perturbed = total_accuracy_perturbed/samples
        total_accuracy_original = total_accuracy_original/samples

        data = {
            'lst_L_2_exp': lst_L_2_exp,
            'total_accuracy_original': total_accuracy_original,
            'total_accuracy_perturbed': total_accuracy_perturbed,
            'lst_SSIM': lst_SSIM,
            'lst_agreement' : lst_agreement,
            'batch_cross_entropy' : batch_cross_entropy,
            'batch_explanation_loss' : batch_explanation_loss,
            'batch_loss' : batch_loss,
        }
        
        #save metrics only if default parameters are used
        if default_params:
            with open(file_name, "w") as results_file:
                json.dump(data, results_file, indent=4)
        
        #save adversarial examples and explanations
        folder_name= f"adversarial_examples/{self.model_name}/{self.train_data_name}/{data_name}_test"
        os.makedirs(folder_name, exist_ok=True)
        
        adv_example_file = f"{folder_name}/{algorithm}_{epsilon}.npz"
        if not os.path.exists(adv_example_file):
            np.savez_compressed(adv_example_file, **adversarial_examples)
            
        
        folder_name= f"adversarial_explanations/{self.model_name}/{self.train_data_name}/{data_name}_test"
        os.makedirs(folder_name, exist_ok=True)
        
        adv_example_file = f"{folder_name}/{algorithm}_{epsilon}.npz"
        if not os.path.exists(adv_example_file):
            np.savez_compressed(adv_example_file, **adversarial_explanations)
    
def pgd_linf(
            n_batch:int,
            total_batch:int,
            inputs: torch.Tensor,
            labels: torch.Tensor,
            names:List[str],
            algorithm: str,
            ε: float,
            explainableModel : ExplainableModel,
            n_steps: int,
            lr : int,
            expl_loss_function: str,
            loss_function: str,
            restarts: int = 1,
            callback: Optional[VisdomLogger] = None) -> torch.Tensor:
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = len(inputs)

        inputs = inputs.to(device)
        labels = labels.to(device)

        adv_inputs = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if isinstance(ε, numbers.Real):
            ε = torch.full_like(adv_found, ε, dtype=inputs.dtype)
    
        pgd_attack = partial(_pgd_linf, 
                            method=algorithm, explainableModel = explainableModel, n_steps=n_steps,
                            expl_loss_function = expl_loss_function,
                            loss_function = loss_function,
                            lr = lr, n_batch=n_batch, total_batch=total_batch)

        for i in range(restarts):
            original_expl, adv_found_run, adv_inputs_run, epoch_loss, cross_entropy, explanation_loss = pgd_attack(inputs=inputs[~adv_found], ε=ε[~adv_found])
            adv_inputs[~adv_found] = adv_inputs_run
            adv_found[~adv_found] = adv_found_run
        
            
            if callback:
                callback.line('success', i + 1, adv_found.float().mean())

            if adv_found.all():
                break
        
        # get the original and adversarial predictions
        model=explainableModel.model.eval().cuda()
        original_labels = model(inputs).argmax(1)
        adv_pred_labels = model(adv_inputs).argmax(1)
        
        # get the original and adversarial explanations
        adversarial_expl = explainableModel.applyXAI(algorithm=algorithm,input_tensor = adv_inputs, target_classes = adv_pred_labels, post_processing=True)
        
        # calculate the metrics
        accuracy_original = torch.sum(original_labels==labels)/len(inputs)
        accuracy_corrupted = torch.sum(adv_pred_labels == labels)/len(inputs)
        
        metrics={"accuracy_original": accuracy_original.detach().cpu().tolist(), 
                    "accuracy_perturbed": accuracy_corrupted.detach().cpu().tolist(),
                    "l2": [],
                    "SSIM": [],
                    "agreement": [],
                    "epoch_loss": epoch_loss.tolist(),
                    "cross_entropy": cross_entropy.tolist(),
                    "explanation_loss": explanation_loss.tolist()}
        
        for original, adversarial in zip(original_expl, adversarial_expl):
            
            original = original.cpu().detach().numpy()[0]
            adversarial = adversarial.cpu().detach().numpy()[0]
            
            metrics['l2'].append( l2_distance(original, adversarial) )
            metrics['SSIM'].append(  SSIM(original, adversarial) )
            metrics['agreement'].append(  image_density_agreement(original, adversarial)[0] )

        
        explainableModel.model.cpu()
        del original_expl
        gc.collect()
        torch.cuda.empty_cache()
        
        return metrics, adv_inputs.cpu().detach(), adversarial_expl.cpu().detach()

def _pgd_linf(
                n_batch: int,
                total_batch: int,
                inputs: torch.Tensor,
                ε: torch.Tensor,
                method: str,
                explainableModel : ExplainableModel,
                n_steps: int,
                lr : int,
                expl_loss_function: str,
                loss_function: str,
                new_process: bool=False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        model = explainableModel.model.eval().cuda()
        
        _loss_functions = {
            'ce': (partial(torch.nn.functional.cross_entropy, reduction='none'), 0.0001),
        }

        _expl_loss_functions = {
            'topk': (partial(topk),1),
        }

        expl_loss_func,expl_multiplier = _expl_loss_functions[expl_loss_function.lower()] 

        loss_func, multiplier = _loss_functions[loss_function.lower()]

        device = inputs.device
        batch_size = len(inputs)
        batch_view = lambda tensor: tensor.view(batch_size, *[1] * (inputs.ndim - 1))
        lower, upper = torch.maximum(-inputs, -batch_view(ε)), torch.minimum(1 - inputs, batch_view(ε))
        
        δ = torch.zeros_like(inputs, requires_grad=True)
        best_adv = inputs.clone()
        adv_found = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
        δ.data.uniform_(-1, 1).mul_(batch_view(ε))
        clamp_(δ, lower=lower, upper=upper)

        optimizer = torch.optim.Adam([δ], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min= lr/ 10)
        logits = model(inputs)
        pred_labels = logits.argmax(1)

        original_expl = explainableModel.applyXAI(algorithm=method, input_tensor = inputs, target_classes = pred_labels, post_processing=True) #maybe we can load them from file
        n = int(inputs.shape[2]//np.sqrt(10))
        target_expl = torch.zeros((1,224,224))
        target_expl[:,:n,:n] = torch.full((1,n,n),1)     
        
        target_expls = ((target_expl.unsqueeze(0), ) * batch_size)
        target=torch.concat((target_expls)).reshape(original_expl.shape).float()
        target=target.to('cuda:0')
    
        
        epoch_loss = []
        cross_entropy = []
        explanation_loss = []
        epoch_efficacy = []
        best_loss = float('inf')
        # Get multiprocessing context
        ctx = get_context("spawn")
        pool = None
        restart_every = 5  # batches
    
        for i in trange(n_steps, desc=f"steps...{n_batch+1}/{total_batch}", leave=False):
            
            
            optimizer.zero_grad()
            x_adv = inputs + δ
            adv_logits = model(x_adv)
            
            adv_expl = explainableModel.applyXAI(algorithm=method, input_tensor = x_adv, target_classes = pred_labels, post_processing=True)
            loss_expl = expl_multiplier*expl_loss_func(adv_expl,original_expl).float()
            cls_loss =  multiplier * loss_func(adv_logits, pred_labels)

            tot_loss = (cls_loss.cuda() + loss_expl.cuda()).mean()
        
            tot_loss.backward(retain_graph=True)
           
            optimizer.step()
            scheduler.step()
        
            is_clean = (adv_logits.argmax(1) == pred_labels)
        
            if tot_loss < best_loss:

                best_adv = torch.where(batch_view(is_clean), x_adv.detach(), best_adv)
                adv_found.logical_or_(is_clean)
                best_loss = tot_loss.item()

            clamp_(δ, lower=lower, upper=upper)
            tot_loss = tot_loss.detach().cpu().numpy()
            epoch_loss.append(tot_loss)
            cross_entropy.append((torch.sum(cls_loss)/len(inputs)).detach().cpu().numpy())
            explanation_loss.append((torch.sum(loss_expl)/len(inputs)).detach().cpu().numpy())

            assert (torch.min(δ) >= -ε[0]) & (torch.max(δ) <= ε[0])
            
            assert (torch.min(x_adv) >= 0) & (torch.max(x_adv) <= 1)

            del adv_expl,cls_loss,tot_loss,loss_expl,x_adv

            gc.collect()

            torch.cuda.empty_cache()
        
        if False: # True: #if you want to save the plots
            # Create a figure with 3 subgraphs (3 rows, 1 column)
            fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            epochs = list(range(1, len(epoch_loss) + 1))


            # Primo subplot
            axs[0].plot(epochs, epoch_loss, marker='o', color='blue')
            axs[0].set_title('Epoch Loss')
            axs[0].set_ylabel('Loss')
            axs[0].grid(True)

            # Secondo subplot
            axs[1].plot(epochs, cross_entropy, marker='s', color='green')
            axs[1].set_title('Cross Entropy')
            axs[1].set_ylabel('Loss')
            axs[1].grid(True)

            # Terzo subplot
            axs[2].plot(epochs, explanation_loss, marker='^', color='red')
            axs[2].set_title('Explanation Loss')
            axs[2].set_xlabel('Epoch')
            axs[2].set_ylabel('Loss')
            axs[2].grid(True)

            # Ottimizza layout e salva il file
            plt.tight_layout()
            plt.savefig('training_subplots.png')
            plt.close()
            
            
        epoch_loss = torch.tensor([torch.tensor(elem) for elem in epoch_loss])
        cross_entropy = torch.tensor([torch.tensor(elem) for elem in cross_entropy])
        explanation_loss = torch.tensor([torch.tensor(elem) for elem in explanation_loss])
        
        del δ
        gc.collect()
        torch.cuda.empty_cache()

        return original_expl, adv_found, best_adv, epoch_loss,cross_entropy,explanation_loss

def topk(adv_expl, original_expl):
    n = adv_expl.shape[0]
    adv_expl = adv_expl.reshape(n,-1)
    original_expl = original_expl.reshape(n,-1)
    l = adv_expl.shape[1]

    top_k_values, top_k_indices = torch.topk(original_expl, l//10, dim=1)
    top_k_values_adv = torch.gather(adv_expl, 1, top_k_indices)
    return top_k_values_adv.mean(dim=1)

def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # per configurazioni multi-GP
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False