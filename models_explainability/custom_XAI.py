#!/usr/bin/env python3
from typing import Any, Callable, Tuple, Union, List, cast, Optional
from torch import Tensor

import torch
from captum._utils.typing import TargetType
from captum.attr import Saliency, GuidedBackprop, NoiseTunnel

from captum.attr._core.noise_tunnel import NoiseTunnelType, SUPPORTED_NOISE_TUNNEL_TYPES

from captum._utils.common import _run_forward

from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import torch.nn.functional as F
import gc


import cv2
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
import random 
import os

import ttach as tta

from .ActivationsAndGradients import ActivationsAndGradients 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

from captum._utils.common import (
    _expand_and_update_additional_forward_args,
    _expand_and_update_baselines,
    _expand_and_update_feature_mask,
    _expand_and_update_target,
    _format_tensor_into_tuples,
    _is_tuple,
)

from captum.attr._utils.common import _validate_noise_tunnel_type



def compute_gradients_with_graph(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs, create_graph=True)
    return grads

class Saliency_with_grad(Saliency):
    r"""
    This class is a modification of the original Saliency class from captum library.
    The modification is that the gradients are computed with the graph enabled.
    """
    
    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Saliency.__init__(self, forward_func)
        self.gradient_func = compute_gradients_with_graph
        
class GuidedBackProp_with_grad(GuidedBackprop):
    r"""
    This class is a modification of the original GuidedBackProp class from captum library.
    The modification is that the gradients are computed with the graph enabled.
    """
    
    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        GuidedBackprop.__init__(self, forward_func)
        self.gradient_func = compute_gradients_with_graph


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
        detach: bool = True,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.detach = detach
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform, self.detach)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> torch.Tensor:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
  

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> torch.Tensor:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = torch.maximum(cam, torch.zeros(1, device=cam.device))
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: torch.Tensor) -> torch.Tensor:
        cam_per_target_layer = torch.cat(cam_per_target_layer, dim=1)
        cam_per_target_layer = torch.clamp(cam_per_target_layer, min=0)
        result = torch.mean(cam_per_target_layer, dim=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class GradCAM_with_grad(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM_with_grad,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
            detach=False)
        #self.activations_and_grads = ActivationsAndGradients(self.model, self.target_layers, self.reshape_transform, self.detach)
        

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
    
        if len(grads.shape) == 4:
            #return np.mean(grads, axis=(2, 3))
            return grads.mean(dim=(2, 3)) #np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 5:
            
            return grads.mean(dim=(2, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
        
def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - torch.min(img)
        img = img / (1e-7 + torch.max(img))
        if target_size is not None:
            img = F.interpolate(img.float().unsqueeze(0).unsqueeze(0), 
                            size=target_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        result.append(img)
    result = torch.stack(result, dim=0)

    return result


class ScoreCAM_with_grad(BaseCAM):

    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            ScoreCAM_with_grad,
            self).__init__(
            model = model,
            target_layers = target_layers,
            reshape_transform = reshape_transform,
            compute_input_gradient = True,
            uses_gradients = True,
            detach = False)
    
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> torch.Tensor:
        
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads).squeeze(-1)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
      
        return cam
    
    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool
    ) -> torch.Tensor:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]
            cam = self.get_cam_image(input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = torch.maximum(cam, torch.zeros(1, device=cam.device))
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer
    
    def aggregate_multi_layers(self, cam_per_target_layer: torch.Tensor) -> torch.Tensor:
        cam_per_target_layer = torch.cat(cam_per_target_layer, dim=1)
        cam_per_target_layer = torch.clamp(cam_per_target_layer, min=0)
        result = torch.mean(cam_per_target_layer, dim=1)
        return scale_cam_image(result)
    

    def get_cam_weights(self, input_tensor, target_layer, targets, activations, grads):
        """
        Calcola i CAM weights in modo differenziabile ed efficiente usando torch.autograd.grad.
        """

        upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
        activation_tensor = (
            torch.from_numpy(activations).to(self.device)
            if isinstance(activations, np.ndarray)
            else activations.to(self.device)
        )

        upsampled = upsample(activation_tensor)
        maxs = upsampled.amax(dim=(-1, -2), keepdim=True)
        mins = upsampled.amin(dim=(-1, -2), keepdim=True)
        norm_upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

        scores = []

        for idx, (target, feature_maps) in enumerate(zip(targets, norm_upsampled)):
            class_scores = []

            for c in range(feature_maps.size(0)):
                # Prepara il weighted input
                weighted_input = input_tensor[idx:idx+1] * feature_maps[c:c+1]
                weighted_input.requires_grad_(True)

                # Forward pass
                output = self.model(weighted_input)

                # Calcola score target (supponiamo sia scalare)
                score = target(output)

                # Usiamo autograd per calcolare il gradiente rispetto all'input
                grad = torch.autograd.grad(
                    outputs=score,
                    inputs=weighted_input,
                    retain_graph=True,      # Non accumuliamo grafo extra
                    create_graph=True        # Manteniamo differenziabilità a monte
                )[0]

                # Usiamo lo score (o potenzialmente un summary del grad) come peso
                class_scores.append(score)  # oppure class_scores.append(grad.mean()) per alternative

                # Cleanup
                del grad, score, output, weighted_input
                gc.collect()
                torch.cuda.empty_cache()
                print(c)
                
            class_scores_tensor = torch.stack(class_scores)  # [C]
            scores.append(class_scores_tensor)

            del class_scores
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Processed sample {idx}")

        scores = torch.stack(scores)  # [N, C]
        weights = F.softmax(scores, dim=-1)

        return weights
    


class XGradCAM_with_grad(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            XGradCAM_with_grad,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
            detach=False)


    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        """
        Calcola i pesi CAM (differenziabili) da attivazioni e gradienti.
        Args:
            input_tensor: input batch [B, C, H, W]
            target_layer: unused here (placeholder for compatibilità)
            target_category: unused here (dipende dal tuo `target` in altri casi)
            activations: tensor [B, C, H, W]
            grads: tensor [B, C, H, W]
        Returns:
            weights: tensor [B, C] (differenziabile)
        """
        eps = 1e-7

        # Calcola somma delle attivazioni per ogni canale
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # Evita divisione per zero
        norm = grads * activations / (sum_activations + eps)  # [B, C, H, W]

        # Somma spaziale per ottenere i pesi
        weights = norm.sum(dim=(2, 3))  # [B, C]

        return weights

            
class GradCAM_plus_plus_with_grad(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM_plus_plus_with_grad,
            self).__init__(
            model,
            target_layers,
            reshape_transform,
            detach=False)
        
 
        
    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        """
        Grad-CAM++ weight computation in PyTorch (differentiable).
        Args:
            input_tensor: torch.Tensor [B, C, H, W]
            target_layers: unused here
            target_category: unused here
            activations: torch.Tensor [B, C, H, W]
            grads: torch.Tensor [B, C, H, W]
        Returns:
            weights: torch.Tensor [B, C]
        """

        eps = 1e-6
        grads_power_2 = (grads ** 2)                   # [B, C, H, W]
        grads_power_3 = grads_power_2 * grads          # [B, C, H, W]
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        denominator = 2 * grads_power_2 + grads_power_3 * sum_activations + eps  # [B, C, H, W]
        aij = grads_power_2 / denominator                                          # [B, C, H, W]

        # Azzeriamo aij dove il gradiente è esattamente zero
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))  # [B, C, H, W]

        # Grad-CAM++: ReLU sui gradienti e pesatura con aij
        weights = torch.relu(grads) * aij                         # [B, C, H, W]
        weights = weights.sum(dim=(2, 3))                         # [B, C]

        return weights
       
class FinerCAM:
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module], reshape_transform: Callable = None, base_method=GradCAM_with_grad):
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients
        self.detach = False
        #self.base_cam.detach = False
        self.activations_and_grads = ActivationsAndGradients(self.base_cam.model, self.base_cam.target_layers, self.base_cam.reshape_transform, self.detach)
        

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    
    def forward(self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module] = None,
            eigen_smooth: bool = False,
            alpha: float = 1,
            comparison_categories: List[int] = [1, 2, 3],
            target_idx: int = None
            ) -> torch.Tensor:

        input_tensor = input_tensor.to(self.base_cam.device)

        if self.compute_input_gradient:
            input_tensor.requires_grad_(True)

        outputs = self.base_cam.activations_and_grads(input_tensor)

        if self.uses_gradients:
            self.base_cam.model.zero_grad()

            loss = sum([target(output) for target, output in zip(targets, outputs)])

            # Questo mantiene la connessione all'input_tensor
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=input_tensor,
                retain_graph=True,
                create_graph=True,
                only_inputs=True
            )[0]

            self.input_grads = grads
        

        cam_per_layer = self.base_cam.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        # cam_per_layer deve contenere solo tensori PyTorch, non .detach() o .numpy()
        final_cam = self.base_cam.aggregate_multi_layers(cam_per_layer)

        return final_cam  # Tensor collegato all’input

    




    



        

    

        



class NoiseTunnel_with_grad(NoiseTunnel):
    
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        nt_type: str = "smoothgrad",
        nt_samples: int = 5,
        nt_samples_batch_size: int = None,
        stdevs: Union[float, Tuple[float, ...]] = 1.0,
        draw_baseline_from_distrib: bool = False,
        **kwargs: Any,
    ) -> Union[
        Union[
            Tensor,
            Tuple[Tensor, Tensor],
            Tuple[Tensor, ...],
            Tuple[Tuple[Tensor, ...], Tensor],
        ]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            nt_type (str, optional): Smoothing type of the attributions.
                        `smoothgrad`, `smoothgrad_sq` or `vargrad`
                        Default: `smoothgrad` if `type` is not provided.
            nt_samples (int, optional): The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `nt_samples` is not provided.
            nt_samples_batch_size (int, optional): The number of the `nt_samples`
                        that will be processed together. With the help
                        of this parameter we can avoid out of memory situation and
                        reduce the number of randomly generated examples per sample
                        in each batch.
                        Default: None if `nt_samples_batch_size` is not provided. In
                        this case all `nt_samples` will be processed together.
            stdevs    (float or tuple of float, optional): The standard deviation
                        of gaussian noise with zero mean that is added to each
                        input in the batch. If `stdevs` is a single float value
                        then that same value is used for all inputs. If it is
                        a tuple, then it must have the same length as the inputs
                        tuple. In this case, each stdev value in the stdevs tuple
                        corresponds to the input with the same index in the inputs
                        tuple.
                        Default: `1.0` if `stdevs` is not provided.
            draw_baseline_from_distrib (bool, optional): Indicates whether to
                        randomly draw baseline samples from the `baselines`
                        distribution provided as an input tensor.
                        Default: False
            **kwargs (Any, optional): Contains a list of arguments that are passed
                        to `attribution_method` attribution algorithm.
                        Any additional arguments that should be used for the
                        chosen attribution method should be included here.
                        For instance, such arguments include
                        `additional_forward_args` and `baselines`.

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution with
                        respect to each input feature. attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            - **delta** (*float*, returned if return_convergence_delta=True):
                        Approximation error computed by the
                        attribution algorithm. Not all attribution algorithms
                        return delta value. It is computed only for some
                        algorithms, e.g. integrated gradients.
                        Delta is computed for each input in the batch
                        and represents the arithmetic mean
                        across all `nt_samples` perturbed tensors for that input.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> ig = IntegratedGradients(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Creates noise tunnel
            >>> nt = NoiseTunnel(ig)
            >>> # Generates 10 perturbed input tensors per image.
            >>> # Computes integrated gradients for class 3 for each generated
            >>> # input and averages attributions across all 10
            >>> # perturbed inputs per image
            >>> attribution = nt.attribute(input, nt_type='smoothgrad',
            >>>                            nt_samples=10, target=3)
        """

        def add_noise_to_inputs(nt_samples_partition: int) -> Tuple[Tensor, ...]:
            if isinstance(stdevs, tuple):
                assert len(stdevs) == len(inputs), (
                    "The number of input tensors "
                    "in {} must be equal to the number of stdevs values {}".format(
                        len(inputs), len(stdevs)
                    )
                )
            else:
                assert isinstance(
                    stdevs, float
                ), "stdevs must be type float. " "Given: {}".format(type(stdevs))
                stdevs_ = (stdevs,) * len(inputs)
            return tuple(
                add_noise_to_input(input, stdev, nt_samples_partition).requires_grad_()
                if self.is_gradient_method
                else add_noise_to_input(input, stdev, nt_samples_partition)
                for (input, stdev) in zip(inputs, stdevs_)
            )

        def add_noise_to_input(
            input: Tensor, stdev: float, nt_samples_partition: int
        ) -> Tensor:
            # batch size
            bsz = input.shape[0]

            # expand input size by the number of drawn samples
            input_expanded_size = (bsz * nt_samples_partition,) + input.shape[1:]

            # expand stdev for the shape of the input and number of drawn samples
            stdev_expanded = torch.tensor(stdev, device=input.device).repeat(
                input_expanded_size
            )

            # draws `np.prod(input_expanded_size)` samples from normal distribution
            # with given input parametrization
            # FIXME it look like it is very difficult to make torch.normal
            # deterministic this needs an investigation
            noise = torch.normal(0, stdev_expanded)
            return input.repeat_interleave(nt_samples_partition, dim=0) + noise

        def update_sum_attribution_and_sq(
            sum_attribution: List[Tensor],
            sum_attribution_sq: List[Tensor],
            attribution: Tensor,
            i: int,
            nt_samples_batch_size_inter: int,
        ) -> None:
            bsz = attribution.shape[0] // nt_samples_batch_size_inter
            attribution_shape = cast(
                Tuple[int, ...], (bsz, nt_samples_batch_size_inter)
            )
            if len(attribution.shape) > 1:
                attribution_shape += cast(Tuple[int, ...], tuple(attribution.shape[1:]))

            attribution = attribution.view(attribution_shape)
            current_attribution_sum = attribution.sum(dim=1, keepdim=False)
            current_attribution_sq = torch.sum(attribution**2, dim=1, keepdim=False)

            sum_attribution[i] = (
                current_attribution_sum
                if not isinstance(sum_attribution[i], torch.Tensor)
                else sum_attribution[i] + current_attribution_sum
            )
            sum_attribution_sq[i] = (
                current_attribution_sq
                if not isinstance(sum_attribution_sq[i], torch.Tensor)
                else sum_attribution_sq[i] + current_attribution_sq
            )

        def compute_partial_attribution(
            inputs_with_noise_partition: Tuple[Tensor, ...], kwargs_partition: Any
        ) -> Tuple[Tuple[Tensor, ...], bool, Union[None, Tensor]]:
            # smoothgrad_Attr(x) = 1 / n * sum(Attr(x + N(0, sigma^2))
            # NOTE: using __wrapped__ such that it does not log the inner logs

            attributions = attr_func.__wrapped__(  # type: ignore
                self.attribution_method,  # self
                inputs_with_noise_partition
                if is_inputs_tuple
                else inputs_with_noise_partition[0],
                **kwargs_partition,
            )
            delta = None

            if self.is_delta_supported and return_convergence_delta:
                attributions, delta = attributions

            is_attrib_tuple = _is_tuple(attributions)
            attributions = _format_tensor_into_tuples(attributions)

            return (
                cast(Tuple[Tensor, ...], attributions),
                cast(bool, is_attrib_tuple),
                delta,
            )

        def expand_partial(nt_samples_partition: int, kwargs_partial: dict) -> None:
            # if the algorithm supports targets, baselines and/or
            # additional_forward_args they will be expanded based
            # on the nt_samples_partition and corresponding kwargs
            # variables will be updated accordingly
            _expand_and_update_additional_forward_args(
                nt_samples_partition, kwargs_partial
            )
            _expand_and_update_target(nt_samples_partition, kwargs_partial)
            _expand_and_update_baselines(
                cast(Tuple[Tensor, ...], inputs),
                nt_samples_partition,
                kwargs_partial,
                draw_baseline_from_distrib=draw_baseline_from_distrib,
            )
            _expand_and_update_feature_mask(nt_samples_partition, kwargs_partial)

        def compute_smoothing(
            expected_attributions: Tuple[Union[Tensor], ...],
            expected_attributions_sq: Tuple[Union[Tensor], ...],
        ) -> Tuple[Tensor, ...]:
            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad:
                return expected_attributions

            if NoiseTunnelType[nt_type] == NoiseTunnelType.smoothgrad_sq:
                return expected_attributions_sq

            vargrad = tuple(
                expected_attribution_sq - expected_attribution * expected_attribution
                for expected_attribution, expected_attribution_sq in zip(
                    expected_attributions, expected_attributions_sq
                )
            )

            return cast(Tuple[Tensor, ...], vargrad)

        def update_partial_attribution_and_delta(
            attributions_partial: Tuple[Tensor, ...],
            delta_partial: Tensor,
            sum_attributions: List[Tensor],
            sum_attributions_sq: List[Tensor],
            delta_partial_list: List[Tensor],
            nt_samples_partial: int,
        ) -> None:
            for i, attribution_partial in enumerate(attributions_partial):
                update_sum_attribution_and_sq(
                    sum_attributions,
                    sum_attributions_sq,
                    attribution_partial,
                    i,
                    nt_samples_partial,
                )
            if self.is_delta_supported and return_convergence_delta:
                delta_partial_list.append(delta_partial)

        return_convergence_delta: bool
        return_convergence_delta = (
            "return_convergence_delta" in kwargs and kwargs["return_convergence_delta"]
        )
        
        nt_samples_batch_size = (
            nt_samples
            if nt_samples_batch_size is None
            else min(nt_samples, nt_samples_batch_size)
        )

        nt_samples_partition = nt_samples // nt_samples_batch_size

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = _format_tensor_into_tuples(inputs)  # type: ignore

        _validate_noise_tunnel_type(nt_type, SUPPORTED_NOISE_TUNNEL_TYPES)

        kwargs_copy = kwargs.copy()
        expand_partial(nt_samples_batch_size, kwargs_copy)

        attr_func = self.attribution_method.attribute

        sum_attributions: List[Union[None, Tensor]] = []
        sum_attributions_sq: List[Union[None, Tensor]] = []
        delta_partial_list: List[Tensor] = []

        for _ in range(nt_samples_partition):
            inputs_with_noise = add_noise_to_inputs(nt_samples_batch_size)
            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = compute_partial_attribution(inputs_with_noise, kwargs_copy)

            if len(sum_attributions) == 0:
                sum_attributions = [None] * len(attributions_partial)
                sum_attributions_sq = [None] * len(attributions_partial)

            update_partial_attribution_and_delta(
                cast(Tuple[Tensor, ...], attributions_partial),
                cast(Tensor, delta_partial),
                cast(List[Tensor], sum_attributions),
                cast(List[Tensor], sum_attributions_sq),
                delta_partial_list,
                nt_samples_batch_size,
            )

        nt_samples_remaining = (
            nt_samples - nt_samples_partition * nt_samples_batch_size
        )
        if nt_samples_remaining > 0:
            inputs_with_noise = add_noise_to_inputs(nt_samples_remaining)
            expand_partial(nt_samples_remaining, kwargs)
            (
                attributions_partial,
                is_attrib_tuple,
                delta_partial,
            ) = compute_partial_attribution(inputs_with_noise, kwargs)

            update_partial_attribution_and_delta(
                cast(Tuple[Tensor, ...], attributions_partial),
                cast(Tensor, delta_partial),
                cast(List[Tensor], sum_attributions),
                cast(List[Tensor], sum_attributions_sq),
                delta_partial_list,
                nt_samples_remaining,
            )

        expected_attributions = tuple(
            [
                cast(Tensor, sum_attribution) * 1 / nt_samples
                for sum_attribution in sum_attributions
            ]
        )
        expected_attributions_sq = tuple(
            [
                cast(Tensor, sum_attribution_sq) * 1 / nt_samples
                for sum_attribution_sq in sum_attributions_sq
            ]
        )
        attributions = compute_smoothing(
            cast(Tuple[Tensor, ...], expected_attributions),
            cast(Tuple[Tensor, ...], expected_attributions_sq),
        )

        delta = None
        if self.is_delta_supported and return_convergence_delta:
            delta = torch.cat(delta_partial_list, dim=0)

        return self._apply_checks_and_return_attributions(
            attributions, is_attrib_tuple, return_convergence_delta, delta
        )
    

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-11



import torch
import torch.nn.functional as F
from tqdm import tqdm




class SmoothGrad:
    def __init__(self, model, cuda=False, stdev_spread=0.15, n_samples=5, magnitude=True):
        self.model = model.eval()
        self.cuda = cuda
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude
        if self.cuda:
            self.model.cuda()

    def generate(self, x, index=None):
        if self.cuda:
            x = x.cuda()
        x = x.requires_grad_()  # Needed for gradient tracking

        B, C, H, W = x.shape
        stdev = self.stdev_spread * (x.max() - x.min())

        # Repeat input B x n_samples times → shape: (B * n_samples, C, H, W)
        x_expanded = x.unsqueeze(1).repeat(1, self.n_samples, 1, 1, 1)  # (B, n_samples, C, H, W)
        x_expanded = x_expanded.view(B * self.n_samples, C, H, W)
        noise = torch.randn_like(x_expanded) * stdev
        x_noisy = x_expanded + noise
        #x_noisy.requires_grad = True

        # Forward pass
        outputs = self.model(x_noisy)  # shape: (B * n_samples, num_classes)
        num_classes = outputs.shape[1]

        # Expand target indices
        if index is None:
            # infer most probable class per sample (batch-aware)
            with torch.no_grad():
                base_outputs = self.model(x)
                index = base_outputs.argmax(dim=1)  # (B,)
        if isinstance(index, int):
            index = torch.full((B,), index, device=x.device, dtype=torch.long)
        index = index.repeat_interleave(self.n_samples)  # (B * n_samples,)

        # One-hot target
        one_hot = F.one_hot(index.long(), num_classes=torch.tensor(num_classes)).float()  # (B * n_samples, num_classes)
        target_scores = (outputs * one_hot).sum(dim=1)  # (B * n_samples,)

        # Backward
        grads = torch.autograd.grad(
            outputs=target_scores.sum(),
            inputs=x_noisy,
            create_graph=True
        )[0]  # shape: (B * n_samples, C, H, W)

        # Reshape and average
        grads = grads.view(B, self.n_samples, C, H, W)
        if self.magnitude:
            grads = grads ** 2
        avg_grads = grads.mean(dim=1)  # (B, C, H, W)

        return avg_grads

