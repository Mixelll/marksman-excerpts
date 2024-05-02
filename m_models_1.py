import math
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as tud
import torch.nn.functional as F

import m_layers as ml

import sklearn as sk

from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torch.nn import MultiheadAttention
from torch.nn import ModuleList

from itertools import product
from type_classes import *


class ModelUtils:
    is_frozen_ensemble = False
    keywords = ('config',)

    def __init__(self, config=None):
        self.config = {} if config is None else config

    @classmethod
    def get_name(cls):
        return cls.name


    def get_parameters(self):
        raise NotImplementedError

    def get_name_and_parameters(self):
        return {'name': self.get_name(), **self.get_parameters()}

    def description(self):
        pass

    def describe(self):
        return {**self.description(), **self.get_name_and_parameters()}

    def _get_name_with_parameters_str(self):
        return f'{self.get_name()}({", ".join(map(lambda x: "=".join(map(str, x)), self.get_parameters().items()))})'

    def str(self):
        return self._get_name_with_parameters_str()

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def weight_init(weight, mode=None, **kwargs):
        match mode:
            case None:
                pass
            case 'xavier_uniform':
                init.xavier_uniform_(weight, gain=kwargs.get('gain', 1.0))
            case 'xavier':
                init.xavier_normal_(weight, gain=kwargs.get('gain', 1.0))
            case 'xavier_normal':
                init.xavier_normal_(weight, gain=kwargs.get('gain', 1.0))
            case 'kaiming_uniform':
                init.kaiming_uniform_(weight, nonlinearity=kwargs.get('nonlinearity', 'relu'))
            case 'kaiming_normal':
                init.kaiming_normal_(weight, nonlinearity=kwargs.get('nonlinearity', 'relu'))
            case 'uniform':
                init.uniform_(weight, kwargs.get('a', -0.1), kwargs.get('b', 0.1))
            case 'normal':
                init.normal_(weight, kwargs.get('mean', 0.0), kwargs.get('std', 0.001))
            case 'sparse':
                init.sparse_(weight, sparsity=kwargs.get('sparsity', 0.1), std=kwargs.get('std', 0.01))
            case 'orthogonal':
                init.orthogonal_(weight, gain=kwargs.get('gain', 1.0))
            case 'zeros':
                init.zeros_(weight)
            case 'ones':
                init.ones_(weight)

            case 'adaptive_depth_scaled':
                # Custom 'secret sauce' initialization
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
                scale = np.sqrt(2.0 / (fan_in + fan_out)) * np.sqrt(3.0) * np.sqrt(depth_index / total_depth)
                init.uniform_(weight, -scale, scale)

            case 'experimental':
                ModelUtils.experimental(weight, depth_index=depth_index, total_depth=total_depth, **kwargs)

            case _:
                raise ValueError(f"Unsupported initialization mode: '{mode}'.")

    @staticmethod
    def experimental(weight, depth_index, total_depth, stability_factor=0.9, **kwargs):
        """
        A novel weight initialization method aimed at corporate-level performance.
        It considers layer depth, stability criteria, and aims to balance the
        spectral norm across layers.

        :param weight: Weight tensor to be initialized.
        :param depth_index: Index of the current layer in the network depth.
        :param total_depth: Total number of layers in the network.
        :param stability_factor: A factor to ensure stability in signal propagation.
        """
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        depth_scale = np.sqrt(2.0) * np.sqrt(depth_index / total_depth)
        spectral_scale = stability_factor / np.sqrt(fan_in)

        # Combine scales for final weight initialization
        scale = depth_scale * spectral_scale
        init.uniform_(weight, -scale, scale)

        # Optionally apply spectral normalization to enforce the stability criterion
        if kwargs.get('apply_spectral_norm', False):
            nn.utils.parametrizations.spectral_norm(weight)

    @staticmethod
    def spectral_norm_based_initialization(weight, threshold=0.95, activation='relu'):
        """
        Initializes the weight tensor using a Spectral Norm-based approach for controlling the Lipschitz constant.

        Args:
            weight (torch.Tensor): Weight tensor to initialize.
            threshold (float): The desired threshold for the spectral norm of the weight tensor.
            activation (str): Type of activation function, which could influence the initial scaling factor.
        """
        # Step 1: Initial Variance-Scaling Initialization
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        if activation == 'relu':
            scale = 2.0
        else:  # Use Xavier-like initialization for other activations
            scale = 1.0
        std_dev = math.sqrt(scale / float(fan_in))
        nn.init.normal_(weight, 0.0, std_dev)

        # Step 2: Spectral Norm Adjustment
        u, s, v = torch.svd(weight)
        spectral_norm = torch.max(s)
        if spectral_norm > threshold:
            # Scale down weights to meet the spectral norm threshold
            weight.data = weight.data * (threshold / spectral_norm)


    @staticmethod
    def optimized_variance_scaling(weight, layer_index=1, total_layers=1, activation='relu', for_attention=False):
        """
        Applies Optimized Variance Scaling initialization.

        Args:
            weight (torch.Tensor): Weight tensor to initialize.
            layer_index (int): Index of the current layer.
            total_layers (int): Total number of layers in the model.
            activation (str): Type of activation function used after this layer.
            for_attention (bool): Whether this initialization is for an attention mechanism.
        """
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        depth_factor = math.sqrt(layer_index / total_layers)  # Depth-aware adjustment

        # Adjust variance based on activation function
        if activation == 'relu':
            scale = 2.0
        elif activation == 'leaky_relu':
            scale = math.sqrt(2.0 / (1 + 0.01 ** 2))
        else:  # Xavier-like adjustment for other activations
            scale = 1.0

        variance = scale * depth_factor / fan_in
        std_dev = math.sqrt(variance)

        # Custom adjustment for attention mechanisms
        if for_attention:
            std_dev /= 2  # Tighten the distribution for finer control over attention scores

        init.normal_(weight, 0.0, std_dev)

        if for_attention:
            bias = torch.empty_like(weight).fill_(0.1)  # Encourage early exploration
            return bias  # Return bias if needed for manual assignment

    @staticmethod
    def _init_fc_weights(fc, mode='xavier_uniform', **kwargs):
        """
        Initializes weights of a fully connected layer using the specified mode.
        :param fc: Fully connected layer whose weights are to be initialized.
        :param mode: Initialization mode.
        :param kwargs: Additional keyword arguments for specific initialization modes.
        """
        ModelUtils.weight_init(fc.weight, mode=mode, **kwargs)
        if fc.bias is not None:
            bias_value = kwargs.get('bias', 0.0)  # Default bias initialization value
            fc.bias.data.fill_(bias_value)

    @staticmethod
    def _init_lstm_weights(lstm, mode=1, dict_inp=None):
        if dict_inp is None:
            dict_inp = {}
        match mode:
            case 1:
                for name, param in lstm.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            case 2:
                for name, param in lstm.named_parameters():
                    if 'weight_ih' in name:
                        # Scale adjustment based on volatility
                        scale = np.sqrt(3.0 / (param.size(0) * dict_inp['volatility']))
                        init.uniform_(param, -scale, scale)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        # Higher forget gate bias
                        init.constant_(param, 0)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1)
    @staticmethod
    def _custom_init_attention_weights(attention, mode=None):
        # Initialize with a bias towards recent time steps
        match mode:
            case 1:
                if isinstance(attention, nn.parameter.Parameter):
                    init.xavier_uniform_(attention)
                else:
                    for name, param in attention.named_parameters():
                        if 'in_proj_weight' in name:
                            init.xavier_uniform_(param)
                        elif 'bias' in name:
                            init.constant_(param, 0.01)



class Ensemble(nn.Module, ModelUtils):
    name = 'Ensemble'
    parameters_names = ['models', 'is_frozen_ensemble', 'meta_model', 'meta_model_kwargs', 'meta_model_set_output_size', 'meta_model_squeeze_last', 'to_cpu']
    def __init__(self, models, freeze=True, is_frozen_ensemble=True, meta_model=MetaModel, meta_model_kwargs=None,
                 meta_model_set_output_size=True, meta_model_squeeze_last=False, to_cpu=False,  **kwargs):
        # super(Ensemble, self).__init__()
        super().__init__(**{k: v for k, v in kwargs.items() if k not in ModelUtils.keywords and k not in meta_model.parameters_names})
        ModelUtils.__init__(self, **{k: v for k, v in kwargs.items() if k in ModelUtils.keywords})
        if meta_model_kwargs is None:
            meta_model_kwargs = {}
        elif isinstance(meta_model_kwargs, (list, tuple)):
            meta_model_kwargs = dict(meta_model_kwargs)
        self.is_frozen_ensemble = is_frozen_ensemble
        self.models = nn.ModuleList(models)
        if freeze:
            for param in self.models.parameters():
                param.requires_grad = False
        num_models = len(models)
        input_dim = models[0].output_size  # Assuming all models have the same output_size
        meta_model_args = [(num_models, input_dim)]
        if meta_model_set_output_size:
            meta_model_args.append(1)
        self.meta_model = meta_model(*meta_model_args, squeeze_last=meta_model_squeeze_last,  **meta_model_kwargs,  **kwargs)


    def forward(self, x):
        predictions = torch.cat([model(x).unsqueeze(-1) for model in self.models], dim=-1)
        output = self.meta_model(predictions)
        if output.dim() >= 3:
            output = output.squeeze(-1)
        if output.dim() == 1:
            output = output.unsqueeze(-1)
        return output

    def prepare_meta_dataloader(self, dataloader, batch_size=None, shuffle=True, to_cpu=False):
        if batch_size is None:
            batch_size = dataloader.batch_size
        self.eval()  # Ensure the model is in evaluation mode
        device = next(self.parameters()).device  # Get the device of the model
        meta_inputs = []
        meta_targets = []

        with torch.no_grad():  # No gradient needed
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move data to the same device as the model
                predictions = torch.cat([model(inputs).unsqueeze(-1) for model in self.models], dim=-1)
                if to_cpu:
                    predictions = predictions.cpu()
                    targets = targets.cpu()
                meta_inputs.append(predictions)  # Move predictions back to CPU if necessary
                meta_targets.append(targets)  # Move targets back to CPU if necessary

        # Concatenate all batches
        meta_inputs = torch.cat(meta_inputs, dim=0)
        meta_targets = torch.cat(meta_targets, dim=0)

        # Create TensorDataset and DataLoader for meta model training
        meta_dataset = TensorDataset(meta_inputs, meta_targets)
        meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=shuffle)

        return meta_dataloader

    def get_parameters(self):
        return {'meta_model': self.meta_model, 'meta_model_kwargs': self.meta_model_kwargs, 'is_frozen_ensemble': self.is_frozen_ensemble}

