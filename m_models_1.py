import copy
import inspect
import itertools
import json
import math
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as tud
from sklearn.ensemble import RandomForestRegressor
from torch.nn import MultiheadAttention, ModuleList, TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset

import m_layers as ml
from special_dicts import NestedDict, NestedDictList
from type_classes import *


class ModelUtils:
    is_frozen_ensemble = False
    keywords = ('config', 'keep_history')
    uuid = None
    uuid_copy = None

    def __init__(self, config=None):
        self.config = {} if config is None else config
        self.keep_history = NestedDict()
        self.uuid = uuid4()

    @classmethod
    def get_self_kwargs(cls, kwargs):
        return {k: v for k, v in kwargs.items() if k in cls.keywords}

    @classmethod
    def not_in_kwargs(cls, kwargs, tuples=()):
        return {k: v for k, v in kwargs.items() if k not in cls.keywords and not any(k in d for d in tuples)}

    @classmethod
    def get_name(cls):
        return cls.name

    @classmethod
    def get_parameter_names(cls):
        sig = inspect.signature(cls.__init__)
        return [name for name in sig.parameters.keys() if name != 'self']

    def get_parameters(self):
        sig = inspect.signature(self.__init__)
        self.params = {
            name: value.default if name not in locals() else locals()[name]
            for name, value in sig.parameters.items() if name != 'self'
        }

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
        self.uuid_copy = uuid4()
        return copy.deepcopy(self)
    
    @staticmethod
    def kw_fn(kwargs, key, default=None):
        if key in kwargs:
            value = kwargs[key]
            del kwargs[key]
            return value
        else:
            return default
    
    @classmethod
    def weight_init(cls, weight, mode=None, **kwargs):
        match mode:
            case None:
                pass
            case 'xavier_uniform':
                init.xavier_uniform_(weight, gain=cls.kw_fn(kwargs, 'gain', 1.0))
            case 'xavier':
                init.xavier_normal_(weight, gain=cls.kw_fn(kwargs, 'gain', 1.0))
            case 'xavier_normal':
                init.xavier_normal_(weight, gain=cls.kw_fn(kwargs, 'gain', 1.0))
            case 'kaiming_uniform':
                init.kaiming_uniform_(weight, nonlinearity=cls.kw_fn(kwargs, 'nonlinearity', 'relu'), **kwargs)
            case 'kaiming_normal':
                init.kaiming_normal_(weight, nonlinearity=cls.kw_fn(kwargs, 'nonlinearity', 'relu'), **kwargs)
            case 'uniform':
                init.uniform_(weight, cls.kw_fn(kwargs, 'a', -0.1), cls.kw_fn(kwargs, 'b', 0.1))
            case 'normal':
                init.normal_(weight, cls.kw_fn(kwargs, 'mean', 0.0), cls.kw_fn(kwargs, 'std', 0.001))
            case 'sparse':
                init.sparse_(weight, sparsity=cls.kw_fn(kwargs, 'sparsity', 0.1), std=cls.kw_fn(kwargs, 'std', 0.01))
            case 'orthogonal':
                init.orthogonal_(weight, gain=cls.kw_fn(kwargs, 'gain', 1.0))
            case 'zeros':
                init.zeros_(weight)
            case 'ones':
                init.ones_(weight)

            case 'adaptive_depth_scaled':
                # Custom 'secret sauce' initialization
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
                scale = np.sqrt(2.0 / (fan_in + fan_out)) * np.sqrt(3.0) * np.sqrt(depth_index / total_depth)
                init.uniform_(weight, -scale, scale)

            case 'special_init':
                ModelUtils.special_init(weight, depth_index=depth_index, total_depth=total_depth, **kwargs)

            case _:
                raise ValueError(f"Unsupported initialization mode: '{mode}'.")

    @staticmethod
    def special_init(weight, depth_index, total_depth, stability_factor=0.9, **kwargs):
        """
        A novel weight initialization method.
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



class ForestModel(RandomForestRegressor, ModelUtils):
    pass


class ForestModelDubious(nn.Module, ModelUtils):
    name = 'ForestModel'
    def __init__(self, input_size, output_size, hidden_size=None, num_layers=None):
        # super(ForestModelDubious, self).__init__()
        super().__init__(**ModelUtils.not_in_kwargs(kwargs))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forest = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.forest.fit(x, y)
        out = self.fc(out[:, -1, :])
        return out


# Define the LSTM model
class LSTMModel(nn.Module, ModelUtils):
    name = 'LSTM'
    def __init__(self, input_size, output_size, seq_len=None,  meta_model=None, meta_model_kwargs=None, hidden_size=None, num_layers=None, num_heads=None, init_hidden=True,
                 dropout_rate=None, volatility=0.1, lstm_init=None, fc_init=None, attention_init=None, use_all_timestamps=True, **kwargs):
        meta_kw_names = meta_model.parameters_names if meta_model is not None else ()
        super().__init__(**ModelUtils.not_in_kwargs(kwargs, meta_kw_names))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.output_size = output_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_heads = num_heads
        self.init_hidden = init_hidden
        self.dropout_rate = dropout_rate
        self.lstm_init = lstm_init
        
        self.fc_init = fc_init
        self.attention_init = attention_init

        self.volatility = volatility
        self.use_all_timestamps = use_all_timestamps


        
        if hidden_size is None:
            hidden_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self._init_lstm_weights(self.lstm, mode=lstm_init, dict_inp={'volatility': volatility})

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        # assert hidden_size % num_heads == 0, "hidden_dim must be divisible by num_heads"
        if num_heads is not None:
            self.attention = MultiheadAttention(hidden_size, num_heads, batch_first=True)
            ModelUtils._custom_init_attention_weights(self.attention, mode=attention_init)


        if meta_model is None:
            self.meta_model = None
        else:
            if isinstance(meta_model_kwargs, (list, tuple)):
                meta_model_kwargs = dict(meta_model_kwargs)
            self.meta_model_kwargs = meta_model_kwargs if meta_model_kwargs is not None else {}
            meta_model_args = (hidden_size, output_size)
            self.meta_model = meta_model(*meta_model_args, **self.meta_model_kwargs, **kwargs)


        if hidden_size is not None and self.meta_model is None:
            if self.use_all_timestamps:
                self.fc = nn.Linear(hidden_size*self.seq_len, output_size)
            else:
                self.fc = nn.Linear(hidden_size, output_size)
            ModelUtils._init_fc_weights(self.fc, mode=fc_init)
        else:
            self.fc = None

    def forward(self, x):
        if self.hidden_size is not None and self.init_hidden:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
        else:
            out, _ = self.lstm(x)
        if self.dropout is not None:
            out = self.dropout(out)
        if self.num_heads is not None:
            out, _ = self.attention(out, out, out)
        if self.hidden_size is not None:
            if self.use_all_timestamps:
                out = self.fc(out.reshape(out.shape[0], -1))
            else:
                out = self.fc(out[:, -1, :])
            return out
        else:
            out = out[:, -1, :]
        return out


class LSTMModel_old(nn.Module, ModelUtils):
    name = 'LSTM'
    def __init__(self, input_size, output_size, hidden_size=None, num_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.output_size = output_size
        self.hidden_size = hidden_size

        if hidden_size is None:
            hidden_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # assert hidden_size % num_heads == 0, "hidden_dim must be divisible by num_heads"
        if hidden_size is not None:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def get_parameters(self):
        return {'hidden_size': self.hidden_size, 'num_layers': self.num_layers}

# Meta-Model (Neural Network on top of LSTMs)
class MetaModel(nn.Module, ModelUtils):
    def __init__(self, input_size, output_size, hidden_size=12, squeeze_last=True):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.squeeze_last = squeeze_last

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        if self.squeeze_last:
            out = out.squeeze(-1)
        return out


class Ensemble(nn.Module, ModelUtils):
    name = 'Ensemble'
    def __init__(self, models, freeze=True, is_frozen_ensemble=True, meta_model=MetaModel, meta_model_kwargs=None,
                 meta_model_set_output_size=True, meta_model_squeeze_last=False, to_cpu=False,  **kwargs):
        # super(Ensemble, self).__init__()
        super().__init__(**ModelUtils.not_in_kwargs(kwargs, meta_model.parameters_names))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
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



class TransformerTimeSeries(nn.Module, ModelUtils):
    name = 'TransformerTimeSeries'
    def __init__(self, input_size, output_size, num_layers=2, dropout=0.1, num_heads=12, **kwargs):
        # super(TransformerTimeSeries, self).__init__()
        super().__init__(**ModelUtils.not_in_kwargs(kwargs))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.num_features = input_size
        self.output_dim = output_size
        self.model_type = 'Transformer'
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_heads = num_heads
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_size, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.encoder = nn.Linear(input_size, input_size)
        self.decoder = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            mask = self._generate_square_subsequent_mask(len(x)).to(x.device)
            self.src_mask = mask

        x = self.encoder(x)  # shape: (batch_size, sequence_length, num_features)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output[:, -1, :])
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AdvancedMLP1(nn.Module, ModelUtils):
    name = 'AdvancedMLP1'

    def __init__(self, input_size, output_size, hidden_sizes=(32, 16), dropout_rate=None, use_attention=True,
                 return_attention=False, use_residual=False, squeeze_last=False, weight_init_mode=None, **kwargs):
        super().__init__(**ModelUtils.not_in_kwargs(kwargs))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.input_size = input_size if isinstance(input_size, Number) else math.prod(input_size)
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, (tuple, list)) else (hidden_sizes,)
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.return_attention = return_attention
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.attention_weights = nn.ParameterList()
        self.residual_layers = nn.ModuleList()
        self.squeeze_last = squeeze_last
        self.weight_init_mode = weight_init_mode

        last_size = self.input_size
        for idx, hidden_size in enumerate(self.hidden_sizes):
            linear_layer = nn.Linear(last_size, hidden_size)
            # Apply weight initialization here for the linear layer
            ModelUtils._init_fc_weights(linear_layer, mode=weight_init_mode)

            layer_components = [
                linear_layer,
                ml.PermuteLayer(1, -1),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU()
            ]
            if self.dropout_rate is not None:
                layer_components.append(nn.Dropout(self.dropout_rate))
            layer_components.append(ml.PermuteLayer(1, -1))
            self.layers.append(nn.Sequential(*layer_components))

            if self.use_attention:
                # Adding attention mechanism and initializing weights
                attention_weight = nn.Parameter(torch.randn(hidden_size, 1))
                ModelUtils._custom_init_attention_weights(attention_weight, mode=1)  # Assuming a specific mode or adjust as needed
                self.attention_weights.append(attention_weight)

            if self.use_residual and idx > 0:
                # Adding residual connection if not the first layer and enabled
                adaptive_layer = nn.Linear(last_size, hidden_size)
                ModelUtils._init_fc_weights(adaptive_layer, mode=weight_init_mode)  # Initialize weights for the adaptive layer
                self.residual_layers.append(adaptive_layer)

            last_size = hidden_size

        self.output_layer = nn.Linear(last_size, self.output_size)
        # Initialize weights for the output layer
        ModelUtils._init_fc_weights(self.output_layer, mode=weight_init_mode)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        attention_scores = []
        residual = x  # Initialize residual with the input

        for idx, (layer, attn_weight) in enumerate(zip(self.layers, self.attention_weights)):
            if self.use_residual and idx > 0:
                adjusted_residual = self.residual_layers[idx - 1](residual)
            else:
                adjusted_residual = 0

            x = layer(x)

            if self.use_residual and idx > 0:
                x += adjusted_residual
            residual = x

            if self.use_attention:
                # Applying feature-based attention
                attention_score = F.softmax(torch.matmul(x, attn_weight.T), dim=1)
                attention_scores.append(attention_score)
                x = x * attention_score

        x = self.output_layer(x)
        if self.squeeze_last:
            x = x.squeeze(-1)
        return (x, attention_scores) if self.use_attention and self.return_attention else x


class AdvancedMLP2(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(32, 16), dropout_rate=None,
                 use_attention=False, return_attention=False, use_residual=False, num_heads=1,
                 weight_init_mode=None, **kwargs):
        super().__init__(**ModelUtils.not_in_kwargs(kwargs))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.input_size = input_size if isinstance(input_size, Number) else math.prod(input_size)
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes if isinstance(hidden_sizes, (tuple, list)) else (hidden_sizes,)
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.return_attention = return_attention
        self.use_residual = use_residual
        self.num_heads = num_heads
        self.layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        self.residual_layers = nn.ModuleList()

        last_size = self.input_size
        for idx, hidden_size in enumerate(self.hidden_sizes):
            linear_layer = nn.Linear(last_size, hidden_size)
            if weight_init_mode is not None:
                self._init_weights(linear_layer, mode=weight_init_mode)

            components = [linear_layer, nn.BatchNorm1d(hidden_size), nn.LeakyReLU()]
            if self.dropout_rate is not None:
                components.append(nn.Dropout(self.dropout_rate))

            self.layers.append(nn.Sequential(*components))

            # Configuring multi-head attention for each layer if enabled
            if self.use_attention:
                attention_layer = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
                self.attention_modules.append(attention_layer)

            # Adding residual connections if applicable
            if self.use_residual and idx > 0:
                adaptive_layer = nn.Linear(last_size, hidden_size)
                if weight_init_mode is not None:
                    self._init_weights(adaptive_layer, mode=weight_init_mode)
                self.residual_layers.append(adaptive_layer)

            last_size = hidden_size

        self.output_layer = nn.Linear(last_size, self.output_size)
        if weight_init_mode is not None:
            self._init_weights(self.output_layer, mode=weight_init_mode)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        attention_scores = []
        residual = x

        for idx, (layer, attention_layer) in enumerate(zip(self.layers, self.attention_modules)):
            if self.use_residual and idx > 0:
                adjusted_residual = self.residual_layers[idx - 1](residual)

            x = layer(x)

            if self.use_residual and idx > 0:
                x += adjusted_residual
            residual = x

            if self.use_attention:
                # Applying multi-head attention
                x = x.unsqueeze(1)  # Add sequence dimension
                attn_output, attn_output_weights = attention_layer(x, x, x)
                x = attn_output.squeeze(1)  # Remove sequence dimension
                attention_scores.append(attn_output_weights)

        x = self.output_layer(x)
        if self.squeeze_last:
            x = x.squeeze(-1)
        return (x, attention_scores) if self.use_attention and self.return_attention else x

    def _init_weights(self, module, mode):
        if mode == 'custom':
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class FutureMLP(nn.Module):
    """
    A model combining a 1D convolutional layer, a Capsule Layer, and a fully connected layer.
    """
    def __init__(self, input_size: int, output_size: int, num_capsules: int = 10, out_channels: int = 32, hidden_size: int = 64, num_iterations: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.capsule_layer = CapsuleLayer(num_capsules=num_capsules, num_route_nodes=out_channels, in_channels=out_channels, out_channels=hidden_size, num_iterations=num_iterations)
        self.fc = nn.Linear(num_capsules * hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # Reshape for Conv1D
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)  # Adjust shape for capsule layer
        x = self.capsule_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
        x = self.fc(x)
        return x


class CNNTimeSeries(nn.Module, ModelUtils):
    name = 'CNNTimeSeries'

    def __init__(self, input_size, output_size, seq_len=None, out_channels=(32, 64), kernel_sizes=(3, 5),
                 strides=(1, 1), paddings=(1, 2), num_linear=1, dropout=0., use_batch_norm=False, permute=False, **kwargs):
        super().__init__(**ModelUtils.not_in_kwargs(kwargs))
        ModelUtils.__init__(self, **ModelUtils.get_self_kwargs(kwargs))
        self.input_size = input_size if permute else seq_len
        self.output_size = output_size
        self.seq_len = seq_len
        self.in_channels = seq_len if permute else input_size
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.num_linear = num_linear
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.permute = permute

        self.layers = nn.ModuleList()
        if permute:
            self.layers.append(ml.PermuteLayer(-2, -1))
        current_channels = self.in_channels
        layer_output_size = self.input_size

        for out_channel, kernel_size, stride, padding in zip(out_channels, kernel_sizes, strides, paddings):
            conv = nn.Conv1d(in_channels=current_channels, out_channels=out_channel, kernel_size=kernel_size,
                             stride=stride, padding=padding)
            ModelUtils.weight_init(conv.weight, mode='kaiming_normal', nonlinearity='relu')  # He initialization
            self.layers.append(conv)
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(out_channel))
            self.layers.append(nn.ReLU())
            current_channels = out_channel
            layer_output_size = (layer_output_size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        self.layers.append(nn.Flatten())
        self.dropout = nn.Dropout(p=dropout)

        # Setting up linear layers
        linear_input_size = current_channels * layer_output_size
        self.linears = nn.ModuleList()

        for i in range(num_linear - 1):
            linear = nn.Linear(linear_input_size, 32)
            ModelUtils.weight_init(conv.weight, mode='xavier_normal', nonlinearity='relu')
            nn.init.xavier_normal_(linear.weight)  # Xavier initialization
            self.linears.append(linear)
            self.linears.append(nn.ReLU())
            self.linears.append(nn.Dropout(p=dropout))
            linear_input_size = 32

        # Final linear layer without ReLU and Dropout after it
        final_linear = nn.Linear(linear_input_size, output_size)
        nn.init.xavier_normal_(final_linear.weight)
        self.linears.append(final_linear)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        for linear in self.linears:
            x = linear(x)

        return x


class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(GatedConvolution, self).__init__()
        self.sigmoid = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation)
        self.tanh = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.sigmoid_activation = nn.Sigmoid()
        self.tanh_activation = nn.Tanh()

    def forward(self, x):
        return self.sigmoid_activation(self.sigmoid(x)) * self.tanh_activation(self.tanh(x))

