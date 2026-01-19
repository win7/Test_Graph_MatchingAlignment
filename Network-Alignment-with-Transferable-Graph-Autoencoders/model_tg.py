import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math


from torch_geometric.nn import GINConv, GINEConv
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential

# GINConv
"""
forward(
    x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
    edge_index: Union[Tensor, SparseTensor], 
    size: Optional[Tuple[int, int]] = None) → Tensor
"""
class TGAE_Encoder_GIN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()

		hidden_layers = n_layers - 2
		self.in_proj = nn.Linear(input_dim, hidden_dim[0])
		self.convs = nn.ModuleList()
		
		# GINConv requiere una red neuronal (MLP) como parámetro
		for i in range(hidden_layers):
			mlp = nn.Sequential(
				Linear(input_dim+hidden_dim[i], 2 * hidden_dim[i+1]),
				BatchNorm(2 * hidden_dim[i+1]),
				ReLU(),
				Linear(2 * hidden_dim[i+1], hidden_dim[i+1])
			)
			self.convs.append(GINConv(mlp))
		
		self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

	def forward(self, X, edge_index):
		initial_X = X.clone()
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			# Concatenar características iniciales con las actuales
			X_ = torch.cat([initial_X, X], dim=1)
			# GINConv de PyG usa edge_index en lugar de matriz de adyacencia
			X = layer(X_, edge_index)
			hidden_states.append(X)
		
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class TGAE_GIN(nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()

		self.encoder = TGAE_Encoder_GIN(input_dim, hidden_dim, output_dim, num_hidden_layers + 2)

	def forward(self, X, edge_index):
		Z = self.encoder(X, edge_index)
		return Z

# GINEConv
"""
forward(
    x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
    edge_index: Union[Tensor, SparseTensor], 
    edge_attr: Optional[Tensor] = None, 
    size: Optional[Tuple[int, int]] = None
)
"""
class TGAE_Encoder_GINE(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()

		hidden_layers = n_layers - 2
		self.in_proj = nn.Linear(input_dim, hidden_dim[0])
		self.convs = nn.ModuleList()
		
		# GINConv requiere una red neuronal (MLP) como parámetro
		for i in range(hidden_layers):
			mlp = nn.Sequential(
				Linear(input_dim+hidden_dim[i], 2 * hidden_dim[i+1]),
				BatchNorm(2 * hidden_dim[i+1]),
				ReLU(),
				Linear(2 * hidden_dim[i+1], hidden_dim[i+1])
			)
			self.convs.append(GINEConv(mlp, edge_dim=3)) # Change edge_atribute
		
		self.out_proj = nn.Linear(sum(hidden_dim), output_dim)

	def forward(self, X, edge_index, edge_attr):
		# print(edge_attr.shape[1])
		initial_X = X.clone()
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			# Concatenar características iniciales con las actuales
			X_ = torch.cat([initial_X, X], dim=1)
			X = layer(X_, edge_index, edge_attr)
			hidden_states.append(X)
		
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class TGAE_GINE(nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()

		self.encoder = TGAE_Encoder_GINE(input_dim, hidden_dim, output_dim, num_hidden_layers + 2)

	def forward(self, X, edge_index, edge_attr):
		Z = self.encoder(X, edge_index, edge_attr)
		return Z

# Ejemplo de uso:
# Si tienes una matriz de adyacencia A, conviértela a edge_index:
# from torch_geometric.utils import dense_to_sparse
# edge_index, _ = dense_to_sparse(A)
# 
# model = TGAE(num_hidden_layers=2, input_dim=10, hidden_dim=[64, 32], output_dim=16)
# output = model(X, edge_index)


# Implementation using torch-geometric
""" from torch_geometric.nn import GINConv
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential

class TGAE_Encoder(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()
		# print(5)
		hidden_layers = n_layers-2
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim[0])
		self.convs = torch.nn.ModuleList()

		for i in range(hidden_layers):
			# self.convs.append(GINConv(input_dim+hidden_dim[i], hidden_dim[i+1]))
			mlp = Sequential(
				Linear(input_dim+hidden_dim[i], 2 * hidden_dim[i+1]),
				BatchNorm(2 * hidden_dim[i+1]),
				ReLU(),
				Linear(2 * hidden_dim[i+1], hidden_dim[i+1])
			)
			self.convs.append(GINConv(mlp, train_eps=True))
		self.out_proj = torch.nn.Linear(sum(hidden_dim), output_dim)
		# print(6)

	def forward(self, edge_index, X):
		# print(7)
		initial_X = X
		X = self.in_proj(X)
		hidden_states = [X]
		# print(8, edge_index.dtype)
		for layer in self.convs:
			# print(4, type(layer))
			X = layer(
				torch.cat([initial_X, X], dim=1),
				edge_index
			)
			hidden_states.append(X)
		# print(5)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class TGAE(torch.nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()
		# print(1)
		self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers+2)
		# print(2)

	def forward(self, X, adj):
		# print(3)
		Z = self.encoder(adj, X)
		# print(4)
		return Z """


# Original code before edits
""" class GINConv(torch.nn.Module):
	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = torch.nn.Linear(input_dim, output_dim)

	def forward(self, A, X):
		X = self.linear(X + A @ X)
		X = torch.nn.functional.relu(X)
		return X

class TGAE_Encoder(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
		super().__init__()
		hidden_layers = n_layers-2
		self.in_proj = torch.nn.Linear(input_dim, hidden_dim[0])
		self.convs = torch.nn.ModuleList()
		for i in range(hidden_layers):
			self.convs.append(GINConv(input_dim+hidden_dim[i], hidden_dim[i+1]))
		self.out_proj = torch.nn.Linear(sum(hidden_dim), output_dim)

	def forward(self, A, X):
		initial_X = torch.empty_like(X).copy_(X)
		X = self.in_proj(X)
		hidden_states = [X]
		for layer in self.convs:
			X = layer(A, torch.cat([initial_X,X],dim=1))
			hidden_states.append(X)
		X = torch.cat(hidden_states, dim=1)
		X = self.out_proj(X)
		return X

class TGAE(torch.nn.Module):
	def __init__(self, num_hidden_layers, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.encoder = TGAE_Encoder(input_dim, hidden_dim, output_dim, num_hidden_layers+2)

	def forward(self, X, adj):
		Z = self.encoder(adj, X)
		return Z """