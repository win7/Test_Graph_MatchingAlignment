📌 Nuevo model.py (torch_geometric compatible)

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool

# Encoder usando GINConv de torch_geometric
class TGAE_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Proyección de entrada
        self.in_proj = nn.Linear(input_dim, hidden_dims[0])

        # Lista de capas GINConv
        self.convs = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Construye el MLP que usa GINConv
            mlp = nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i]),
                nn.ReLU(),
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )
            self.convs.append(GINConv(nn=mlp))

        # Proyección final
        self.out_proj = nn.Linear(sum(hidden_dims), output_dim)

    def forward(self, x, edge_index, batch=None):
        # x: [num_nodes, input_dim]
        # edge_index: [2, num_edges]
        x0 = x.clone()

        # Primera proyección lineal
        x = self.in_proj(x)
        states = [x]

        # Aplicar cada capa GINConv
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            states.append(x)

        # Concatenar todos los estados, luego proyectar
        h = torch.cat(states, dim=1)
        out = self.out_proj(h)

        # Si batch está definido, se puede hacer pooling de grafo:
        if batch is not None:
            out = global_mean_pool(out, batch)

        return out

# Modelo completo TGAE
class TGAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.encoder = TGAE_Encoder(input_dim, hidden_dims, output_dim)

    def forward(self, data):
        # data: objeto torch_geometric.data.Data con:
        # data.x: características de nodos
        # data.edge_index: conexiones
        # data.batch (opcional) para mini-batching
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        z = self.encoder(x, edge_index, batch)
        return z




=================

3️⃣ Adaptación CORRECTA manteniendo los 3 bucles

La adaptación correcta debe asumir:

train_loader[dataset] = [
    Data(...),   # sample 0
    Data(...),   # sample 1
    ...
]


Y entonces el tercer bucle debe existir.

4️⃣ fit_TGAE_subgraph CORREGIDA (3 bucles, PyG, sin DataLoader)

👇 Esta versión respeta exactamente la estructura original

def fit_TGAE_subgraph(
    data,
    no_samples,
    GAE,
    epoch,
    train_loader,
    train_features,  # mantenido por compatibilidad
    device,
    lr,
    test_pairs
):
    optimizer = Adam(GAE.parameters(), lr=lr, weight_decay=5e-4)

    best_hitAtOne = best_hitAtFive = best_hitAtTen = best_hitAtFifty = 0

    for step in range(epoch):                              # (1) epoch
        GAE.train()
        loss = 0.0

        for dataset in train_loader.keys():                # (2) dataset
            for i in range(len(train_loader[dataset])):    # (3) samples

                pyg_data = train_loader[dataset][i].to(device)

                optimizer.zero_grad()

                z = GAE(pyg_data)

                pos_edge_index = pyg_data.edge_index
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=pyg_data.num_nodes,
                    num_neg_samples=pos_edge_index.size(1)
                ).to(device)

                pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
                neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

                pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15)
                neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15)

                batch_loss = pos_loss.mean() + neg_loss.mean()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()

        loss /= no_samples


👉 Aquí no se pierde absolutamente nada de la lógica original
👉 Solo se reemplaza cómo se calcula la pérdida

5️⃣ Evaluación: también respeta la lógica original

La evaluación NO depende del tercer bucle, igual que en tu código:

keys = list(train_loader.keys())
G1 = train_loader[keys[0]][0]
G2 = train_loader[keys[1]][0]


✔ Se mantiene intacta
✔ Se compara solo el grafo base


============
tau = 0.2  # prueba 0.1–0.5
pos_logits = F.cosine_similarity(z[src], z[dst]) / tau
neg_logits = F.cosine_similarity(z[neg_src], z[neg_dst]) / tau


=============

def fit_TGAE_subgraph(
    data,
    no_samples,
    GAE,
    epoch,
    train_loader,      # dict[str, Data]
    train_features,    # dict[str, Tensor]
    device,
    lr,
    test_pairs,
    patience=10,
    min_delta=1e-4,
):
    optimizer = torch.optim.Adam(GAE.parameters(), lr=lr, weight_decay=5e-4)

    best_hitAtOne = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    loop = tqdm(range(1, epoch + 1))

    for step in loop:
        loop.set_description(f"Epoch {step}")

        GAE.train()
        total_loss = 0.0

        # =====================
        # TRAIN
        # =====================
        for dataset, data_obj in train_loader.items():
            x = data_obj.x.to(device)
            edge_index = data_obj.edge_index.to(device)

            z = GAE(x, edge_index)

            # -------- Decoder (cosine + temperature)
            z = F.normalize(z, dim=1)
            src, dst = edge_index
            pos_logits = F.cosine_similarity(z[src], z[dst]) / 0.2

            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=z.size(0),
                num_neg_samples=edge_index.size(1) // 2,
                method="sparse",
            )
            neg_src, neg_dst = neg_edge_index
            neg_logits = F.cosine_similarity(z[neg_src], z[neg_dst]) / 0.2

            loss_pos = F.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits)
            )
            loss_neg = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )

            total_loss += loss_pos + loss_neg

        optimizer.zero_grad()
        total_loss = total_loss / no_samples
        total_loss.backward()
        optimizer.step()

        # =====================
        # EVALUATION
        # =====================
        GAE.eval()
        with torch.no_grad():
            keys = list(train_loader.keys())
            d1, d2 = keys[0], keys[1]

            z1 = GAE(
                train_loader[d1].x.to(device),
                train_loader[d1].edge_index.to(device),
            )
            z2 = GAE(
                train_loader[d2].x.to(device),
                train_loader[d2].edge_index.to(device),
            )

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            D = torch.cdist(z1, z2)

            if data == "ACM_DBLP":
                test_idx = test_pairs[:, 0]
                labels = test_pairs[:, 1]
            else:
                test_idx = test_pairs[0]
                labels = test_pairs[1]

            hitAtOne = 0
            for i in range(len(test_idx)):
                pred = torch.argmin(D[test_idx[i]]).item()
                if pred == labels[i]:
                    hitAtOne += 1

            cur_hitAtOne = hitAtOne / len(test_idx)

        # =====================
        # EARLY STOPPING
        # =====================
        if cur_hitAtOne > best_hitAtOne + min_delta:
            best_hitAtOne = cur_hitAtOne
            best_epoch = step
            best_state = {k: v.cpu() for k, v in GAE.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        loop.set_postfix(
            loss=total_loss.item(),
            hit1=cur_hitAtOne,
            patience=f"{patience_counter}/{patience}",
        )

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {step}")
            break

    # =====================
    # RESTORE BEST MODEL
    # =====================
    if best_state is not None:
        GAE.load_state_dict(best_state)

    print("\nBest Epoch:", best_epoch)
    print("Best Hit@1:", best_hitAtOne)
