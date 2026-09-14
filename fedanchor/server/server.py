import torch
import copy
from typing import Dict, List, Tuple, Any
from fedanchor.models.encoder import Encoder

class AnchorServer:
    """
    Federated Anchor Server.
    Stores global encoder weights and maintains separate client anchor sets.
    Performs weighted FedAvg on encoder weights.
    DOES NOT average anchors across clients.
    """
    def __init__(
        self,
        input_dim: int = 784,
        embedding_dim: int = 2,
        device: str = "cpu"
    ) -> None:
        self.device = torch.device(device)
        self.global_encoder = Encoder(input_dim=input_dim, embedding_dim=embedding_dim).to(self.device)
        self.client_anchors: Dict[int, torch.Tensor] = {}

    def get_global_encoder_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.cpu().clone() for k, v in self.global_encoder.state_dict().items()}

    def get_other_anchors(self, target_client_id: int) -> torch.Tensor:
        other_list = []
        for cid, anc in self.client_anchors.items():
            if cid != target_client_id:
                other_list.append(anc.to(self.device))
                
        if len(other_list) == 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=self.device)
            
        return torch.cat(other_list, dim=0)

    def aggregate_encoders(
        self,
        client_encoder_states: List[Dict[str, torch.Tensor]],
        client_sample_counts: List[int]
    ) -> None:
        total_samples = sum(client_sample_counts)
        if total_samples == 0:
            return
            
        avg_state = copy.deepcopy(client_encoder_states[0])
        for key in avg_state.keys():
            avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
            
        for state, n_m in zip(client_encoder_states, client_sample_counts):
            weight = float(n_m) / float(total_samples)
            for key in avg_state.keys():
                avg_state[key] += state[key].to(torch.float32) * weight
                
        self.global_encoder.load_state_dict(avg_state)

    def update_client_anchors(self, client_id: int, anchors_tensor: torch.Tensor) -> None:
        self.client_anchors[client_id] = anchors_tensor.detach().cpu().clone()

    def compute_communication_bytes(
        self,
        num_clients: int,
        num_anchors: int,
        precision_bytes: int = 4
    ) -> Dict[str, Any]:
        """
        Compute communication volume per round with explicit separation of Encoder vs Anchor transfer bytes.
        """
        encoder_params = sum(p.numel() for p in self.global_encoder.parameters())
        encoder_bytes_per_client = encoder_params * precision_bytes
        
        anchor_scalars_per_client = num_anchors * 2
        anchor_bytes_per_client = anchor_scalars_per_client * precision_bytes
        
        encoder_upload_total = num_clients * encoder_bytes_per_client
        anchor_upload_total = num_clients * anchor_bytes_per_client
        
        encoder_download_total = num_clients * encoder_bytes_per_client
        anchor_download_total = num_clients * (num_clients - 1) * anchor_bytes_per_client
        
        total_upload = encoder_upload_total + anchor_upload_total
        total_download = encoder_download_total + anchor_download_total
        total_round = total_upload + total_download
        
        return {
            "encoder_parameter_count": encoder_params,
            "encoder_bytes_per_client": encoder_bytes_per_client,
            "anchor_scalar_count_per_client": anchor_scalars_per_client,
            "anchor_bytes_per_client": anchor_bytes_per_client,
            "encoder_upload_bytes_total": encoder_upload_total,
            "anchor_upload_bytes_total": anchor_upload_total,
            "encoder_download_bytes_total": encoder_download_total,
            "anchor_download_bytes_total": anchor_download_total,
            "total_upload_bytes": total_upload,
            "total_download_bytes": total_download,
            "total_round_bytes": total_round,
            "anchor_to_encoder_scalar_ratio": float(anchor_scalars_per_client) / float(encoder_params)
        }
