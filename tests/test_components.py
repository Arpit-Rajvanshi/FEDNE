import fedne
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

from fedne.models.encoder import Encoder
from fedne.models.surrogate import SurrogateRepulsion, generate_grid_query_points, compute_repulsion_targets
from fedne.graph.knn import construct_knn_graph
from fedne.losses.attraction import AttractionLoss
from fedne.losses.repulsion import RepulsionLoss
from fedne.server.server import Server

class TestFEDNEComponents(unittest.TestCase):
    
    def test_encoder_shape(self) -> None:
        """
        Verify that the Encoder maps input of shape (N, D) to output shape (N, 2).
        """
        N = 10
        input_dim = 784
        x = torch.randn(N, input_dim)
        encoder = Encoder(input_dim=input_dim, embedding_dim=2)
        z = encoder(x)
        self.assertEqual(z.shape, (N, 2))
        
    def test_knn_properties(self) -> None:
        """
        Verify kNN returns correct neighbors, has no self-edges, and has correct edge count.
        """
        # Create a simple dataset where distances are obvious
        X = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0]
        ])
        # k = 2
        edges = construct_knn_graph(X, k=2)
        
        # Total edges should be N * k = 4 * 2 = 8
        self.assertEqual(len(edges), 8)
        
        # Verify no self edges
        for src, dst in edges:
            self.assertNotEqual(src, dst)
            
        # Point 0 is at (0,0). Neighbors should be 1 (at (1,0)) and 2 (at (0,1)).
        p0_neighbors = edges[edges[:, 0] == 0][:, 1]
        self.assertIn(1, p0_neighbors)
        self.assertIn(2, p0_neighbors)
        self.assertNotIn(3, p0_neighbors)
        
    def test_attraction_loss(self) -> None:
        """
        Verify attraction loss decreases the distance between two nearby points.
        """
        z_i = nn.Parameter(torch.tensor([[0.1, 0.1]]))
        z_j = nn.Parameter(torch.tensor([[0.2, 0.1]]))
        
        loss_fn = AttractionLoss()
        optimizer = optim.SGD([z_i, z_j], lr=0.1)
        
        # Distance before optimization
        dist_before = torch.sum((z_i - z_j) ** 2).item()
        
        # Run optimization step
        optimizer.zero_grad()
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        
        # Distance after optimization
        dist_after = torch.sum((z_i - z_j) ** 2).item()
        
        # Distance should decrease
        self.assertLess(dist_after, dist_before)
        
    def test_repulsion_loss(self) -> None:
        """
        Verify repulsion loss increases the distance between two points.
        """
        z_i = nn.Parameter(torch.tensor([[2.0, 2.0]]))
        z_neg = nn.Parameter(torch.tensor([[[2.1, 2.0]]]))  # Shape [B, b, D] = [1, 1, 2]
        
        loss_fn = RepulsionLoss()
        optimizer = optim.SGD([z_i, z_neg], lr=0.1)
        
        # Distance before optimization
        dist_before = torch.sum((z_i.unsqueeze(1) - z_neg) ** 2).item()
        
        # Run optimization step
        optimizer.zero_grad()
        loss = loss_fn(z_i, z_neg)
        loss.backward()
        optimizer.step()
        
        # Distance after optimization
        dist_after = torch.sum((z_i.unsqueeze(1) - z_neg) ** 2).item()
        
        # Distance should increase
        self.assertGreater(dist_after, dist_before)
        
    def test_surrogate_training(self) -> None:
        """
        Generate synthetic targets and verify surrogate MSE decreases during training.
        """
        surrogate = SurrogateRepulsion(hidden_dim=16)
        optimizer = optim.Adam(surrogate.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Synthetic data: 100 query points and 100 targets
        q_points = torch.randn(100, 2)
        targets = torch.rand(100, 1) * 10.0
        
        # Initial MSE
        surrogate.eval()
        with torch.no_grad():
            initial_mse = criterion(surrogate(q_points), targets).item()
            
        # Train for 20 steps
        surrogate.train()
        for _ in range(20):
            optimizer.zero_grad()
            loss = criterion(surrogate(q_points), targets)
            loss.backward()
            optimizer.step()
            
        # Final MSE
        surrogate.eval()
        with torch.no_grad():
            final_mse = criterion(surrogate(q_points), targets).item()
            
        self.assertLess(final_mse, initial_mse)
        
    def test_fedavg_aggregation(self) -> None:
        """
        Verify weighted average math for FedAvg in Server.
        """
        config = {
            "model": {
                "embedding_dim": 2,
                "hidden_dims": [16]
            },
            "surrogate": {
                "hidden_dim": 16
            }
        }
        
        # Server with 2 clients of sizes 10 and 30
        server = Server(config=config, input_dim=8, client_sizes=[10, 30], device=torch.device("cpu"))
        
        # Dummy client mock
        class ClientMock:
            def __init__(self, client_id, weight_val) -> None:
                self.client_id = client_id
                self.encoder = Encoder(input_dim=8, embedding_dim=2, hidden_dims=[16])
                # Set all weights/biases to weight_val
                for param in self.encoder.parameters():
                    nn.init.constant_(param, weight_val)
            def set_encoder(self, enc):
                pass
                
        # Client 0 weights set to 1.0, Client 1 weights set to 3.0
        client0 = ClientMock(0, 1.0)
        client1 = ClientMock(1, 3.0)
        
        # Aggregate
        server.aggregate_encoder([client0, client1])
        
        # Expected value: (10 * 1.0 + 30 * 3.0) / 40 = 2.5
        global_params = list(server.global_encoder.parameters())
        for param in global_params:
            self.assertTrue(torch.allclose(param, torch.tensor(2.5)))
            
    def test_surrogate_broadcast(self) -> None:
        """
        Verify server stores every surrogate separately, no averaging occurs,
        and each client receives all surrogates.
        """
        config = {
            "model": {
                "embedding_dim": 2,
                "hidden_dims": [16]
            },
            "surrogate": {
                "hidden_dim": 16
            }
        }
        
        # Server with 2 clients
        server = Server(config=config, input_dim=8, client_sizes=[10, 20], device=torch.device("cpu"))
        
        class ClientMock:
            def __init__(self, client_id, val) -> None:
                self.client_id = client_id
                self.surrogate = SurrogateRepulsion(hidden_dim=16)
                for param in self.surrogate.parameters():
                    nn.init.constant_(param, val)
                self.other_surrogates = []
            def set_other_surrogates(self, other_surrs):
                self.other_surrogates = other_surrs
                
        client0 = ClientMock(0, 5.0)
        client1 = ClientMock(1, 9.0)
        
        # Collect surrogates at server
        server.collect_surrogates([client0, client1])
        
        # Verify server stored them separately without averaging
        s0_params = list(server.surrogates[0].parameters())
        s1_params = list(server.surrogates[1].parameters())
        self.assertTrue(torch.allclose(s0_params[0], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(s1_params[0], torch.tensor(9.0)))
        
        # Broadcast surrogates
        server.broadcast_surrogates([client0, client1])
        
        # Verify client0 received reference to both surrogates
        self.assertEqual(len(client0.other_surrogates), 2)
        self.assertTrue(torch.allclose(list(client0.other_surrogates[0].parameters())[0], torch.tensor(5.0)))
        self.assertTrue(torch.allclose(list(client0.other_surrogates[1].parameters())[0], torch.tensor(9.0)))

if __name__ == "__main__":
    unittest.main()
