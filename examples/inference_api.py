"""
Example: Use the SH-GNN inference API.
"""
import requests
import json

url = "http://localhost:8000/predict"

# Create dummy data (replace with your own)
sample = {
    "nodes": {
        "x": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "edge_index": [[0, 1], [1, 0]],
        "edge_attr": [[0.5, 1.2, 2.3], [0.5, 1.2, 2.3]]
    },
    "return_alms": False
}

response = requests.post(url, json=sample)
print(response.json())
