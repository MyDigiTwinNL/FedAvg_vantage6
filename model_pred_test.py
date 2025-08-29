import torch
import argparse
import numpy
from federated_cvdm_training_poc.networks import DeepSurv
from federated_cvdm_training_poc.utils import read_config
import onnxruntime as ort
import numpy as np
import torch

def onnx_model_prediction(input_data, onnx_file_path: str):
    """
    Evaluate an ONNX model on the given input data.

    Args:
        input_data (np.ndarray or torch.Tensor): Input data of shape [batch_size, n_features].
        onnx_file_path (str): Path to the ONNX model file.

    Returns:
        np.ndarray: Model predictions.
    """

    # Convert torch.Tensor to numpy array if needed
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.detach().cpu().numpy()

    # Create an ONNX Runtime session
    ort_session = ort.InferenceSession(onnx_file_path)

    # Name of the input in the ONNX model (must match the one used in export)
    input_name = ort_session.get_inputs()[0].name

    # Run inference
    outputs = ort_session.run(None, {input_name: input_data})

    # Get the first (and only) output
    return outputs[0]


def make_predictions(input_data, weight_file_path):
    # Load the saved aggregated weights
    avged_params = torch.load(weight_file_path,  weights_only=True)    
    
    # Create an instance of the model
    dl_config = read_config("test/lifelines_ci.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSurv(dl_config['network']).to(device)
    
    #print("Model architecture:", model.model)


    # Load the saved weights into the model
    model.load_state_dict(avged_params)
    
    # Set the model to evaluation mode
        
    # Move the model and input data to the appropriate device (CPU/GPU)
    
    model.to(device)
    input_data = input_data.to(device)
    
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Return the predictions
    return predictions







if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    #parser.add_argument("--weight_file", type=str, required=True, help="Path to the file containing the model weights")
    #args = parser.parse_args()

    # Example usage

    #input_data = torch.randn(2, 16).float()

    input_data = torch.tensor([[0. ,0. ,        0. ,        0.,         0.90157047, 0.14333206, 0.9051146 , 0.95057034, 0.17572524 ,1.   ,      0.79366138 ,0.486 , 0. ,    0.49 ,      0.675,      0.49986313],
                               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                               ])
    
    input_data_2 = input_data.detach().cpu().numpy().astype(np.float64)

    #input_data = torch.randn(16,32)  # Replace with your actual input data
    #random_input = torch.randn(1, 16, requires_grad=True)
    #print(input_data.shape)
    
    pred_torch = make_predictions(input_data.double(), weight_file_path="/tmp/aggregated_weights.pth")

    print(pred_torch)

    pred_onnx = onnx_model_prediction(onnx_file_path="/tmp/model_wg.onnx",input_data=input_data_2)

    print(pred_onnx)

    #print(len(predictions))
