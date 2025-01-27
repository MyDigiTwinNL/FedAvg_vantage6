import torch
import argparse
import numpy
from federated_cvdm_training_poc.networks import DeepSurv
from federated_cvdm_training_poc.utils import read_config



def make_predictions(input_data, weight_file_path):
    # Load the saved aggregated weights
    avged_params = torch.load(weight_file_path,  weights_only=True)    
    
    # Create an instance of the model
    dl_config = read_config("lifelines_ci.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSurv(dl_config['network']).to(device)
    
    #print("Model architecture:", model.model)


    # Load the saved weights into the model
    model.load_state_dict(avged_params)
    
    # Set the model to evaluation mode
        
    # Move the model and input data to the appropriate device (CPU/GPU)
    
    model.to(device)
    input_data = input_data.to(device)
    
    #model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(input_data)
    
    # Return the predictions
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained model.")
    parser.add_argument("--weight_file", type=str, required=True, help="Path to the file containing the model weights")
    args = parser.parse_args()

    # Example usage

    #input_data = torch.randn(2, 16).float()

    input_data = torch.tensor([[0. ,0. ,        0. ,        0.,         0.90157047, 0.14333206, 0.9051146 , 0.95057034, 0.17572524 ,1.   ,      0.79366138 ,0.486 , 0. ,    0.49 ,      0.675,      0.49986313],
                               [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                               ])

    #input_data = torch.randn(16,32)  # Replace with your actual input data
    #random_input = torch.randn(1, 16, requires_grad=True)
    #print(input_data.shape)
    predictions = make_predictions(input_data.double(), args.weight_file)
    print(predictions)
    #print(len(predictions))
