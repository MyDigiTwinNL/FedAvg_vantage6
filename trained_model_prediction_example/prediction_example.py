import torch
import argparse

def make_predictions(input_data, weight_file_path, model_class):
    # Load the saved aggregated weights
    avged_params = torch.load(weight_file_path)
    
    # Create an instance of the model
    model = model_class()
    
    # Load the saved weights into the model
    model.load_state_dict(avged_params)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the model and input data to the appropriate device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    
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
    #input_data = torch.randn(1, 3, 224, 224)  # Replace with your actual input data
    #predictions = make_predictions(input_data, args.weight_file, YourModelClass)
    #print(predictions)
