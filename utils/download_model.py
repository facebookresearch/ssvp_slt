import os
import wget

def get_model_path():
    model_path = 'signhiera_mock.pth'
    url = 'https://dl.fbaipublicfiles.com/SONAR/asl/signhiera_mock.pth'

    # Check if the model file exists
    if os.path.exists(model_path):
        print(f"Model already exists at: {model_path}")
    else:
        print("Model not found, downloading...")
        filename = wget.download(url, model_path)
        print(f"Downloaded model to: {filename}")
    
    return model_path