import torch
import transformers
import torchvision
import torchaudio


print("cuda available: ", torch.cuda.is_available())
print("cuda enabled: ", torch.backends.cudnn.enabled)
print("torch version: ", torch.__version__)
print("torch vision version: ", torchvision.__version__)
print("torch audio version: ", torchaudio.__version__)
print("cuda version: ", torch.version.cuda)
print("device count: ", torch.cuda.device_count()) 
print("cuda device: ", torch.cuda.current_device())
print("cuda device name: ", torch.cuda.get_device_name(0))
print("cudnn version: ", torch.backends.cudnn.version())
print("transformers version: ", transformers.__version__)