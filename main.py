import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

with open('wizard_of_oz.txt', 'r', encoding='utf-8') as f:
    text = f.read()
