# import os
# import csv
# import random
# import pathlib
# import argparse
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.amp import GradScaler, autocast
# from tqdm import tqdm
# import torchvision
# from torchvision import transforms
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Physical constants (aligned with Lin et al., 2018)
# WAVELENGTH = 750e-6    # 0.4 THz (750 μm)
# PIXEL_SIZE = 400e-6    # 400 μm pixel pitch
# LAYER_DISTANCE = 10e-3 # Reduced to 10 mm
# LAYER_SIZE = 200       # 200x200 pixels
# NUM_LAYERS = 5         # Reduced to 2 layers

# def detector_region(x):
#     # Circular detector layout (60x60 pixels, 10 regions)
#     batch_size = x.shape[0]
#     outputs = torch.zeros(batch_size, 10, device=device)
#     center = LAYER_SIZE // 2
#     radius = LAYER_SIZE // 4
#     detector_size = 60
#     for i in range(10):
#         angle = 2 * np.pi * i / 10
#         y_center = int(center + radius * np.cos(angle))
#         x_center = int(center + radius * np.sin(angle))
#         y_start = max(0, y_center - detector_size // 2)
#         y_end = min(LAYER_SIZE, y_center + detector_size // 2)
#         x_start = max(0, x_center - detector_size // 2)
#         x_end = min(LAYER_SIZE, x_center + detector_size // 2)
#         outputs[:, i] = x[:, y_start:y_end, x_start:x_end].mean(dim=(1, 2))
#     return outputs

# class DiffractiveLayer(nn.Module):
#     def __init__(self, layer_size=200, wavelength=750e-6, pixel_size=400e-6, distance=10e-3):
#         super(DiffractiveLayer, self).__init__()
#         self.layer_size = layer_size
#         self.wavelength = wavelength
#         self.pixel_size = pixel_size
#         self.distance = distance
        
#         # Precompute transfer function
#         k = 2 * np.pi / wavelength
#         fx = torch.fft.fftfreq(layer_size, d=pixel_size).to(device)
#         fy = torch.fft.fftfreq(layer_size, d=pixel_size).to(device)
#         FX, FY = torch.meshgrid(fx, fy, indexing='ij')
#         kx = 2 * np.pi * FX
#         ky = 2 * np.pi * FY
#         kz_squared = k**2 - kx**2 - ky**2
#         kz_squared = torch.clamp(kz_squared, min=0)
#         kz = torch.sqrt(kz_squared + 0j)
#         kz = torch.where(kz.imag == 0, kz, 0j)
#         self.H = torch.exp(1j * kz * distance).to(device)
#         print(f"DiffractiveLayer: H shape={self.H.shape}, evanescent waves={(kz.imag != 0).sum().item()}")

#     def forward(self, waves):
#         # waves: (batch, 2, 200, 200) [real, imag]
#         waves_complex = waves[:, 0] + 1j * waves[:, 1]
#         fft = torch.fft.fft2(waves_complex)
#         fft = fft * self.H
#         output = torch.fft.ifft2(fft)
#         if torch.isnan(output).any():
#             raise ValueError("NaN in propagation")
#         return torch.stack((output.real, output.imag), dim=1)

# class Net(nn.Module):
#     def __init__(self, num_layers=2):
#         super(Net, self).__init__()
#         self.phase = nn.ParameterList([
#             nn.Parameter(torch.rand(200, 200) * 2 * np.pi)
#             for _ in range(num_layers)
#         ])
#         self.diffractive_layers = nn.ModuleList([
#             DiffractiveLayer() for _ in range(num_layers)
#         ])
#         self.last_diffractive_layer = DiffractiveLayer()
    
#     def forward(self, x):
#         # x: (batch, 2, 200, 200) [real, imag]
#         for index, layer in enumerate(self.diffractive_layers):
#             x = layer(x)
#             exp_j_phase = torch.stack((torch.cos(self.phase[index]), 
#                                      torch.sin(self.phase[index])), dim=-1)
#             x_real = x[:, 0] * exp_j_phase[..., 0] - x[:, 1] * exp_j_phase[..., 1]
#             x_imag = x[:, 0] * exp_j_phase[..., 1] + x[:, 1] * exp_j_phase[..., 0]
#             x = torch.stack((x_real, x_imag), dim=1)
#         x = self.last_diffractive_layer(x)
#         x_abs = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
#         output = detector_region(x_abs)
#         return output

# class EnergyMaximizationLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(EnergyMaximizationLoss, self).__init__()
#         self.margin = margin
    
#     def forward(self, outputs, targets):
#         batch_size = outputs.shape[0]
#         outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
#         target_energy = outputs[torch.arange(batch_size), targets]
#         mask = torch.ones_like(outputs).scatter_(1, targets.unsqueeze(1), 0)
#         non_target_energy = (outputs * mask).max(dim=1)[0]
#         loss = torch.mean(torch.relu(self.margin - target_energy + non_target_energy))
#         return loss

# def visualize_fields(model, images, labels, class_names, epoch):
#     model.eval()
#     with torch.no_grad():
#         outputs = model(images.to(device))
#         x = model.last_diffractive_layer(images.to(device))
#         x_abs = torch.sqrt(x[:, 0]**2 + x[:, 1]**2)
    
#     num_images = min(len(images), 5)
#     fig, axes = plt.subplots(num_images, 2, figsize=(8, num_images * 3))
#     for i in range(num_images):
#         axes[i, 0].imshow(images[i, 0].cpu(), cmap='gray')
#         axes[i, 0].set_title(f'Input: {class_names[labels[i]]}')
#         axes[i, 0].axis('off')
#         ax = axes[i, 1]
#         ax.imshow(x_abs[i].cpu(), cmap='inferno')
#         center = LAYER_SIZE // 2
#         radius = LAYER_SIZE // 4
#         detector_size = 60
#         for j in range(10):
#             angle = 2 * np.pi * j / 10
#             y_center = int(center + radius * np.cos(angle))
#             x_center = int(center + radius * np.sin(angle))
#             rect = plt.Rectangle((x_center - detector_size // 2, y_center - detector_size // 2),
#                                  detector_size, detector_size, edgecolor='blue', facecolor='none')
#             ax.add_patch(rect)
#         ax.set_title(f'Output Intensity')
#         ax.axis('off')
    
#     normalized_outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
#     print(f"Epoch {epoch} - Sample outputs (normalized):\n{normalized_outputs.cpu().numpy()[:5]}")
#     print(f"Epoch {epoch} - Predicted classes: {torch.argmax(outputs, dim=1)[:5].cpu().numpy()}")
#     print(f"Epoch {epoch} - True labels: {labels[:5].numpy()}")
    
#     # Plot detector energies
#     fig2, ax2 = plt.subplots(figsize=(8, 4))
#     for i in range(min(5, len(outputs))):
#         ax2.plot(normalized_outputs[i].cpu().numpy(), label=f'Sample {i} (Label {labels[i]})')
#     ax2.set_xlabel('Detector')
#     ax2.set_ylabel('Normalized Energy')
#     ax2.legend()
#     ax2.set_title(f'Epoch {epoch} - Detector Energies')
    
#     plt.tight_layout()
#     return fig, fig2

# def plot_confusion_matrix(labels, preds, epoch):
#     cm = confusion_matrix(labels, preds, labels=range(10))
#     fig, ax = plt.subplots(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
#     ax.set_xlabel('Predicted')
#     ax.set_ylabel('True')
#     ax.set_title(f'Epoch {epoch} - Confusion Matrix')
#     return fig

# def main(args):
#     if not os.path.exists(args.model_save_path):
#         os.makedirs(args.model_save_path)
#     os.makedirs('results', exist_ok=True)

#     transform = transforms.Compose([
#         transforms.Pad(padding=(86, 86, 86, 86)),  # Pad 28x28 to 200x200
#         transforms.ToTensor(),  # No normalization
#     ])
#     train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
#     val_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform, download=True)
#     train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
#                                   num_workers=8, shuffle=True, pin_memory=True)
#     val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, 
#                                 num_workers=8, shuffle=False, pin_memory=True)

#     model = Net(num_layers=NUM_LAYERS).to(device)
#     if args.whether_load_model:
#         model.load_state_dict(torch.load(args.model_save_path + str(args.start_epoch) + args.model_name))
#         print(f'Model: "{args.model_save_path + str(args.start_epoch) + args.model_name}" loaded.')
#     else:
#         if os.path.exists(args.result_record_path):
#             os.remove(args.result_record_path)
#         with open(args.result_record_path, 'w') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'LR'])

#     criterion = EnergyMaximizationLoss().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
#     warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5*len(train_dataloader))
#     scaler = GradScaler('cuda')

#     best_val_acc = 0.0
#     patience = 10
#     epochs_no_improve = 0

#     for epoch in range(args.start_epoch + 1, args.start_epoch + 1 + args.num_epochs):
#         log = [epoch]

#         model.train()
#         train_len = 0.0
#         train_running_counter = 0.0
#         train_running_loss = 0.0

#         tk0 = tqdm(train_dataloader, ncols=100, total=len(train_dataloader))
#         for train_iter, train_data_batch in enumerate(tk0):
#             train_images = train_data_batch[0].to(device)  # (batch, 1, 200, 200)
#             train_labels = train_data_batch[1].to(device)  # (batch,)
#             train_images = torch.cat((train_images, torch.zeros_like(train_images)), dim=1)  # (batch, 2, 200, 200)

#             optimizer.zero_grad()
#             with autocast('cuda'):
#                 train_outputs = model(train_images)
#                 train_loss = criterion(train_outputs, train_labels)
            
#             scaler.scale(train_loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
#             scaler.step(optimizer)
#             scaler.update()

#             train_len += len(train_labels)
#             train_running_loss += train_loss.item() * len(train_labels)
#             train_counter = torch.eq(torch.argmax(train_outputs, dim=1), train_labels).float().sum()

#             train_running_counter += train_counter
#             train_loss_avg = train_running_loss / train_len
#             train_accuracy = train_running_counter / train_len

#             tk0.set_description_str(f'Epoch {epoch}/{args.start_epoch + args.num_epochs}')
#             tk0.set_postfix({'Train_Loss': f'{train_loss_avg:.5f}', 'Train_Acc': f'{train_accuracy:.5f}'})

#             # Log detector outputs
#             normalized_outputs = train_outputs / (torch.sum(train_outputs, dim=1, keepdim=True) + 1e-6)
#             print(f"Iter {train_iter} - Mean target energy: {normalized_outputs[torch.arange(len(train_labels)), train_labels].mean().item():.4f}")

#         log.append(train_loss_avg)
#         log.append(train_accuracy.item())

#         model.eval()
#         val_len = 0.0
#         val_running_counter = 0.0
#         val_running_loss = 0.0
#         all_preds = []
#         all_labels = []

#         tk1 = tqdm(val_dataloader, ncols=100, total=len(val_dataloader))
#         with torch.no_grad():
#             for val_iter, val_data_batch in enumerate(tk1):
#                 val_images = val_data_batch[0].to(device)
#                 val_labels = val_data_batch[1].to(device)
#                 val_images = torch.cat((val_images, torch.zeros_like(val_images)), dim=1)

#                 val_outputs = model(val_images)
#                 val_loss = criterion(val_outputs, val_labels)
#                 val_counter = torch.eq(torch.argmax(val_outputs, dim=1), val_labels).float().sum()

#                 val_len += len(val_labels)
#                 val_running_loss += val_loss.item() * len(val_labels)
#                 val_running_counter += val_counter

#                 all_preds.append(torch.argmax(val_outputs, dim=1).cpu().numpy())
#                 all_labels.append(val_labels.cpu().numpy())

#                 val_loss_avg = val_running_loss / val_len
#                 val_accuracy = val_running_counter / val_len

#                 tk1.set_description_str(f'Epoch {epoch}/{args.start_epoch + args.num_epochs}')
#                 tk1.set_postfix({'Val_Loss': f'{val_loss_avg:.5f}', 'Val_Acc': f'{val_accuracy:.5f}'})

#         log.append(val_loss_avg)
#         log.append(val_accuracy.item())
#         log.append(optimizer.param_groups[0]['lr'])

#         torch.save(model.state_dict(), args.model_save_path + f'epoch_{epoch}' + args.model_name)
#         print(f'Model: "{args.model_save_path + f"epoch_{epoch}" + args.model_name}" saved.')

#         if val_accuracy > best_val_acc:
#             best_val_acc = val_accuracy
#             torch.save(model.state_dict(), args.model_save_path + 'best' + args.model_name)
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print(f"Early stopping at epoch {epoch}")
#                 break

#         with open(args.result_record_path, 'a', newline='') as csvfile:
#             writer = csv.writer(csvfile)
#             writer.writerow(log)

#         if epoch <= 5:
#             warmup_scheduler.step()
#         else:
#             scheduler.step()

#         # Visualize fields and confusion matrix every 5 epochs
#         if epoch % 5 == 0:
#             sample_loader = DataLoader(
#                 torch.utils.data.Subset(val_dataset, range(10)),
#                 batch_size=10, shuffle=False
#             )
#             sample_images, sample_labels = next(iter(sample_loader))
#             class_names = [str(i) for i in range(10)]
#             fig, fig2 = visualize_fields(model, sample_images, sample_labels, class_names, epoch)
#             fig.savefig(f'results/fields_epoch_{epoch}.png')
#             fig2.savefig(f'results/energies_epoch_{epoch}.png')
#             plt.close(fig)
#             plt.close(fig2)

#             # Confusion matrix
#             all_preds = np.concatenate(all_preds)
#             all_labels = np.concatenate(all_labels)
#             cm_fig = plot_confusion_matrix(all_labels, all_preds, epoch)
#             cm_fig.savefig(f'results/cm_epoch_{epoch}.png')
#             plt.close(cm_fig)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--batch-size', type=int, default=128)
#     parser.add_argument('--num-epochs', type=int, default=50)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--lr', type=float, default=2e-3)  # Increased LR
#     parser.add_argument('--whether-load-model', type=bool, default=False)
#     parser.add_argument('--start-epoch', type=int, default=0)
#     parser.add_argument('--model-name', type=str, default='_model.pth')
#     parser.add_argument('--model-save-path', type=str, default='./saved_model/')
#     parser.add_argument('--result-record-path', type=pathlib.Path, default='./result.csv')

#     torch.backends.cudnn.benchmark = True
#     args = parser.parse_args()
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     main(args)