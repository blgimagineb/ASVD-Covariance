import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


def calib_fisher_info(model, calib_loader, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_calib_fisher_info.pt"
    if os.path.exists(cache_file) and use_cache:
        all_fisher_info = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info = all_fisher_info[name].to(module.weight.device)
        return
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = 0

    # get fisher info
    for batch in tqdm(calib_loader):
        input_ids = batch["input_ids"][:, :-1].to(model.device)
        labels = batch["input_ids"][:, 1:].to(model.device)
        out = model(input_ids=input_ids, labels=labels)
        out[0].backward()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.fisher_info += module.weight.grad.detach().pow(2).mean(0)
        model.zero_grad()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.fisher_info = module.fisher_info.div(len(calib_loader)).sqrt()

    # remove and save fisher_info
    all_fisher_info = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
            all_fisher_info[name] = module.fisher_info
    torch.save(all_fisher_info, cache_file)

def calculate_covariance_matrix(input_accum):
    inputs = torch.cat(input_accum, dim=0)
    if inputs.dim() == 3:
        batch_size, seq_len, feature_dim = inputs.size()
        inputs = inputs.view(batch_size * seq_len, feature_dim)
    inputs = inputs.to(torch.float32)  # 确保是 torch.float32 类型
    mean = inputs.mean(dim=0, keepdim=True)
    inputs_centered = inputs - mean
    matrix = torch.mm(inputs_centered.t(), inputs_centered) / (inputs_centered.size(0) - 1)
    matrix = torch.where(torch.isnan(matrix), torch.zeros_like(matrix), matrix)
    return matrix

@torch.no_grad()
def calib_input_distribution(model, calib_loader, method, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = (
        f"cache/{model_id.replace('/','_')}_calib_input_distribution_{method}.pt"
    )
    if os.path.exists(cache_file) and use_cache:
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
                module.scaling_diag_matrix = all_scaling_diag_matrix[name].to(torch.float32)  # 确保加载后是 torch.float32 类型
                module.scaling_diag_matrix = module.scaling_diag_matrix.to(module.weight.device)
                print(module.scaling_diag_matrix.dtype)
        return
    model.eval()
    # set hook for every Linear layer

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        elif "covariance" in method:
            if not hasattr(module, 'input_accum'):
                module.input_accum = []
            module.input_accum.append(input[0].detach())

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
            module.scaling_diag_matrix = torch.zeros_like(module.weight, dtype=torch.float32)  # 确保初始化为 torch.float32 类型
            module.register_forward_hook(hook)

    # get activation distribution
    for batch in tqdm(calib_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        model(**batch)

    # remove and save scaling_diag_matrix
    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("k_proj" in name or "v_proj" in name):
            module._forward_hooks.clear()
            if "covariance" in method:
                module.scaling_diag_matrix = calculate_covariance_matrix(module.input_accum).to(torch.float32)  # 确保计算后是 torch.float32 类型
                del module.input_accum
            all_scaling_diag_matrix[name] = module.scaling_diag_matrix
    torch.save(all_scaling_diag_matrix, cache_file)
