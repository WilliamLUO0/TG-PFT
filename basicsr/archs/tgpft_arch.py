"""An official PyTorch implementation of "TG-PFT: A Topography-Guided Progressive Focused Transformer for Flood Map Downscaling".

arXiv: https://arxiv.org/abs/XXXX.XXXXX

Built upon PFT: https://github.com/LabShuHangGU/PFT-SR
"""

import math
import torch
import torch.nn as nn
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import smm_cuda


class SMM_QmK(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM) of
    query (Q) and key (K) matrices, based on given sparse indices.

    This function leverages a CUDA-implemented kernel for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Query matrix).
            B: Input tensor B (Key matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save input tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function for sparse matrix multiplication
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication QmK.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class SMM_AmV(Function):
    """
    A custom PyTorch autograd Function for sparse matrix multiplication (SMM)
    between an activation matrix (A) and a value matrix (V), guided by sparse indices.

    This function utilizes a CUDA-optimized implementation for efficient computation.

    Forward computation:
        Computes the sparse matrix multiplication using a custom CUDA function.

    Backward computation:
        Computes the gradients of A and B using a CUDA-implemented backward function.
    """

    @staticmethod
    def forward(ctx, A, B, index):
        """
        Forward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to save tensors for backward computation.
            A: Input tensor A (Activation matrix).
            B: Input tensor B (Value matrix).
            index: Index tensor specifying the sparse multiplication positions.

        Returns:
            Tensor: Result of the sparse matrix multiplication.
        """
        # Save tensors for backward computation
        ctx.save_for_backward(A, B, index)

        # Call the custom CUDA forward function
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """
        Backward function for Sparse Matrix Multiplication AmV.

        Args:
            ctx: Autograd context to retrieve saved tensors.
            grad_output: Gradient of the output from the forward pass.

        Returns:
            Tuple: Gradients of the inputs A and B, with None for the index as it is not trainable.
        """
        # Retrieve saved tensors from the forward pass
        A, B, index = ctx.saved_tensors

        # Compute gradients using the custom CUDA backward function
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )

        # Return gradients for A and B, no gradient for index
        return grad_A, grad_B, None


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.hidden_features, x_size[0], x_size[1])
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of input
        w (int): Width of input

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention (MSA).

    Args:
        dim (int): Number of input channels.
        layer_id (int): Index of the current layer in the network.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values retained for sparsity.
        qkv_bias (bool, optional): If True, add a learnable bias to the query, key, and value tensors. Default: True.
    """

    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        # define a parameter table of relative position bias
        if dim > 100:
            # for classical SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        else:
            # for lightweight SR
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        r"""
        Args:
        qkvp (Tensor): Input tensor containing query, key, value tokens, and LePE positional encoding matrix with shape (num_windows * b, n, c * 4),
        pfa_values (Tensor or None): Precomputed attention values for Progressive Focusing Attention (PFA). If None, standard attention is applied.
        pfa_indices (Tensor or None): Index tensor for Progressive Focusing Attention (PFA), indicating which attention values should be retained or discarded.
        rpi (Tensor): Relative position index tensor, encoding positional information for tokens.
        mask (Tensor or None, optional): Attention mask tensor.
        shift (int, optional): Indicates whether window shifting is applied (e.g., 0 for no shift, 1 for shifted windows). Default: 0.
        """
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        # Standard Attention Computation
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        # # Sparse Attention Computation using SMM_QmK
        else:
            topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:  # Check if in inference mode
                attn.add_(relative_position_bias)  # only in inference
            else:
                attn = attn + relative_position_bias  # Non-inplace if training

        # Use in-place operations where possible (only in inference mode)
        if not self.training:  # Check if in inference mode
            attn = torch.softmax(attn, dim=-1, out=attn)  # in-place softmax
        else:
            attn = self.softmax(attn)  # Non-inplace if training

        # Apply Hadamard product for PFA and normalize.
        if pfa_values[shift] is not None:
            if not self.training:  # only in inference
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        # If sparsification is enabled, select top-k attention values and save the corresponding indexes
        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        # Save the current attention results as PFA maps.
        pfa_values[shift] = attn

        # Check whether sparsification has been applied; if so, use SMM_AmV for computation, otherwise perform standard matrix multiplication A @ V.
        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads)+ v_lepe).transpose(1, 2).reshape(b_, n, c)

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        if self.layer_id < 2:
            flops += self.num_heads * n * (self.dim // self.num_heads) * n
            #  x = (attn @ v)
            flops += self.num_heads * n * n * (self.dim // self.num_heads)
        else:
            # attn = (q @ k.transpose(-2, -1))
            flops += self.num_heads * n * (self.dim // self.num_heads) * self.num_topk[self.layer_id-2]
            #  x = (attn @ v)
            flops += self.num_heads * n * self.num_topk[self.layer_id] * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class PFTransformerLayer(nn.Module):
    r"""
    PFT Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.layer_id = layer_id
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,act_layer=act_layer)

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x

        x = self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        # SW-MSA
        # cyclic shift
        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, h, w, c4), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)
        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, pfa_values, pfa_indices = self.attn_win(x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices, rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        x = shortcut + x_win.view(b, n, c)
        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        flops += self.dim * 3 * self.dim * h * w

        # W-MSA/SW-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)

        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * (self.convffn_kernel_size ** 2) * self.mlp_ratio
        # lepe
        flops += h * w * self.dim * (self.convlepe_kernel_size ** 2)
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic PFT Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Local window size.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 layer_id,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                PFTransformerLayer(
                    dim=dim,
                    block_id=idx,
                    layer_id= layer_id + i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, x_size, params):
        for layer in self.layers:
            # checkpoint_wrapper is not yet supported for PFT
            # idx_checkpoint = 4
            # if self.use_checkpoint and self.idx < idx_checkpoint:
            #     layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x, pfa_list = layer(x, pfa_list, x_size, params)

        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class PFTB(nn.Module):
    """Adaptive Token Dictionary Block (PFTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple(int)): Number of top-k attention values retained in different layers during attention computation.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        flood_map_size: Input flood map size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 idx,
                 layer_id,
                 input_resolution,
                 depth,
                 num_heads,
                 num_topk,
                 window_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 flood_map_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 ):
        super(PFTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            flood_map_size=flood_map_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            flood_map_size=flood_map_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            layer_id=layer_id,
            depth=depth,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, pfa_list, x_size, params):
        x_Basicblock, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_Basicblock, x_size))) + x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        flood_map_size (int): Flood map size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input flood map channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, flood_map_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        flood_map_size = to_2tuple(flood_map_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [flood_map_size[0] // patch_size[0], flood_map_size[1] // patch_size[1]]
        self.flood_map_size = flood_map_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.flood_map_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        flood_map_size (int): Flood map size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input flood map channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, flood_map_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        flood_map_size = to_2tuple(flood_map_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [flood_map_size[0] // patch_size[0], flood_map_size[1] // patch_size[1]]
        self.flood_map_size = flood_map_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat, use_upsample_act=False):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        self.use_upsample_act = use_upsample_act
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            n_stages = int(math.log(scale, 2))
            for i in range(n_stages):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
                if use_upsample_act and i < n_stages - 1:
                    m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
            # flops += self.num_feat * 4 * self.num_feat * 25 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
            # flops += self.num_feat * 9 * self.num_feat * 25 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.scale = scale
        self.num_feat = num_feat
        self.num_out_ch = num_out_ch
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = input_resolution
        flops = h * w* self.num_feat * (self.scale ** 2) * self.num_out_ch * 9
        return flops


def _build_static_ch_ladder(in_ch: int, out_ch: int, n_stages: int):
    if n_stages <= 0:
        raise ValueError(f"[_build_static_ch_ladder] n_stages must be >= 1, got {n_stages}")
    if n_stages == 1:
        return [int(in_ch), int(out_ch)]
    in_ch_f = max(float(in_ch), 1.0)
    out_ch_f = max(float(out_ch), 1.0)
    ratio = (out_ch_f / in_ch_f) ** (1.0 / n_stages)
    ladder = [int(in_ch)]
    for i in range(1, n_stages):
        c = int(round(in_ch_f * (ratio ** i)))
        c = max(4, ((c + 2) // 4) * 4)
        ladder.append(c)
    ladder.append(int(out_ch))
    return ladder


def _build_encoder_channels(in_ch: int, num_feat: int, n_stages: int):
    if n_stages <= 0:
        raise ValueError(f"[_build_encoder_channels] n_stages must be >= 1, got {n_stages}")
    presets = {
        1: [32, num_feat],
        2: [32, 48, num_feat],
        3: [32, 48, 64, num_feat],
        4: [24, 32, 48, 64, num_feat],
    }
    if n_stages in presets:
        return [int(c) for c in presets[n_stages]]
    stem = 32
    chs = [stem]
    for i in range(1, n_stages):
        r = i / n_stages
        c = int(round(stem * (max(num_feat, 1) / stem) ** r))
        c = max(8, ((c + 4) // 8) * 8)
        chs.append(c)
    chs.append(int(num_feat))
    return chs


class DecoderUp(nn.Module):
    def __init__(self, in_ch, out_ch, use_act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, 4 * out_ch, 3, 1, 1), nn.PixelShuffle(2)]
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class DecoderFuseBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_act=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, 1, 1)]
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        if use_act:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


@ARCH_REGISTRY.register()
class TGPFT(nn.Module):
    def __init__(self,
                 flood_map_size=64,
                 patch_size=1,
                 coarse_in_chans=1,
                 static_in_chans=7,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 num_topk=[256, 256,  128, 128, 128, 128,  64, 64, 64, 64, 64, 64,  32, 32, 32, 32, 32, 32,  16, 16, 16, 16, 16, 16],
                 window_size=8,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 upsampler='',
                 resi_connection='1conv',
                 use_shallow_act=True,
                 use_upsample_act=False,
                 use_decoder_act=True,
                 use_global_residual=True,
                 use_aoi_gate=True,
                 aoi_alpha=0.8,
                 couple_mode="detach",
                 couple_eps=0.2,
                 **kwargs):
        super().__init__()
        num_in_ch = coarse_in_chans
        num_out_ch = 1
        self.coarse_in_ch = coarse_in_chans
        self.static_in_ch = static_in_chans
        num_feat = 64
        self.upscale = upscale
        self.upsampler = upsampler

        self.use_shallow_act = use_shallow_act
        self.use_upsample_act = use_upsample_act
        self.use_decoder_act = use_decoder_act
        self.use_global_residual = use_global_residual
        self.use_aoi_gate = use_aoi_gate
        self.aoi_alpha = aoi_alpha

        self.couple_mode = str(couple_mode).lower().strip()
        self.couple_eps = couple_eps
        assert self.couple_mode in ("detach", "coupled", "none")

        if self.upsampler != 'pixelshuffle':
            raise RuntimeError(
                f'[ERROR] TGPFT only supports upsampler="pixelshuffle" '
                f'(got "{self.upsampler}"); the U-Net decoder needs multi-stage upsampling.')

        self.conv_coarse_fm = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        n_stages = int(math.log2(self.upscale))
        self.n_stages = n_stages
        enc_ch = _build_encoder_channels(self.static_in_ch, num_feat, n_stages)
        assert enc_ch[-1] == num_feat, f"encoder ladder must end at num_feat, got {enc_ch}"
        self._enc_ch = enc_ch

        stem_layers = [nn.Conv2d(self.static_in_ch, enc_ch[0], 3, 1, 1)]
        if self.use_shallow_act:
            stem_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.static_stem = nn.Sequential(*stem_layers)

        self.static_downs = nn.ModuleList()
        for i in range(n_stages):
            down = [nn.PixelUnshuffle(2),
                    nn.Conv2d(enc_ch[i] * 4, enc_ch[i + 1], 3, 1, 1)]
            if self.use_shallow_act:
                down.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.static_downs.append(nn.Sequential(*down))

        self._skip_ch = enc_ch[:n_stages]

        self.gn_coarse = nn.GroupNorm(num_groups=8, num_channels=num_feat)
        self.gn_static = nn.GroupNorm(num_groups=8, num_channels=num_feat)

        self.conv_first = nn.Conv2d(2 * num_feat, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.layer_id = 0
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split flood map into non-overlapping patches
        self.patch_embed = PatchEmbed(
            flood_map_size=flood_map_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into flood map
        self.patch_unembed = PatchUnEmbed(
            flood_map_size=flood_map_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        # build Residual Adaptive Token Dictionary Blocks (PFTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PFTB(
                dim=embed_dim,
                idx=i_layer,
                layer_id=self.layer_id,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads,
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                flood_map_size=flood_map_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
            self.layer_id = self.layer_id + depths[i_layer]

        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            # self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 5, 1, 2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        self.dec_proj = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)

        skip_ch_dec = list(reversed(self._skip_ch))
        self.dec_ups = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        for i in range(self.n_stages):
            self.dec_ups.append(DecoderUp(num_feat, num_feat, use_act=self.use_decoder_act))
            self.dec_fuse.append(
                DecoderFuseBlock(num_feat + skip_ch_dec[i], num_feat, use_act=self.use_decoder_act))

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.flood_head = nn.Conv2d(num_feat, 3, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])

        # Define progressive focusing attention (PFA) values and their corresponding indices
        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        flood_map_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                flood_map_mask[:, hs, ws, :] = cnt
                cnt += 1

        mask_windows = window_partition(flood_map_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, coarse_fm, static_f):
        B, _, Hc_orig, Wc_orig = coarse_fm.shape
        _, _, Hf_orig, Wf_orig = static_f.shape

        if (Hf_orig != Hc_orig * self.upscale) or (Wf_orig != Wc_orig * self.upscale):
            raise RuntimeError(
                f'[ERROR] fine/coarse size mismatch with upscale: '
                f'Hc={Hc_orig}, Wc={Wc_orig}, Hf={Hf_orig}, Wf={Wf_orig}, scale={self.upscale}'
            )

        mod = self.window_size
        Hc_pad = ((Hc_orig + mod - 1) // mod) * mod - Hc_orig
        Wc_pad = ((Wc_orig + mod - 1) // mod) * mod - Wc_orig

        if (Hc_pad != 0) or (Wc_pad != 0):
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [2])], 2)[:, :, :Hc_orig + Hc_pad, :]
            coarse_fm = torch.cat([coarse_fm, torch.flip(coarse_fm, [3])], 3)[:, :, :, :Wc_orig + Wc_pad]
            Hf_pad = Hc_pad * self.upscale
            Wf_pad = Wc_pad * self.upscale
            static_f = torch.cat([static_f, torch.flip(static_f, [2])], 2)[:, :, :Hf_orig + Hf_pad, :]
            static_f = torch.cat([static_f, torch.flip(static_f, [3])], 3)[:, :, :, :Wf_orig + Wf_pad]

        Hc, Wc = Hc_orig + Hc_pad, Wc_orig + Wc_pad

        attn_mask = self.calculate_mask([Hc, Wc]).to(coarse_fm.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        coarse_h_pad = coarse_fm[:, 0:1, :, :]

        if self.use_aoi_gate:
            aoi_mask_fine = static_f[:, -1:, :, :]
            aoi_mask_coarse = torch.nn.functional.adaptive_avg_pool2d(aoi_mask_fine, (Hc, Wc))
            gate = (1.0 - self.aoi_alpha) + self.aoi_alpha * aoi_mask_coarse
            gate = gate.clamp(0.0, 1.0).detach()
        else:
            gate = None

        s = self.static_stem(static_f)
        skips = [s]
        for down in self.static_downs:
            s = down(s)
            skips.append(s)
        static_deep = skips[-1]
        enc_skips = skips[:-1]

        coarse_feat = self.conv_coarse_fm(coarse_fm)
        coarse_feat = self.gn_coarse(coarse_feat)
        static_deep = self.gn_static(static_deep)
        x = self.conv_first(torch.cat([coarse_feat, static_deep], dim=1))
        x_rc = x

        if gate is not None:
            x = x * gate
        x = self.conv_after_body(self.forward_features(x, params)) + x_rc
        if gate is not None:
            x = x * gate

        d = self.dec_proj(x)
        skips_dec = list(reversed(enc_skips))
        for i in range(self.n_stages):
            d_up = self.dec_ups[i](d)
            fused = self.dec_fuse[i](torch.cat([d_up, skips_dec[i]], dim=1))
            d = fused + d_up

        depth = self.conv_last(d)
        flood_logit = self.flood_head(d)

        if self.use_global_residual:
            coarse_h_up = torch.nn.functional.interpolate(
                coarse_h_pad, scale_factor=self.upscale, mode='bicubic', align_corners=False)
            depth = depth + coarse_h_up

        depth = depth[..., :Hf_orig, :Wf_orig]
        flood_logit = flood_logit[..., :Hf_orig, :Wf_orig]

        if self.couple_mode != "none":
            p_wet = torch.sigmoid(flood_logit[:, 0:1, :, :])
            if self.couple_mode == "detach":
                p_wet = p_wet.detach()
            depth = depth * (self.couple_eps + (1 - self.couple_eps) * p_wet)

        return depth, flood_logit

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        num_feat = 64
        Hf, Wf = h * self.upscale, w * self.upscale

        flops += h * w * self.coarse_in_ch * num_feat * 9

        enc_ch = self._enc_ch
        flops += Hf * Wf * self.static_in_ch * enc_ch[0] * 9
        curH, curW = Hf, Wf
        for i in range(self.n_stages):
            curH //= 2
            curW //= 2
            flops += curH * curW * (enc_ch[i] * 4) * enc_ch[i + 1] * 9

        flops += h * w * (2 * num_feat) * self.embed_dim * 9

        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += h * w * self.embed_dim * self.embed_dim * 9

        flops += h * w * self.embed_dim * num_feat * 9
        skip_ch_dec = list(reversed(self._skip_ch))
        curH, curW = h, w
        for i in range(self.n_stages):
            flops += curH * curW * num_feat * (4 * num_feat) * 9
            curH *= 2
            curW *= 2
            in_c = num_feat + skip_ch_dec[i]
            flops += curH * curW * in_c * num_feat * 9
            flops += curH * curW * num_feat * num_feat * 9

        flops += Hf * Wf * num_feat * 1 * 9
        flops += Hf * Wf * num_feat * 3 * 1

        return flops


if __name__ == '__main__':
    upscale = 8
    model = TGPFT(
        upscale=upscale,
        flood_map_size=64,
        coarse_in_chans=3,
        static_in_chans=7,
        embed_dim=96,
        depths=[2, 4, 6, 6],
        num_heads=6,
        num_topk=[256, 256,
                  128, 128, 128, 128,
                  64, 64, 64, 64, 64, 64,
                  32, 32, 32, 32, 32, 32,],
        window_size=16,
        convffn_kernel_size=7,
        mlp_ratio=2,
        upsampler='pixelshuffle',
        use_shallow_act=True,
        use_decoder_act=True,
        use_global_residual=True,
        use_aoi_gate=True,
        aoi_alpha=0.5,
        couple_mode='detach',
    )

    # Parameters and computational complexity
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    print("encoder skip channels (enc_ch):", model._enc_ch)
    print("n_stages:", model.n_stages)
    print("approx GFLOPs @ coarse 64x64:", model.flops((64, 64)) / 1e9)
