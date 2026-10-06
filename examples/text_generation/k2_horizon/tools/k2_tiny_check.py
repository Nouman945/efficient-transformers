"""
Tiny random K2 Horizon model (remote code, reduced config) through the QEff stack:
HF PyTorch -> QEff KV PyTorch -> ONNX export -> ONNX Runtime, token match on all three.
Fast mechanics check for the wrapper before spending time on the real 7B weights.
"""

import copy
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

MODEL = "IFM/K2-Horizon-7B"

torch.manual_seed(0)
config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
config.hidden_size = 128
config.intermediate_size = 256
config.num_hidden_layers = 2
config.num_attention_heads = 4
config.num_key_value_heads = 2
config.head_dim = 32
config.rope_head_dim = 32
config.layernorm_num_groups = 4
config.max_position_embeddings = 256
config.mlp_only_layers = [0, 1]
config.query_key_norm = True
config.attention_gate_func = "softplus"
config.dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model_hf = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager").eval()
model_hf.config.torch_dtype = torch.float32
print("HF classes:", type(model_hf).__name__, type(model_hf.model.layers[0].self_attn).__name__)

prompt = "My name is"
runner = ApiRunner(1, tokenizer, config, [prompt], 8, 24)
hf_tokens = np.asarray(runner.run_hf_model_on_pytorch_CB(copy.deepcopy(model_hf))[0]).reshape(-1)

qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=MODEL)
attn = qeff_model.model.model.layers[0].self_attn
print("QEff attention forward:", attn.forward.__func__.__module__)
assert "k2_horizon" in attn.forward.__func__.__module__, "transforms did not apply"
kv_tokens = np.asarray(runner.run_kv_model_on_pytorch(qeff_model.model)).reshape(-1)
print("HF vs KV:", (hf_tokens == kv_tokens).all(), hf_tokens.tolist(), kv_tokens.tolist())

onnx_path = qeff_model.export()
ort_tokens = np.asarray(runner.run_kv_model_on_ort(str(onnx_path))).reshape(-1)
print("HF vs ORT:", (hf_tokens == ort_tokens).all(), ort_tokens.tolist())
print("ONNX:", onnx_path)
sys.exit(0 if (hf_tokens == kv_tokens).all() and (hf_tokens == ort_tokens).all() else 1)
