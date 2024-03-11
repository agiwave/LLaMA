import aka.nn as nn
import aka.numpy as np

def LLaMA(name):
    import os, json
    from transformers import AutoTokenizer
    from CausalLM import CausalLM

    tokenizer = AutoTokenizer.from_pretrained(name)
    if os.path.exists(name+"/config.json") == False:
        assert False, f"Can't find config file:{name+'/config.json'}"
        
    cfg = json.load(open(name+'/config.json'))  # Can't use AutoConfig here for ver reason.
    class Args():
        def __init__(self, **kwargs): 
            for key in kwargs:
                setattr(self, key, kwargs[key])

    # -- TODO Check it --
    args = Args(
        tokenizer = tokenizer,
        vocab_size = 32000,
        block_size = 2048,
        latent_dim = 4096,
        rotary_embedding = True,
        lm_head = True,

        layers = ['Attention', 'MLP']*32,
        attn_args = Args(
            num_heads = 8,
            num_kv_groups = 1
        ),
        mlp_args = Args(
            kv_size = 4096*4,   # ???
            kv_gate = True,
        ),
        dropout = 0.2,
        bias = False,
        rope_theta = 10000,
    )

    from CausalLM import CausalLM
    llama = CausalLM(args)
    if os.path.exists(name+"/model.safetensors"):
        def copy(desc, src):
            if not (desc.shape == src.shape):
                print("Unmatch shape was found.", desc.shape, src.shape)
                assert False
            desc.copy_(src)

        from safetensors import safe_open
        with safe_open(name+"/model.safetensors", framework="pt") as f:
            with np.no_grad():
                copy(llama.embedding.weight, f.get_tensor('backbone.tok_embeddings.weight'))
                copy(llama.post_norm.weight, f.get_tensor(f'backbone.norm.weight'))
                copy(llama.lm_head.weight, f.get_tensor(f'backbone.output.weight'))
                for i in range(len(llama.layers)//2):
                    copy(llama.layers[i*2].attention_norm.weight, f.get_tensor(f'backbone.layers.{i}.attention_norm.weight'))
                    copy(llama.layers[i*2].layer.c_proj.weight, f.get_tensor(f'backbone.layers.{i}.attention.wo.weight'))
                    wq = f.get_tensor(f'backbone.layers.{i}.attention.wo.weight')
                    wk = f.get_tensor(f'backbone.layers.{i}.attention.wk.weight')
                    wv = f.get_tensor(f'backbone.layers.{i}.attention.wv.weight')
                    copy(llama.layers[i*2].layer.c_attn.weight, np.cat((wq,wk,wv), dim=1))
                    copy(llama.layers[i*2+1].ffn_norm.weight, f.get_tensor(f'backbone.layers.{i}.ffn_norm.weight'))
                    copy(gemma.layers[i*2+1].layer.gate_proj.weight, state[f'model.layers.{i}.feed_forward.w1.weight'])
                    copy(gemma.layers[i*2+1].layer.up_proj.weight, state[f'model.layers.{i}.feed_forward.w3.weight'])
                    copy(gemma.layers[i*2+1].layer.down_proj.weight, state[f'model.layers.{i}.feed_forward.w2.weight'])
    return llama

if __name__ == "__main__":
    mamba = Mamba('data/Llama-2-7b-chat-hf')
    print('Model loaded')
    print(mamba.generate("Mamba is"))
