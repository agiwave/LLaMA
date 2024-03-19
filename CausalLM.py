import aka.nn as nn
import aka.numpy as np

def RMSNorm(dim: int, eps: float = 1e-5):
    '''
    Reference: LLaMA and Gemma
    '''
    def forward(self, x):
        x = (x.float() * np.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return x * self.weight
    return nn.Module(
        forward = forward,
        eps = eps,
        weight = nn.Parameter(np.ones(dim)))

def MLPBlock(args):
    '''
    Reference: Gemma, LLaMA
    Common ver:
        (b,l,latent_dim) --up--> (b,l,kv_size, kv_size) --down--> (b, l, latent_dim)
    Full ver:
        (b,l,latent_dim) --in--> (b,l,qk_dim) --up--> (b,l,kv_size, kv_size) 
        --down--> (b, l, hidden_dim) --out--> (b,l,latent_dim)
    Args:
        args.mlp_args.kv_size = 384*4,
        args.mlp_args.kv_gate = False,
        args.mlp_args.qk_dim = 384,
        args.mlp_args.hidden_dim = 384,
        args.mlp_args.num_heads = 6,      # not support.
        args.mlp_args.num_kv_groups = 6,  # not support.
        args.bias = False
    Examples:
        args.mlp_gate == True ==> GateMLP
    '''
    def forward(self, x, **kwargs):
        if self.in_proj is not None:
            x = self.in_proj(x)
        up = self.up_proj(x)
        if(self.gate_proj is not None):
            gate = self.gate_proj(x)
            gate = np.gelu(gate)    # silu LLaMA ?
            up = gate * up
        else:
            up = np.gelu(up)
        down = self.down_proj(up)
        if self.out_proj is not None:
            down = self.out_proj(down)
        return down

    # -- Global Args --
    latent_dim = args.latent_dim
    bias = getattr(args,'bias', False)

    # -- MLP Args
    args = args.mlp_args
    kv_size = getattr(args, 'kv_size', latent_dim)
    kv_gate = getattr(args, 'kv_gate', False)
    qk_dim = getattr(args, 'qk_dim', latent_dim)
    hidden_dim = getattr(args, 'hidden_dim', latent_dim)
    return nn.Module(
        forward = forward,
        in_proj = None if qk_dim == latent_dim else nn.Linear(latent_dim, qk_dim, bias=bias),   # Q
        up_proj = nn.Linear(qk_dim, kv_size, bias=bias),                                        # K(reversed)
        gate_proj = None if not kv_gate else nn.Linear(qk_dim, kv_size, bias=bias),             # G or mask
        down_proj = nn.Linear(kv_size, hidden_dim, bias=bias),                                  # V
        out_proj = None if hidden_dim == latent_dim else nn.Linear(hidden_dim, args.latent_di, bias=bias)
    )

def MetaLayer(name, args):
    '''
    Build resident meta layer by name. Include: GQA(Group-Query Attention), MLP, GateMLP, ...
    '''
    def forward(self, x, **kwargs):
        y = self.norm(x)
        return x + self.layer(y, **kwargs)

    match name:
        case 'Attention':
            from Attention import AttentionBlock
            m = AttentionBlock(args)
        case 'MLP':
            m = MLPBlock(args)
        case 'GateMLP':
            kv_gate = getattr(args, 'kv_gate', False)
            args.kv_gate = True
            m = MLPBlock(args)
            args.kv_gate = kv_gate
        case 'Mamba':
            from Mamba import MambaBlock
            m = MambaBlock(args)
        case _:
            assert False, f"Unknown layer:{name}"

    return nn.Module(
        forward = forward,
        norm = RMSNorm(args.latent_dim),
        layer = m
    )

def CausalLM(args):
    '''
    Causal Language Model.
    '''
    def forward(self, inputs, targets=None, state=None):
        _, L = inputs.shape
        assert L-1 <= self.block_size, f"Input size:{L} too large. Max size: {self.block_size-1}"

        x = inputs

        # -- Shift inputs and targets --
        if(targets is not None):
            t = x[:,1:]
            x = x[:,:L-1]

        # -- Embedding and layers
        x = self.embedding(x)
        if self.in_proj is not None:
            x = self.in_proj(x)
        if self.prev_norm:
            x = x * (x.size(-1)**0.5)   # -- Gemma, Why? --
        if self.pe is not None:
            x = x + pe
        freqs_cls = self.freqs_cis
        enable_cache = self.enable_cache
        if(state is not None):
            if('layer_states' in state):
                layer_states = state['layer_states']
            else:
                layer_states = [{} for _ in self.layers]
                state['layer_states'] = layer_states
            for i in range(len(self.layers)):
                x = self.layers[i](x, freqs_cis=freqs_cls, state=layer_states[i])
        else:
            for l in self.layers:
                x = l(x, freqs_cis=freqs_cls)
        x = self.post_norm(x)
        if self.out_proj is not None:
            x = self.out_proj(x)

        if self.output is not None:
            y = self.output(x)    # -- LLaMA vs embedding.weight ? --
        else:
            y = np.einsum('bld,nd->bln', x, self.embedding.weight)
        if(targets is not None):
            loss = np.cross_entropy(y.view(-1, y.size(-1)), t.reshape(-1), ignore_index=-1)
            return y, loss
        else:
            return y

    def generate(self, prompts : str, max_length : int = 64):
        prompt_tokens = [self.tokenizer.bos_token_id]+self.tokenizer.encode(prompts)
        print('prompt_tokens', len(prompt_tokens))
        if hasattr(self, 'eval'):
            self.eval()

        with np.no_grad():
            if self.enable_cache:
                state = {}
                for i in range(len(prompt_tokens)):
                    outputs = self(np.array([[prompt_tokens[i]]]), state=state)
                    output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)

                response_token_ids = output_token_ids
                for _ in range(max_length):
                    outputs = self(output_token_ids, state=state)
                    output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                    response_token_ids = np.cat((response_token_ids, output_token_ids), dim=1)
                    if self.tokenizer.eos_token_id in output_token_ids:
                        break
            else:
                input_token_ids = np.array([prompt_tokens])
                for _ in range(max_length):
                    outputs = self(input_token_ids)
                    output_token_ids = np.argmax(outputs[:,-1:,:], dim=-1)
                    input_token_ids = np.cat((input_token_ids, output_token_ids), dim=1)
                    if self.tokenizer.eos_token_id in output_token_ids:
                        break
                response_token_ids = input_token_ids[:,len(prompt_tokens):]

        response_tokens = response_token_ids.squeeze(0).tolist()
        return self.tokenizer.decode(response_tokens)

    # -- Reference: LLaMA and Gemma， Could be learned automaticlly? --
    def precompute_freqs_cis(dim: int,
                            end: int,
                            theta: float = 10000.0):
        """Precomputes the frequency cis."""
        freqs = 1.0 / (theta**(np.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        t = np.arange(end, device=freqs.device)
        freqs = np.outer(t, freqs).float()
        freqs_cis = np.polar(np.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    freqs_cis = None
    if getattr(args, 'rotary_embedding', False):
        # Pre-compute rotary embedding table.
        rope_theta = getattr(args.attn_args, 'rope_theta', 10000)
        attn_hidden_dim = getattr(args.attn_args, 'qk_dim', args.latent_dim)
        attn_heads = getattr(args.attn_args, 'num_heads', 1)
        freqs_cis = precompute_freqs_cis(
                            attn_hidden_dim//attn_heads,
                            args.block_size,
                            theta=rope_theta)
    
    pe = None
    if getattr(args, 'position_embedding', False):
        pe = nn.Parameter(np.rand(args.block_size, args.latent_dim), require_grads=True)

    in_proj, out_proj = None, None
    vocab_dim = getattr(args, 'vocab_dim', args.latent_dim)
    if vocab_dim != args.latent_dim:
        in_proj = nn.Linear(vocab_dim, args.latent_dim, bias=args.bias)
        out_proj = nn.Linear(args.latent_dim, vocab_dim, bias=args.bias)

    lm_head = getattr(args, 'lm_head', False)
    make_layer = MetaLayer if not hasattr(args, 'MetaLayer') else args.MetaLayer
    return nn.Module(
        forward = forward,
        generate = generate,
        tokenizer = args.tokenizer,
        block_size = args.block_size,
        embedding = nn.Embedding(num_embeddings=args.vocab_size, embedding_dim=vocab_dim),
        layers = nn.ModuleList([make_layer(key, args) for key in args.layers]),
        in_proj = in_proj,
        out_proj = out_proj,
        output = None if not lm_head else nn.Linear(vocab_dim, args.vocab_size,bias=False),
        prev_norm = getattr(args, 'prev_norm', False),
        post_norm = RMSNorm(args.latent_dim),
        pe = pe,
        enable_cache = getattr(args, 'enable_cache', False),
        freqs_cis = freqs_cis
    )

