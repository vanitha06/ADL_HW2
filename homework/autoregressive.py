import abc

import torch
from torch.nn import ReLU


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        self.n_codebook = n_tokens
        self.embed_dim = d_latent

        # 1. Token Embedding: Maps discrete codebook indices to continuous vectors
        self.embedding = torch.nn.Embedding(self.n_codebook, self.embed_dim)
        
        # # 2. Positional Embedding: Learnable parameters for a 16x16 grid (256 tokens)
        # self.pos_emb = torch.nn.Parameter(torch.zeros(1, 1024, self.embed_dim))

        # Learnable "Start of Sequence" embedding for Hint 3
        self.sos_token = torch.nn.Parameter(torch.zeros(1,self.embed_dim))

        # 3. Transformer "Decoder": Using EncoderLayer with causal masking
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=4, 
            dim_feedforward=self.embed_dim * 4,
            dropout=0,
            batch_first=True,
            norm_first=True,
            activation='relu',
        )
        self.norm=torch.nn.LayerNorm(self.embed_dim)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=3,norm=self.norm)

        # 4. Predictor: Maps transformer output back to codebook size for distribution
        self.predictor = torch.nn.Linear(self.embed_dim, self.n_codebook)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
       orig_shape = x.shape
      #  if x.dim() == 3:
       x = x.flatten(1)  # (B, L)
      #  x=x.view(-1)
       B, L = x.shape

        # Embed the values
       x_emb = self.embedding(x)  # (B, L, D)

        # Hint 3: Shift the input sequence by 1 position.
  
       sos = self.sos_token.expand(B, -1, -1)
       x_shifted = torch.cat([sos, x_emb[:, :-1, :]], dim=1) # (B, L, D)

      #   # Hint 2: Add positional embedding
      #  x_shifted = x_shifted + self.pos_emb[:, :L, :]

        # Apply causal mask to be safe (prevents looking ahead in the shifted seq)
       mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        
        # Pass through model
       h = self.transformer(x_shifted, mask=mask, is_causal=True) # (B, L, D)


        # Map to logits and reshape back to (B, H, W, n_codebook)
       logits = self.predictor(h)
      #  if len(orig_shape) == 3:
      #       logits = logits.view(B, orig_shape[1], orig_shape[2], self.n_codebook)
       return logits,{}

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
       self.eval()
       seq_len = h * w
        # Start with an empty sequence; the shift logic in forward handles the "start" token
       generated = torch.zeros((B, h, w), dtype=torch.long, device=device)
        
        # Note: For generation, you typically fill the tensor index by index
        # This is a simplified greedy loop
       for i in range(h):
            for j in range(w):
                with torch.no_grad():
                    logits = self.forward(generated)
                    # Get probabilities for the current pixel (i, j)
                    probs = torch.softmax(logits[:, i, j, :], dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated[:, i, j] = next_token.squeeze(-1)
              
       return generated
                
