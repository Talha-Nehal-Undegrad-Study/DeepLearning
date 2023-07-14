# 1. Create a ViT class that inherits from nn.Module which implements the complete vit architecture according to the original paper

import torch
from torch import nn
import subprocess
import sys
# Get relevant patch, msa, mlp and transformer encoder scripts

try:
    import patch, msa, mlp, transformer_encoder
except ImportError:
    print("[INFO] Cloning the repository and importing utils script...")
    subprocess.run(["git", "clone", "https://github.com/TalhaAhmed2000/DeepLearning.git"])
    subprocess.run(["mv", "DeepLearning/Task 4/python_scripts", "py_scripts"])
    sys.path.append('py_scripts')
    import patch, msa, mlp, transformer_encoder

class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int = 224, # Training resolution from Table 3 in ViT paper
                 in_channels:int = 3, # Number of channels in input image
                 patch_size:int = 16, # Patch size
                 num_transformer_layers:int = 12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int = 768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int = 3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int = 12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float = 0, # Dropout for attention projection
                 mlp_dropout:float = 0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float = 0.1, # Dropout for patch and position embeddings
                 num_classes:int = 1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 3. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size ** 2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim),
                                            requires_grad = True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data = torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad = True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p = embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = patch.PatchEmbedding(in_channels = in_channels,
                                              patch_size = patch_size,
                                              embedding_dim = embedding_dim)

        # 9. Create Transformer Encoder blocks
        self.transformer_encoder = nn.Sequential(*[transformer_encoder.TransformerEncoderBlock(embedding_dim = embedding_dim,
                                                                            num_heads = num_heads,
                                                                            mlp_size = mlp_size,
                                                                            mlp_dropout = mlp_dropout) for _ in range(num_transformer_layers)])

        # 10. Create classifier head --> Eq.4
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape = embedding_dim),
            nn.Linear(in_features = embedding_dim,
                      out_features = num_classes)
        )

    # 11. Create a forward() method
    def forward(self, x):

        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" 

        # 14. Create patch embedding 
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding
        x = torch.cat((class_token, x), dim = 1)

        # 16. Add position embedding to patch embedding
        x = self.position_embedding + x

        # 17. Run embedding dropout 
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
