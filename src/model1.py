
# from keras import Model
# from keras.layers import Dense, Dropout, Flatten, Conv2D
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18, resnet50, resnext50_32x4d


# -------------- ResNet ------------
# MNIST
#https://zablo.net/blog/post/pytorch-resnet-mnist-jupyter-notebook-2021/
class resnet18MNIST(nn.Module):
    def __init__(self, in_channel = 1):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(in_channel,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

class resnet50MNIST(nn.Module):
    def __init__(self, in_channel = 1):
        super().__init__()
        self.model = resnet50(num_classes=10)
        self.model.conv1 = nn.Conv2d(in_channel,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

# -------------- ResNext ------------
# MNIST in anlehnung an https://zablo.net/blog/post/pytorch-resnet-mnist-jupyter-notebook-2021/
class resnext50_32x4d_MNIST(nn.Module):
    def __init__(self, in_channel = 1):
        super().__init__()
        self.model = resnext50_32x4d(num_classes=10)
        self.model.conv1 = nn.Conv2d(in_channel,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self,x):
        return self.model(x)

# -------------- CNNs ------------
# MNIST


class CNN(nn.Module):
    """Basic CNN architecture."""

    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 28, 28) --> (batch_size, 64, 21, 21)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 21, 21) --> (batch_size, 128, 8, 8)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 8, 8) --> (batch_size, 128, 4, 4)
        self.fc1 = nn.Linear(
            128 * 4 * 4, 128
        )  # (batch_size, 128, 4, 4) --> (batch_size, 2048)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class PyNet(nn.Module):
    """CNN architecture. This is the same MNIST model from 
    pytorch/examples/mnist repository"""

    def __init__(self, in_channels=1):
        super(PyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x=x.reshape(1,1,28,28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class PyNetSoftmax(nn.Module):
    """CNN architecture. This is the same MNIST model from 
    pytorch/examples/mnist repository"""

    def __init__(self, in_channels=1):
        super(PyNetSoftmax, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output


# CIFAR-10 - by 
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


class NetCifar(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------- Transformer ------------
# MNIST


# CIFAR-10 - Impl by
# https://github.com/pytorch/examples/blob/main/vision_transformer/main.py


class PatchExtractor(nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, input_data):
        batch_size, channels, height, width = input_data.size()

        assert height % self.patch_size == 0 and width % self.patch_size == 0

        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        patches = input_data.unfold(2, self.patch_size, self.patch_size). \
            unfold(3, self.patch_size, self.patch_size). \
            permute(0, 2, 3, 1, 4, 5). \
            contiguous(). \
            view(batch_size, num_patches, -1)

        # Expected shape of a patch on default settings is (4, 196, 768)

        return patches


class InputEmbedding(nn.Module):

    def __init__(self, args):
        super(InputEmbedding, self).__init__()
        self.patch_size = args.patch_size
        self.n_channels = args.n_channels
        self.latent_size = args.latent_size
        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.batch_size = args.batch_size
        self.input_size = self.patch_size * self.patch_size * self.n_channels

        # Linear projection
        self.LinearProjection = nn.Linear(self.input_size, self.latent_size)
        # Class token
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, 
                                                    self.latent_size)).to(
            self.device)
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(self.batch_size, 1,
                                                      self.latent_size)).to(
            self.device)

    def forward(self, input_data):
        input_data = input_data.to(self.device)
        # Patchifying the Image
        patchify = PatchExtractor(patch_size=self.patch_size)
        patches = patchify(input_data)

        linear_projection = self.LinearProjection(patches).to(self.device)
        b, n, _ = linear_projection.shape
        linear_projection = torch.cat((self.class_token, linear_projection),
                                      dim=1)
        pos_embed = self.pos_embedding[:, :n + 1, :]
        linear_projection += pos_embed

        return linear_projection


class EncoderBlock(nn.Module):

    def __init__(self, args):
        super(EncoderBlock, self).__init__()

        self.latent_size = args.latent_size
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.norm = nn.LayerNorm(self.latent_size)
        self.attention = nn.MultiheadAttention(self.latent_size, 
                                               self.num_heads, dropout=self.
                                               dropout)
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size * 4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size * 4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, emb_patches):
        first_norm = self.norm(emb_patches)
        attention_out = self.attention(first_norm, first_norm, first_norm)[0]
        first_added = attention_out + emb_patches
        second_norm = self.norm(first_added)
        mlp_out = self.enc_MLP(second_norm)
        output = mlp_out + first_added

        return output


class ViT32(nn.Module):
    def __init__(self, args):
        super(ViT32, self).__init__()

        self.num_encoders = args.num_encoders
        self.latent_size = args.latent_size
        self.num_classes = args.num_classes
        self.dropout = args.dropout

        self.embedding = InputEmbedding(args)
        # Encoder Stack
        self.encoders = nn.ModuleList([EncoderBlock(args) 
                                       for _ in range(self.num_encoders)])
        self.MLPHead = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes),
        )

    def forward(self, test_input):
        enc_output = self.embedding(test_input)
        for enc_layer in self.encoders:
            enc_output = enc_layer(enc_output)

        class_token_embed = enc_output[:, 0]
        return self.MLPHead(class_token_embed)

# MNIST - Impl --- Impl by https://www.kaggle.com/code/fold10/mnist-vision-transformer-vit/notebook


# class Attention(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.scale = dim ** -0.5
#         self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

#     def forward(self, x):
#         b, n, _, h = *x.shape, 2
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d',
#                                           h=h), qkv)
#         dots = einsum('b h i d, b h j d -> b h i j',
#                       q, k) * self.scale
#         attn = dots.softmax(dim=-1)
#         out = einsum('b h i j, b h j d -> b h i d',
#                      attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return out


# class Transformer(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.attention = Attention(32)
#         self.norm1 = nn.LayerNorm(32)
#         self.fc1 = nn.Linear(32, 32)
#         self.norm2 = nn.LayerNorm(32)

#     def forward(self, x):
#         out = nn.functional.relu(self.attention(self.norm1(x)) + x)
#         out = nn.functional.relu(self.fc1(self.norm2(out)) + out)
#         return out


# class MNISTTransformer(nn.Module):
#     def __init__(self, depth):
#         super().__init__()
#         image_size = 28
#         patch_size = 7
#         num_patches = (image_size // patch_size) ** 2
#         patch_dim = patch_size ** 2
#         self.to_patches = lambda x: rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
#                                               p1=patch_size, p2=patch_size)
#         self.embedding = nn.Linear(patch_dim, 32)
#         self.pos_embedding = nn.Parameter(
#             torch.randn(1, num_patches + 1, 32))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, 32))
#         self.features = nn.Sequential()
#         for i in range(depth):
#             self.features.append(Transformer())
#         self.classifier = nn.Linear(32, 10)

#     def forward(self, x):
#         patches = self.to_patches(x)
#         x = self.embedding(patches)
#         b, n, _ = x.shape
#         cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         out = self.features(x)[:, 0]
#         return self.classifier(out).flatten(1)
