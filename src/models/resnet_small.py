import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Ortho(nn.Module):

    def __init__(self, input_size, dim):
        super(Ortho, self).__init__()
        # [input_size, dim]
        self.M = torch.empty(input_size, dim, requires_grad=True)
        nn.init.xavier_normal_(self.M)
        self.input_size = input_size
        self.dim = dim
    
    def get_null_basis(self):
        '''Return a basis for the left null space of self.M.'''

        # U has shape [input_size, input_size]
        U, S, Vt = torch.svd(self.M, some=False)

        # We assume that self.M is full-rank (so it uses up the first self.dim rows in U)
        null_basis = U[:self.input_size - self.dim]
        return null_basis

    def get_null_proj_matrix(self):
        '''Return a matrix that projects an input onto the null space of self.M'''

        # Get a basis for the null space of self.M (assume self.M is rank self.dim)
        # [input_size - dim, input_size]
        null_basis = self.get_null_basis()

        # We can do this because A (A^T A)^-1 A^T reduces to A A^T when the rows of A are orthonormal (and A^T A = I)
        # [input_size, input_size]
        null_proj_matrix = torch.matmul(null_basis.T, null_basis)
        return null_proj_matrix

    def forward(self, x):
        self.M = self.M.to(x.device)
        '''x has dimension [batch_size, input_size, ...]'''
        if x.dim() > 2:
            # Mean pool along other dimensions
            # E.g. if shape is [batch_size, num_filters, x, y], this pools along x and y
            pool_dims = list(range(2, x.dim()))
            x_pooled = x.mean(dim=pool_dims)

        # [batch_size, input_size] x [input_size, dim] -> [batch_size, dim]
        layer_repr = torch.matmul(x_pooled, self.M) 

        # [input_size, input_size]
        null_proj_matrix = self.get_null_proj_matrix()
        
        # [batch_size, ..., input_size]
        x = x.transpose(1, -1)  # Shift input_size dimension to end for matmul.
        #  Project each hypercolumn (channel features for each x,y point) onto the null space.
        # [batch_size, ..., input_size] x [input_size, input_size] -> [batch_size, ..., input_size]
        nulled_x = torch.matmul(x, null_proj_matrix)
        # [batch_size, input_size, ...]
        nulled_x = nulled_x.transpose(1, -1)

        # Cut gradients. TODO: revisit?
        nulled_x = nulled_x.detach()

        return nulled_x, layer_repr


class ResNetNull(nn.Module):
    def __init__(self, block, num_blocks, output_dim=128):
        super(ResNetNull, self).__init__()
        self.in_planes = 64
        self.output_dim = output_dim
        self.num_layers = 4

        if self.output_dim % self.num_layers:
            raise ValueError(f"output_dim={self.output_dim} must be a multiple of {self.num_layers}")
        self.ortho_dim = self.output_dim // self.num_layers

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        # Add the orthogonalization layer.
        layers.append(Ortho(planes, self.ortho_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x, layer=7):
        if layer <= 0: 
            return x
        out = F.relu(self.bn1(self.conv1(x)))
        if layer == 1:
            return out
        out = self.layer1(out)
        if layer == 2:
            return out
        out = self.layer2(out)
        if layer == 3:
            return out
        out = self.layer3(out)
        if layer == 4:
            return out
        out = self.layer4(out)
        if layer == 5:
            return out
        # [batch_size, output_dim]
        out = torch.cat([ortho1, ortho2, ortho3, ortho4], dim=-1)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_channels=3, input_size=32):
        super(ResNet, self).__init__()
        assert input_size in [32, 64]
        self.in_planes = 64
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        fc_input_size = 512 * block.expansion * (4 if input_size == 64 else 1)
        self.fc = nn.Linear(fc_input_size, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=7):
        if layer <= 0: 
            return x
        out = F.relu(self.bn1(self.conv1(x)))
        if layer == 1:
            return out
        out = self.layer1(out)
        if layer == 2:
            return out
        out = self.layer2(out)
        if layer == 3:
            return out
        out = self.layer3(out)
        if layer == 4:
            return out
        out = self.layer4(out)
        if layer == 5:
            return out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if layer == 6:
            return out
        out = self.fc(out)
        return out


def ResNet18(num_classes, num_channels=3, input_size=32):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, num_channels=num_channels, 
                  input_size=input_size)

def ResNetNull18(num_classes):
    return ResNetNull(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)