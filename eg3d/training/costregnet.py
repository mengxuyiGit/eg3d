import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN

from ipdb import set_trace as st

#----------------------------------------
class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.act = nn.LeakyReLU()
        # self.bn = nn.ReLU()
        # self.conv.apply(conv3d_weights_init)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
    return feat_mean, feat_std

class ConvInReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act='IN'):
        super(ConvInReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.inst_norm = nn.InstanceNorm3d(out_channels) # without learnable parameters  
        self.act = nn.LeakyReLU()
        ## encoder: open learnable parameter
        ## decoder: 

    def forward(self, x):
        return self.act(self.inst_norm(self.conv(x)))

class CostRegNet_Deeper(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, in_channels, out_dim=8, norm_act=InPlaceABN):
        super(CostRegNet_Deeper, self).__init__()
        
        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        conv0 = self.conv0(x)

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        # if self.conv3.bn.weight.grad != None:
        # st()

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))
        # print("CostRegNetDeeper bottleneck:", conv62.shape): # 256^3 -> 8^3; 128^3 -> 4^3
        x = conv61 + self.conv27(conv62)
        x = conv6 + self.conv17(x)

        x = conv4 + self.conv7(x)
        # del conv4
        x = conv2 + self.conv9(x)
        # x = conv2 + self.conv9(conv4)
        del conv2, conv4
        x = conv0 + self.conv11(x)
        del conv0
        # x = self.conv12(x)
        return x


class PcWsUnet(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, in_channels, in_resolution, block_resolutions, out_dim=8, norm_act=InPlaceABN):
        super(PcWsUnet, self).__init__()
        self.block_resolutions = block_resolutions
        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

        ## construct FC layers
        max_res, min_res = max(self.block_resolutions), min(self.block_resolutions)
        ## in_res --> outdim
        ## inres//2 --> 16
        self.max_res = min(max_res, in_resolution, 32)
        res = self.max_res
        channels = {128:8, 64:16, 32:32, 16:64, 8:64, 4:64}
        while (res>= min_res):
            ch = channels.get(res)
            layer = nn.Linear((res**3)*ch, ch*3)
            setattr(self, f'fc{res}', layer)
            # print(res, ch)
            res = res//2


    def forward(self, x):
        res_feature = {}

        conv0 = self.conv0(x)
        # res_feature[conv0.shape[-1]]=conv0

        conv2 = self.conv2(self.conv1(conv0))
        # res_feature[conv2.shape[-1]]=conv2

        conv4 = self.conv4(self.conv3(conv2))
        # res_feature[conv4.shape[-1]]=conv4
        # if self.conv3.bn.weight.grad != None:

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))
        # res_feature[conv6.shape[-1]]=conv6

        conv61 = self.conv61(self.conv51(conv6))
        # res_feature[conv61.shape[-1]]=conv61

        conv62 = self.conv62(self.conv52(conv61))
        # res_feature[conv62.shape[-1]]=conv62

        # print("CostRegNetDeeper bottleneck:", conv62.shape): # 256^3 -> 8^3; 128^3 -> 4^3
        x = conv61 + self.conv27(conv62)
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1))
        except:
            pass

        x = conv6 + self.conv17(x)
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass
        # res_feature[x.shape[-1]]=x

        x = conv4 + self.conv7(x)
        # res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass

        # del conv4
        x = conv2 + self.conv9(x)
        # res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass

        # x = conv2 + self.conv9(conv4)
        del conv2, conv4
        x = conv0 + self.conv11(x)
        # if x.shape[-1] <= self.max_res:
        #     res_feature[x.shape[-1]]=x
        try:
            res = x.shape[-1]
            layer = getattr(self, f'fc{res}')
            res_feature[x.shape[-1]]=layer(x.flatten(1)) 
        except:
            pass
        
        del conv0
        # x = self.conv12(x)

        ## use FC layer to project 3D volumes to 1d ws feature
        # res_ws={}
        # for res, vol in res_feature.items():
        #     layer = getattr(self, f'fc{res}')
        #     ws = layer(vol.flatten(1))
        #     res_feature.update({res:ws})

        return res_feature


class Synthesis3DUnet_v0(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, 
            in_channels, 
            out_dim=8, 
            use_noise=False,
            noise_strength = 0.5,
            ws_channel=512,
            affine_act='relu', #### ???? FIXME: is this a good activation 
            norm_act=InPlaceABN):

        super(Synthesis3DUnet, self).__init__()

        self.use_noise = use_noise
        # noise_strength = 0.5
        self.noise_strength = noise_strength

        self.conv0 = ConvBnReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine27 = nn.Sequential(
                        nn.Linear(ws_channel, 64),
                        nn.ReLU()
                    )
        
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine17 = nn.Sequential(
                        nn.Linear(ws_channel, 64),
                        nn.ReLU()
                    )

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))
        self.affine7 = nn.Sequential(
                        nn.Linear(ws_channel, 32),
                        nn.ReLU()
                    )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))
        self.affine9 = nn.Sequential(
                        nn.Linear(ws_channel, 16),
                        nn.ReLU()
                    )

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))
        self.affine11 = nn.Sequential(
                        nn.Linear(ws_channel, out_dim),
                        nn.ReLU()
                    )
        

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x, ws):
        st() # not implement BN version
        conv0 = self.conv0(x)
        

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        # if self.conv3.bn.weight.grad != None:
        # st()

        # x = self.conv6(self.conv5(conv4))
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))
        # print("CostRegNetDeeper bottleneck:", conv62.shape) # 256^3 -> 8^3; 128^3 -> 4^3

        ### below is upconv process: add noises and latent
        w_idx=0


        x = conv61 + self.conv27(conv62)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            
            x = x.add_(noise.to(x.dtype))
        style = self.affine27(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        # st()
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style
        # st() # x.shape


        x = conv6 + self.conv17(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine17(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style

        x = conv4 + self.conv7(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine7(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style
        # del conv4

        x = conv2 + self.conv9(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine9(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style

        x = conv0 + self.conv11(x)
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))
        style = self.affine11(ws.narrow(1, w_idx, 1)).permute(0,2,1)
        w_idx += 1
        assert x.shape[:2] == style.shape[:2]
        B, C = x.shape[:2]
        style = style.reshape(B,C,1,1,1) # extend to 3D 
        x = x*style

        # print(f"Totally used up to {w_idx} ws in synthesis3DUnet") ## currently used only 5
        
        return x

## completely collapsed: 
# maybe because the IN does not have learnable parameters, the model capacity is severely reduced
class Synthesis3DUnet_AdaIN(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, 
            in_channels, 
            out_dim=8, 
            use_noise=False,
            noise_strength = 0.5,
            ws_channel=512,
            affine_act='relu', #### ???? FIXME: is this a good activation 
            norm_act=nn.InstanceNorm3d):

        super(Synthesis3DUnet, self).__init__()

        self.use_noise = use_noise
        # noise_strength = 0.5
        self.noise_strength = noise_strength

        self.conv0 = ConvInReLU3D(in_channels, out_dim, norm_act=norm_act)

        self.conv1 = ConvInReLU3D(out_dim, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvInReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvInReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvInReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvInReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvInReLU3D(64, 64, norm_act=norm_act)

        self.conv51 = ConvInReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv61 = ConvInReLU3D(64, 64, norm_act=norm_act)

        self.conv52 = ConvInReLU3D(64, 64, stride=2, norm_act=norm_act)
        self.conv62 = ConvInReLU3D(64, 64, norm_act=norm_act)

        self.conv27 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine27 = nn.Sequential(
                        nn.Linear(ws_channel, 64*2),
                        nn.ReLU()
                    )
        
        
        self.conv17 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(64))
        self.affine17 = nn.Sequential(
                        nn.Linear(ws_channel, 64*2),
                        nn.ReLU()
                    )

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))
        self.affine7 = nn.Sequential(
                        nn.Linear(ws_channel, 32*2),
                        nn.ReLU()
                    )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))
        self.affine9 = nn.Sequential(
                        nn.Linear(ws_channel, 16*2),
                        nn.ReLU()
                    )

        # self.conv11 = nn.Sequential(
        #     nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(8))
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, out_dim, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(out_dim))
        self.affine11 = nn.Sequential(
                        nn.Linear(ws_channel, out_dim*2),
                        nn.ReLU()
                    )
        

        # self.conv12 = nn.Conv3d(8, 8, 3, stride=1, padding=1, bias=True)

    def forward(self, x, ws):
        conv0 = self.conv0(x)
        

        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
     
        conv6 = self.conv6(self.conv5(conv4))

        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))

        # print("CostRegNetDeeper bottleneck:", conv62.shape) # 256^3 -> 8^3; 128^3 -> 4^3
        # print("IN 3D conv")

        ### below is upconv process: add noises and latent
        ## norm -> mod -> noise
        w_idx=0

        # 1. x is the sum of instance-normed feature volumes
        x = conv61 + self.conv27(conv62) 
        
        # 2. apply style
        style = self.affine27(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
        # 3. add noise
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))


        x = conv6 + self.conv17(x)
        # 2. apply style
        style = self.affine17(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
        # 3. add noise
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))

        x = conv4 + self.conv7(x)
        # 2. apply style
        style = self.affine7(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
        # 3. add noise
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))      

        x = conv2 + self.conv9(x)
        # 2. apply style
        style = self.affine9(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
        # 3. add noise
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
            x = x.add_(noise.to(x.dtype))

        x = conv0 + self.conv11(x)
        # 2. apply style
        style = self.affine11(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
       
        # last layer, no noise anymore
        
        return x

# @_jiayuan version
class Synthesis3DUnet(nn.Module): # 256^3 -> 8^3; 128^3 -> 4^3
    def __init__(self, 
            in_channels, 
            out_dim=8, 
            use_noise=False,
            noise_strength = 0.5,
            ws_channel=512,
            affine_act='relu', #### ???? FIXME: is this a good activation 
            norm_act=None):

        super(Synthesis3DUnet, self).__init__()

        self.use_noise = use_noise
        # noise_strength = 0.5
        self.noise_strength = noise_strength

        self.conv0 = ConvBnReLU3D(in_channels, out_dim//2, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(out_dim//2, 8, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(8, 8, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv51 = ConvBnReLU3D(32, 32, stride=2, norm_act=norm_act)
        self.conv61 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv52 = ConvBnReLU3D(32, 32, stride=2, norm_act=norm_act)
        self.conv62 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        ## take apart previous conv3dblock
        ## discarded: 
        # self.conv27 = nn.Sequential(
        #     nn.ConvTranspose3d(32, 32, 3, padding=1, output_padding=1,
        #                        stride=2, bias=False),
        #     norm_act(32))
        ## new:
        self.conv27 = nn.ConvTranspose3d(32, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=True) # StyleGAN has bias for conv
        self.act27 = nn.LeakyReLU(64)
        self.in27 = nn.InstanceNorm3d(64) # without learnable parameters  
        self.affine27 = nn.Linear(ws_channel, 64*2) # styleGAN uses "linear activation", and in implementation is just a pure linear layer (class FullyConnectedLayer)
        
        self.conv17 = nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=True)
        self.act17 = nn.LeakyReLU(64)
        self.in17 = nn.InstanceNorm3d(64)
        self.affine17 = nn.Linear(ws_channel, 64*2)

        self.conv7 = nn.ConvTranspose3d(64, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=True)
        self.act7 = nn.LeakyReLU(32) 
        self.in7 = nn.InstanceNorm3d(32)
        self.affine7 = nn.Linear(ws_channel, 32*2)

        self.conv9 = nn.ConvTranspose3d(32, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=True)
        self.act9 = nn.LeakyReLU(16) 
        self.in9 = nn.InstanceNorm3d(16)
        self.affine9 = nn.Linear(ws_channel, 16*2)

        self.conv11 = nn.ConvTranspose3d(16, out_dim//2, 3, padding=1, output_padding=1,
                               stride=2, bias=True)
        self.act11 = nn.LeakyReLU(out_dim) 
        self.in11 = nn.InstanceNorm3d(16)
        self.affine11 = nn.Linear(ws_channel, out_dim*2)

    def synthesis_block(self, x, residual, pure_conv, conv_act, inst_norm, cur_w, style_affine):
        #### begin one block
        # 1) conv
        x = pure_conv(x)
        # 2) cat
        x = torch.cat([residual,x],1)
        # 3) add noise
        if self.use_noise:
            noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
            noise = noise*self.noise_strength
        x = x.add_(noise.to(x.dtype))
        # 4) act
        x = conv_act(x)
        # 5) IN: non-learnable
        x = inst_norm(x)
        # 6) + style
        style = style_affine(cur_w).permute(0,2,1) # style.shape: B,C,2
        # w_idx += 1
        B, C = x.shape[:2]
        style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        x = x * (style[:,:,0] + 1) + style[:,:,1]
        #### finish one block
        return x

    def forward(self, x, ws):

        ## encoder still BN
        ## decoder
        # cat = cat([conv61,conv62])
        # conv(cat)
        # + noise
        # leaky_relu(should align with styleGAN act)
        # IN (non-learnable)
        # + style(can still use BN, even the output layer still has BN, it also have some act, and the affine may add shift to it) 
        # mean/std + conv61(if not cat with conv62 )

        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        conv6 = self.conv6(self.conv5(conv4))
        conv61 = self.conv61(self.conv51(conv6))
        conv62 = self.conv62(self.conv52(conv61))

        # print("CostRegNetDeeper bottleneck:", conv62.shape) # 256^3 -> 8^3; 128^3 -> 4^3
        # print("IN 3D conv")

        ### below is upconv process: add noises and latent
        ## norm -> mod -> noise
        w_idx=0

        # #### begin one block
        # # 1) conv
        # x = self.conv27(conv62)
        # # 2) cat
        # x = torch.cat([conv61,x],1)
        # # 3) add noise
        # if self.use_noise:
        #     noise = torch.rand(x.shape, device=x.device, dtype=torch.float32)
        #     noise = noise*self.noise_strength
        #     x = x.add_(noise.to(x.dtype))
        # # 4) act
        # x = self.act27(x)
        # # 5) IN: non-learnable
        # x = self.in27(x)
        # # 6) + style
        # style = self.affine27(ws.narrow(1, w_idx, 1)).permute(0,2,1) # style.shape: B,C,2
        # w_idx += 1
        # B, C = x.shape[:2]
        # style = style.reshape(B,C,2,1,1,1) # extend to 3D 
        # x = x * (style[:,:,0] + 1) + style[:,:,1]
        # #### finish one block

        
        x = self.synthesis_block(conv62, conv61, self.conv27, self.act27, self.in27, ws.narrow(1, w_idx, 1), self.affine27)
        w_idx += 1
        
        x = self.synthesis_block(x, conv6, self.conv17, self.act17, self.in17, ws.narrow(1, w_idx, 1), self.affine17)
        w_idx += 1

        x = self.synthesis_block(x, conv4, self.conv7, self.act7, self.in7, ws.narrow(1, w_idx, 1), self.affine7)
        w_idx += 1

        x = self.synthesis_block(x, conv2, self.conv9, self.act9, self.in9, ws.narrow(1, w_idx, 1), self.affine9)
        w_idx += 1
        
        x = self.synthesis_block(x, conv0, self.conv11, self.act11, self.in11, ws.narrow(1, w_idx, 1), self.affine11)
        w_idx += 1

       
        return x