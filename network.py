import torch.nn as nn


"""Define Network"""
class ConvLayer(nn.Module):
    def __init__(self,in_size,out_size,kernel=(3,3),strides=1,padding=1,activation = nn.ReLU()):
        super(ConvLayer,self).__init__()
        model = [nn.Conv2d(in_size,out_size,kernel_size=kernel,stride=strides,padding=padding)]

        model.append(nn.BatchNorm2d(out_size))

        if activation is not None: model.append(activation)

        self.model = nn.Sequential(*model)

    def forward(self,x):
        x = self.model(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        #Encoder
        self.conv1 = ConvLayer(1,64,strides=2)
        self.conv2 = ConvLayer(64,128)
        self.conv3 = ConvLayer(128,128,strides = 2)
        self.conv4 = ConvLayer(128,256)
        self.conv5 = ConvLayer(256,256,strides = 2)
        self.conv6 = ConvLayer(256,512)
        self.conv7 = ConvLayer(512,512)
        self.conv8 = ConvLayer(512,256)
        #Decoder
        self.conv9 = ConvLayer(256,128)
        self.up1 = nn.Upsample(scale_factor = 2, mode = 'bicubic')
        self.conv10 = ConvLayer(128,64)
        self.up2 = nn.Upsample(scale_factor = 2, mode = 'bicubic')
        self.conv11 = ConvLayer(64,32)
        self.conv12 = ConvLayer(32,16)
        self.conv13 = ConvLayer(16,2,activation=nn.Tanh())
        self.up3 = nn.Upsample(scale_factor = 2, mode = 'bicubic')


    def forward(self,x):

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        c8 = self.conv8(c7)

        c9 = self.conv9(c8)
        u1 = self.up1(c9)
        c10 = self.conv10(u1)
        u2 = self.up2(c10)
        c11 = self.conv11(u2)
        c12 = self.conv12(c11)
        c13 = self.conv13(c12)
        u3 = self.up3(c13)
        
        print(x.size())
        print("1")
        print(c1.size())
        print("2")
        print(c2.size())
        print("3")
        print(c3.size())
        print("4")
        print(c4.size())
        print("5")
        print(c5.size())
        print("6")
        print(c6.size())
        print("7")
        print(c7.size())
        print("8")
        print(c8.size())
        print("9")
        print(c9.size())
        print("u1")
        print(u1.size())
        print("10")
        print(c10.size())
        print("u2")
        print(u2.size())
        print("11")
        print(c11.size())
        print("12")
        print(c12.size())
        print("13")
        print(c13.size())
        print("u4")
        print(u3.size())

        return u3
