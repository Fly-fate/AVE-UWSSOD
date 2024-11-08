#from isp.filters import *
from isp.filters_pytorch import *
from isp.config import cfg
import numpy as np

class DIP:
    def __init__(self, input):
        self.filters = [ImprovedWhiteBalanceFilter,GammaFilter,ContrastFilter,UsmFilter]
        #self.filters = [ImprovedWhiteBalanceFilter,GammaFilter,ToneFilter,ContrastFilter,UsmFilter]
        #self.filter_input = np.transpose(input.cpu().numpy(),[0,2,3,1])
        self.filter_input = input.permute(0,2,3,1) #将pytorch格式传入的图像(b,c,h,w)转换成tensorflow格式(b,h,w,c)

        # self.dark = np.zeros([self.filter_input.shape[0], self.filter_input.shape[1], self.filter_input.shape[2]])
        # self.defog_A = np.zeros([self.filter_input.shape[0], self.filter_input.shape[3]])
        # self.IcA = np.zeros((self.filter_input.shape[0], self.filter_input.shape[1], self.filter_input.shape[2]))
        # for i in range(self.filter_input.shape[0]): #循环6次，遍历一个batch中的每一张图像
        #     dark_i = DarkChannel(self.filter_input[i])
        #     defog_A_i = AtmLight(self.filter_input[i], dark_i)
        #     IcA_i = DarkIcA(self.filter_input[i], defog_A_i)
        #     self.dark[i, ...] = dark_i
        #     self.defog_A[i, ...] = defog_A_i
        #     self.IcA[i, ...] = IcA_i
        # self.IcA = np.expand_dims(self.IcA, axis=-1)

        # self.defog_A = torch.tensor(self.defog_A)
        # self.defog_A = self.defog_A.cuda()
        # self.defog_A = self.defog_A.to(torch.float32)

        # self.IcA = torch.tensor(self.IcA)
        # self.IcA = self.IcA.cuda()
        # self.IcA = self.IcA.to(torch.float32)

        # self.filter_input = torch.tensor(self.filter_input)
        # self.filter_input = self.filter_input.cuda() #转为gpu tensor

    def image_process(self,filter_features):
    #def image_process(self,filtered_image_batch,filter_features,defog_A, IcA):
        #-------------------------------------------DIP------------------------------------------------#
        filtered_image_batch = self.filter_input
        filters = [x(self.filter_input, cfg) for x in self.filters]
        filter_parameters = []
        filter_imgs_series =[]
        for j, filter in enumerate(filters): #遍历每个filter #构建filter
            print('    creating filter:', j, 'name:', str(filter.__class__), 'abbr.',
                    filter.get_short_name())
            print('      filter_features:', filter_features.shape)

            filtered_image_batch, filter_parameter = filter.apply(filtered_image_batch, filter_features)
            #filtered_image_batch, filter_parameter = filter.apply(self.filter_input, filter_features, self.defog_A, self.IcA)
            
            #filter_parameters.append(filter_parameter)
            #filter_imgs_series.append(filtered_image_batch)
            print('      output:', filtered_image_batch.shape)

        #self.filter_params = filter_parameters

        return filtered_image_batch
        #-------------------------------------------DIP------------------------------------------------#

#一些函数
def DarkChannel(im):
        b, g, r = cv2.split(im)
        dc = cv2.min(cv2.min(r, g), b)
        return dc

def AtmLight(im, dark):
        [h, w] = im.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz, 1)
        imvec = im.reshape(imsz, 3)

        indices = darkvec.argsort(0)
        indices = indices[(imsz - numpx):imsz]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A  
def DarkIcA(im, A):
        im3 = np.empty(im.shape, im.dtype)
        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]
        return DarkChannel(im3)