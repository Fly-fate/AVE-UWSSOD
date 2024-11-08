import torch
import torch.nn.functional as F
import numpy as np
from isp.util_filters import lrelu, rgb2lum, tanh_range, lerp
import cv2
import math


class Filter:

  def __init__(self, net, cfg):
    self.cfg = cfg
    # self.height, self.width, self.channels = list(map(int, net.get_shape()[1:]))

    # Specified in child classes
    self.num_filter_parameters = None 
    self.short_name = None
    self.filter_parameters = None

  def get_short_name(self):
    assert self.short_name
    return self.short_name

  def get_num_filter_parameters(self):
    assert self.num_filter_parameters
    return self.num_filter_parameters

  def get_begin_filter_parameter(self):
    return self.begin_filter_parameter

  def extract_parameters(self, features): #CNN-PP提取DIP参数
    return features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())], \
           features[:, self.get_begin_filter_parameter():(self.get_begin_filter_parameter() + self.get_num_filter_parameters())]

  # Should be implemented in child classes
  def filter_param_regressor(self, features):
    assert False

  # Process the whole image, without masking
  # Should be implemented in child classes
  def process(self, img, param, defog, IcA):
    assert False

  def debug_info_batched(self):
    return False

  def no_high_res(self):
    return False

  # Apply the whole filter with masking
  def apply(self,
            img,
            img_features=None,
            defog_A=None,
            IcA=None,
            specified_parameter=None,
            high_res=None):
    assert (img_features is None) ^ (specified_parameter is None)
    if img_features is not None:
      filter_features, mask_parameters = self.extract_parameters(img_features)
      filter_parameters = self.filter_param_regressor(filter_features)
    else:
      assert not self.use_masking()
      filter_parameters = specified_parameter
      mask_parameters = torch.zeros( ##
          shape=(1, self.get_num_mask_parameters()), dtype=np.float32)
    if high_res is not None:
      # working on high res...
      pass
    debug_info = {}
    # We only debug the first image of this batch
    if self.debug_info_batched():
      debug_info['filter_parameters'] = filter_parameters
    else:
      debug_info['filter_parameters'] = filter_parameters[0]
    # self.mask_parameters = mask_parameters
    # self.mask = self.get_mask(img, mask_parameters)
    # debug_info['mask'] = self.mask[0]
    #low_res_output = lerp(img, self.process(img, filter_parameters), self.mask)
    low_res_output = self.process(img, filter_parameters, defog_A, IcA)

    if high_res is not None:
      if self.no_high_res():
        high_res_output = high_res
      else:
        self.high_res_mask = self.get_mask(high_res, mask_parameters)
        # high_res_output = lerp(high_res,
        #                        self.process(high_res, filter_parameters, defog, IcA),
        #                        self.high_res_mask)
    else:
      high_res_output = None
    #return low_res_output, high_res_output, debug_info
    return low_res_output, filter_parameters

  def use_masking(self):
    return self.cfg.masking

  def get_num_mask_parameters(self):
    return 6


  # def visualize_filter(self, debug_info, canvas):
  #   # Visualize only the filter information
  #   assert False

  def visualize_mask(self, debug_info, res):
    return cv2.resize(
        debug_info['mask'] * np.ones((1, 1, 3), dtype=np.float32),
        dsize=res,
        interpolation=cv2.cv2.INTER_NEAREST)

  def draw_high_res_text(self, text, canvas):
    cv2.putText(
        canvas,
        text, (30, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0),
        thickness=5)
    return canvas

class DefogFilter(Filter):#Defog_param is in [Defog_range] #去雾

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'DF'
    self.begin_filter_parameter = cfg.defog_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.defog_range)(features)

  def process(self, img, param, defog_A, IcA):
    print('      img:', img.shape)
    print('      IcA:', IcA.shape)
    print('      defog_A:', defog_A.shape)

    tx = 1 - param[:, None, None, :]*IcA ##可能IcA不对
    # tx = 1 - 0.5*IcA

    tx_1 = tx.repeat(1, 1, 1, 3)
    #tx_1 = tf.tile(tx, [1, 1, 1, 3]) #tile()在同一维度上的复制tx1,1,1,3次
    bijiao = torch.ones_like(tx_1)/100
    new_a1 = defog_A[:, None, None, :]
  
    return (img - new_a1)/torch.maximum(tx_1, bijiao) + new_a1
    #return (img - defog_A[:, None, None, :])/torch.maximum(tx_1, bijiao) + defog_A[:, None, None, :]
    #return (img - defog_A[:, None, None, :])/tf.maximum(tx_1, 0.01) + defog_A[:, None, None, :]

class ImprovedWhiteBalanceFilter(Filter): #色偏

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'W'
    self.channels = 3
    self.begin_filter_parameter = cfg.wb_begin_param
    self.num_filter_parameters = self.channels

  def filter_param_regressor(self, features):
    log_wb_range = 0.5
    mask = np.array(((0, 1, 1)), dtype=np.float32).reshape(1, 3)
    mask = torch.tensor(mask)
    mask = mask.cuda()
    #print(mask.shape)
    assert mask.shape == (1, 3)
    features = features * mask
    color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(features))
    #color_scaling = tf.exp(tanh_range(-log_wb_range, log_wb_range)(features))

    # There will be no division by zero here unless the WB range lower bound is 0
    # normalize by luminance
    # color_scaling *= 1.0 / (
    #     1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
    #     0.06 * color_scaling[:, 2])[:, None]
    color_scaling = color_scaling * ( 1.0 / (
        1e-5 + 0.27 * color_scaling[:, 0] + 0.67 * color_scaling[:, 1] +
        0.06 * color_scaling[:, 2])[:, None] )
    return color_scaling

  def process(self, img, param, defog, IcA):
    return img * param[:, None, None, :]

class GammaFilter(Filter):  #gamma_param is in [-gamma_range, gamma_range]

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'G'
    self.begin_filter_parameter = cfg.gamma_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    log_gamma_range = np.log(self.cfg.gamma_range)
    return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))
    #return tf.exp(tanh_range(-log_gamma_range, log_gamma_range)(features))

  def process(self, img, param, defog_A, IcA):
    param_1 = param.repeat(1, 3)
    #param_1 = tf.tile(param, [1, 3])
    bijiao1 = torch.ones_like(img)/10000
    return torch.pow(torch.maximum(img, bijiao1), param_1[:, None, None, :])
    #return tf.pow(tf.maximum(img, 0.0001), param_1[:, None, None, :])

class ToneFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.curve_steps = cfg.curve_steps
    self.short_name = 'T'
    self.begin_filter_parameter = cfg.tone_begin_param

    self.num_filter_parameters = cfg.curve_steps

  def filter_param_regressor(self, features):
    tone_curve = torch.reshape(
        features, shape=(-1, 1, self.cfg.curve_steps))[:, None, None, :]
    tone_curve = tanh_range(*self.cfg.tone_curve_range)(tone_curve)
    return tone_curve

  def process(self, img, param, defog, IcA):
    tone_curve = param
    tone_curve_sum = torch.sum(tone_curve, axis=4) + 1e-30
    #tone_curve_sum = tf.reduce_sum(tone_curve, axis=4) + 1e-30
    total_image = img * 0
    for i in range(self.cfg.curve_steps):
        total_image += torch.clamp(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
                     * param[:, :, :, :, i]
        # total_image += tf.clip_by_value(img - 1.0 * i / self.cfg.curve_steps, 0, 1.0 / self.cfg.curve_steps) \
        #              * param[:, :, :, :, i]
    total_image *= self.cfg.curve_steps / tone_curve_sum
    img = total_image
    return img

class ContrastFilter(Filter):

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'Ct'
    self.begin_filter_parameter = cfg.contrast_begin_param

    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return torch.tanh(features)
    #return tf.tanh(features)

  def process(self, img, param, defog, IcA):
    bijiao2 = torch.zeros_like(rgb2lum(img))/10000
    bijiao3 = torch.ones_like(torch.maximum(rgb2lum(img), bijiao2))
    luminance = torch.minimum(torch.maximum(rgb2lum(img), bijiao2), bijiao3)
    contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
    #contrast_lum = -tf.cos(math.pi * luminance) * 0.5 + 0.5
    contrast_image = img / (luminance + 1e-6) * contrast_lum
    return lerp(img, contrast_image, param[:, :, None, None])

class UsmFilter(Filter):#Usm_param is in [Defog_range]

  def __init__(self, net, cfg):
    Filter.__init__(self, net, cfg)
    self.short_name = 'UF'
    self.begin_filter_parameter = cfg.usm_begin_param
    self.num_filter_parameters = 1

  def filter_param_regressor(self, features):
    return tanh_range(*self.cfg.usm_range)(features)

  def process(self, img, param, defog_A, IcA):
    def make_gaussian_2d_kernel(sigma, dtype=torch.float32):
      radius = 12
      x = torch.arange(-radius, radius + 1)
      x = x.to(dtype=torch.float32)
      #x = tf.cast(tf.range(-radius, radius + 1), dtype=dtype)
      k = torch.exp(-0.5 * torch.square(x / sigma))
      #k = tf.exp(-0.5 * tf.square(x / sigma))
      k = k / torch.sum(k)
      #k = k / tf.reduce_sum(k)
      return k.unsqueeze(1) * k
      #return tf.expand_dims(k, 1) * k

    kernel_i = make_gaussian_2d_kernel(5)
    #print('kernel_i.shape', kernel_i.shape)
    meijie = kernel_i[None, None, :, :] #不确定有没有问题
    kernel_i = meijie.repeat(1, 1, 1, 1)
    kernel_i = kernel_i.cuda()
    #kernel_i = tf.tile(kernel_i[:, :, tf.newaxis, tf.newaxis], [1, 1, 1, 1])
      
    pad_w = (25 - 1) // 2 #12
    padded = F.pad(img, pad=(0,0,pad_w,pad_w,pad_w,pad_w), mode='reflect')
    # pad_w = (25 - 1) // 2
    # padded = tf.pad(img, [[0, 0], [pad_w, pad_w], [pad_w, pad_w], [0, 0]], mode='REFLECT')

    outputs = []
    weight = torch.nn.Parameter(kernel_i)
    weight.requires_grad = False 
    for channel_idx in range(3):
        data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
        data_c = data_c.permute(0,3,1,2) #为了下一步的卷积，转换成pytorch形式
        #print(data_c.shape)
        data_c = F.conv2d(data_c,weight,stride=1,groups=1) #考虑该卷积需不需要求导，及参数更新
        outputs.append(data_c)
    # outputs = []
    # for channel_idx in range(3):
    #     data_c = padded[:, :, :, channel_idx:(channel_idx + 1)]
    #     data_c = tf.nn.conv2d(data_c, kernel_i, [1, 1, 1, 1], 'VALID')
    #     outputs.append(data_c)
    outputs = tuple(outputs) #list转换成tuple
    output = torch.cat(outputs, axis=1) #合并3个通道
    output = output.permute(0,2,3,1) #转换成tf格式，来进行下一步运算
    #output = tf.concat(outputs, axis=3)

    img_out = (img - output) * param[:, None, None, :] + img

    return img_out