#####################################################################################
### When using the medical (RSNA, ISIC, COVID) datasets, the following parameters ###
#####################################################################################

hiddenChannels = 256
n_res_blocks = 3
weight_decay = 5e-5
zero_use_logscale = True
normalize = True
actNormScale = 1.0
zero_logscale_factor = 3.0
imShape = [3, 128, 128]
K = 32
L = 3
perm = "invconv"
coupling = "affine"
y_classes = 1
y_learn_top = False
y_condition = False
y_weight = 0.01
conditional = False
LU = True

####################################################################
### When using CIFAR, SVHN datasets, the following parameters ###
####################################################################

# hiddenChannels = 64
# n_res_blocks = 3
# weight_decay = 5e-5
# zero_use_logscale = True
# normalize = True
# actNormScale = 1.0
# zero_logscale_factor = 3.0
# y_condition = False  # not implemented
# imShape = [3, 32, 32]
# # inChannels = 3 # redundant
# K = 32
# L = 3
# perm = "invconv"
# coupling = "affine"
# y_classes = 1
# y_learn_top = False
# y_condition = False
# y_weight = 0.01
# conditional = False
# LU = True