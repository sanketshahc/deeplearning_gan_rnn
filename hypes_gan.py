xout_size = 28 * 28 # Fixed
epochs = 50  # iterations
batch_size = 64 #
# use larger batch.....256? 1024?
# z, hidden sizes for generator
z_size = 200
hs_g1 = 1024
hs_g2 = 2048
hs_g3 = 2048

# hidden sizes for discriminator
hs_d1= 4096
hs_d2= 4096
hs_d3= 512

# optim params generator
learning_rate_g = .00005
w_g = 0.000 # weight decay l2 try .005
beta1_g = .9 # adam optim
beta2_g = .99 # adam optim
c_g = 1000 # clamp value for gradients


rg = 1 # number repitions
rd = 3 # try more d, then more of both
rolls = 3

# optim params discriminator
learning_rate_d = .00005
w_d = 0.01 # try l2, .005
beta1_d = .5 # try .75
beta2_d = .99
c_d = 1000 # clamp value for gradient norm
c_w = .001 # clamp value for weights (absolute)
# c_w = [0.1, 0.01, 0.001, 0.0001]
# Add the gradients from the all-real and all-fake batches


#### PROB 1


# batch_size = 64
# hidden_size = 64
# embed_size = 32
# learning_rate = .01
# epochs = 30  # iterations
# w = .003
