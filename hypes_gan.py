xout_size = 28 * 28 # Fixed
epochs = 50  # iterations
batch_size = 64 #

# z, hidden sizes for generator
z_size = 256
hs_g1 = 512
hs_g2 = 1024
hs_g3 = 2048

# hidden sizes for discriminator
hs_d1= 2048
hs_d2= 1024
hs_d3= 512

# optim params
learning_rate_g = .00001
learning_rate_d = .0001
w = 0
beta1 = .5
beta2 = .9

c = 20


hypes = {


}