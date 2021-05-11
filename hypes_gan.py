xout_size = 28 * 28 # Fixed
epochs = 75  # iterations
batch_size = 64 #

# z, hidden sizes for generator
z_size = 200
hs_g1 = 1024
hs_g2 = 2048
hs_g3 = 1024

# hidden sizes for discriminator
hs_d1= 1024
hs_d2= 1024
hs_d3= 512

# optim params
learning_rate_g = .0001
learning_rate_d = .0001
w = 0
beta1 = .5
beta2 = .9

c = 10

hypes = {


}