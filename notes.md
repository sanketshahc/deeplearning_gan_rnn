
Embed layer is what wraps the the table which wraps the words in feature vectors....row
dim is vocab size, col dim is embed dim, ie 32, of choosing....the weights in the layer
are trained and are the thing trainined. The input here is the batch of sentences as
indices...so (BATCH x SEQ_LEN)
output is tables of embeds....so (BATCH x SEQ_LEN x EMBED_DIM).
self.h0, defailts to 0

GRU is like normal rnn, with extra gates (and weights for those gates). 3x the weights
for each moment. input here is the sequence...shape (BATCH x SEQ_LEN x EMBED_DIM).
ouput is both the last hidden state as well as the whole output vector across seq...so
shapes (BATCH x SEQ_LEN x HIDDEN_DIM) and (BATCH x HIDDEN_DIM)...things like
bi-directionality and extra GRU layers add dimensionality...

Good 'ol Linear layer is what receives the GRU output, which is still the hidden size,
needs to bdownsized to the final output

# q why does adam need a learning rate input...how does it work anyway? still sgd right? why need decay still?
# q why doesn't mlp work here? Seems like it should....maybe would just get unweildy?

Essentially, a gan is a competition between 2 networks, A and B. A is a generator,
B is a discriminator, sometimes called critic. the basic flow is A 'generates' sample -> sends
it to B -> B determines whether it's fake or real. This decision is compared against the truth
and loss calculated. this loss backprops all the way to the generator, which then adjusts it's
'generation' to better fool the critic. Then, the critic takes in a real input and does the
same thing. Only this time, the loss backprop does not go all the way to the generator. Or if
it does, gradients are reset when the generator trains anyway. 

# q unclear here why not just reset. 
# q why new sample each time in gan? not cintnuent
# q also unclear here why not feed them in at same time.

            # q would it matter if the gen code was first? essentially if during the discrim loop, that the
            #  gen inpput has to ba new input from same space, or prior one...
the critic then adjusts it's weights. In a sense they are trying to optimize an objective
function with the loss function. Use sigmoid for the discriminator output, and tanh for the
generator output, which has a range of -1,1, which I'd imagine the inputs must also be
normalized to. GANS training poses many challenges and questions...When is training done? What
does convergence look like? What is a good outcome? How to deal with 'multimodal'
distributions? (q Why not make the z multimodal? or why not have multiple adversarial components
 to a single discriminator?) There are also problems such as modal collapse,
 non-convergence, and vanishing gradients.
Note on Unroll Gan and "replay:
Note on Conditional GAN: