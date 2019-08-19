from CutMD import * 

conv_net = train(4000, 1e-4, 1000)
print(test(conv_net))
