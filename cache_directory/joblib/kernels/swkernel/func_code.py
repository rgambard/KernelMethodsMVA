# first line: 133
@memory.cache
def swkernel(X,Y,e,S):
    """ compute the kernel between x and y """
    XeqY = X.shape==Y.shape and (X==Y).all()
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in tqdm(range(X.shape[0])):
        x = convert(X[i])
        for j in range(i if XeqY else 0,Y.shape[0]):
            y = convert(Y[j])
            Kij = swscore(x,y, e, S)
            Kii = swscore(x,x, e, S)
            Kjj = swscore(y,y, e, S)
            K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
            if XeqY:
                K[j,i] = Kij/math.sqrt(Kii*Kjj) +1
    return K
