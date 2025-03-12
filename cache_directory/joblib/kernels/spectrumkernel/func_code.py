# first line: 53
@memory.cache
def spectrumkernel(X,Y,nb_spectr):
    XeqY = X.shape==Y.shape and (X==Y).all()
    K = np.zeros((X.shape[0], Y.shape[0]))
    for i in tqdm(range(X.shape[0])):
        x = X[i]
        subsequencesx = [x[k:k+nb_spectr] for k in range(len(x)-nb_spectr+1)]
        dictsubx = Counter(subsequencesx)
        for j in range(i if XeqY else 0,Y.shape[0]):
            y = Y[j]
            subsequencesy = [y[k:k+nb_spectr] for k in range(len(y)-nb_spectr+1)]
            dictsuby = Counter(subsequencesy)
            Kij = sum(dictsubx[sub]*dictsuby[sub] for sub in dictsubx.keys()&dictsuby.keys())
            Kii = sum(dictsubx[sub]*dictsubx[sub] for sub in dictsubx.keys())
            Kjj = sum(dictsuby[sub]*dictsuby[sub] for sub in dictsuby.keys())
            K[i,j] = Kij/math.sqrt(Kii*Kjj) +1
            if XeqY:
                K[j,i] = Kij/math.sqrt(Kii*Kjj) +1
    return K
