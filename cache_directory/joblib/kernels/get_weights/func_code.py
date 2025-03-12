# first line: 12
@memory.cache
def get_weights(x0,y0,k, keys):
    kernel = spectrum_kernel_vec(k,keys)
    vec0 = kernel.to_vectors(x0[y0==0])
    vec1 = kernel.to_vectors(x0[y0==1])
    weights = np.zeros(len(keys))
    for i in tqdm(range(len(keys))):
        weights[i]=np.abs(np.sum(vec0[:,i][:,None]@vec0[:,i][None,:])+
                    np.sum(vec1[:,i][:,None]@vec1[:,i][None,:])-
                    2*np.sum(vec0[:,i][:,None]@vec1[:,i][None,:]))
    return weights
