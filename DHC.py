def self_conj(a):
    return np.real(np.multiply(a,np.conj(a)))

def complex_rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result_real = cv2.warpAffine(np.real(image), rot_mat, image.shape[1::-1], flags=cv2.INTER_LANCZOS4)
    result_imag = cv2.warpAffine(np.imag(image), rot_mat, image.shape[1::-1], flags=cv2.INTER_LANCZOS4)
    return result_real+result_imag*1j

def phase_stretch(im_rd_0_1,stretch,p0=0):
    im_stretch = np.abs(im_rd_0_1)*np.exp(stretch*np.angle(im_rd_0_1)*1j+np.pi*p0*1j)
    return np.fft.fft2(im_stretch)

def phase_stretch_f(U_1_c_f,stretch,p0=0):
    im_rs = np.fft.ifft2(U_1_c_f)
    im_stretch = np.abs(im_rs)*np.exp(stretch*np.angle(im_rs)*1j+np.pi*p0*1j)
    return np.fft.fft2(im_stretch)

def dual_conj(a,b):
    return np.abs(np.multiply(a,np.conj(b)))

def DHC(image,filter_set,phase_on = 1, rotate_on = 1, coeff_12_on = 1, norm_on = 1):
    Nx, Ny = image.shape

    out_coeff = []

    ##Coeff at (1,0) level
    #takes care of image normalization
    S0 = np.zeros((2,1),dtype=np.float64)
    if norm_on == 1:
        S0[0] = np.mean(image)
        norm_im = image-S0[0]
        S0[1] = np.sum(np.square(norm_im))/(Nx*Ny)
        norm_im = norm_im/np.sqrt(Nx*Ny*S0[1])
    else:
        norm_im = image

    out_coeff.append(S0)

    ##Coeff at (1,1) level
    #store the fft of images and im*filters for later
    im_fd_0 = np.fft.fft2(norm_im)

    S1 = np.zeros((8,8),dtype=np.float64)
    im_fd_0_1 = np.zeros((8,8,256,256),dtype=np.complex128)
    for j in range(J):
        for l in range(L):
            im_fd_0_1[j,l] = np.multiply(im_fd_0,filter_set[j,l]) #wavelet already in fft domain not shifted
            S1[j,l]=np.sum(self_conj(im_fd_0_1[j,l])) #normalization choice arb to make order unity

    out_coeff.append(S1)

    ##Coeff at (2,0) level
    S20 = np.zeros((J,J,L,L),dtype=np.float64)

    im_rd_0_1 = np.zeros((8,8,256,256),dtype=np.complex128)
    for j1 in range(J):
        for l1 in range(L):
            im_rd_0_1[j1,l1] = np.fft.ifft2(im_fd_0_1[j1,l1])

    for j1 in range(J):
        for l1 in range(L):
            for l2 in range(L):
                for j2 in range(J):
                    S20[j1,j2,l1,l2] = np.sum(
                        dual_conj(
                            im_rd_0_1[j1,l1],
                            im_rd_0_1[j2,l2]
                        )
                    )

    out_coeff.append(S20)

    ##Coeff at (1,2) level
    if coeff_12_on == 1:
        if np.logical_and(rotate_on == 0,phase_on == 0):
            S12 = np.zeros((J,J,L,L),dtype=np.float64)
            for j1 in range(J):
                for l1 in range(L):
                    not_rot_im = np.fft.fftshift(im_fd_0_1[j1,l1,:,:])
                    for l2 in range(L):
                        for j2 in range(J):
                            S12[j1,j2,l1,l2] = np.sum(
                                dual_conj(
                                    not_rot_im,
                                    np.fft.fftshift(im_fd_0_1[j2,l2,:,:])
                                )
                            )

            out_coeff.append(S12)

        if np.logical_and(rotate_on == 1,phase_on == 0):
            S12 = np.zeros((J,J,L,L),dtype=np.float64)
            for j1 in range(J):
                for l1 in range(L):
                    for l2 in range(L):
                        rot_im = complex_rotate_image(np.fft.fftshift(im_fd_0_1[j1,l1,:,:]),(l1-l2)*180/L)
                        for j2 in range(J):
                            S12[j1,j2,l1,l2] = np.sum(
                                dual_conj(
                                    rot_im,
                                    np.fft.fftshift(im_fd_0_1[j2,l2,:,:])
                                )
                            )

            out_coeff.append(S12)

        if np.logical_and(rotate_on == 0,phase_on == 1):
            S12 = np.zeros((J,J,L,L),dtype=np.float64)
            for j1 in range(J):
                for j2 in range(J):
                    for l1 in range(L):
                        stretch_im = np.fft.fftshift(phase_stretch(im_rd_0_1[j1,l1,:,:],2**(j1-j2)))
                        for l2 in range(L):
                            S12[j1,j2,l1,l2] = np.sum(
                                dual_conj(
                                    stretch_im,
                                    np.fft.fftshift(im_fd_0_1[j2,l2,:,:])
                                )
                            )

            out_coeff.append(S12)

        if np.logical_and(rotate_on == 1,phase_on == 1):
            S12 = np.zeros((J,J,L,L),dtype=np.float64)
            for j1 in range(J):
                for l1 in range(L):
                    for l2 in range(L):
                        rot_im = complex_rotate_image(np.fft.fftshift(im_fd_0_1[j1,l1,:,:]),(l1-l2)*180/L)
                        for j2 in range(J):
                            stretch_im = np.fft.fftshift(phase_stretch_f(np.fft.fftshift(rot_im),2**(j1-j2)))
                            S12[j1,j2,l1,l2] = np.sum(
                                dual_conj(
                                    stretch_im,
                                    np.fft.fftshift(im_fd_0_1[j2,l2,:,:])
                                )
                            )

            out_coeff.append(S12)

    return out_coeff
