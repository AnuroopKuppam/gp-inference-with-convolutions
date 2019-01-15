def TorchConv(N, kernel, noise, y, missing_indices):
    y_flat = torch.from_numpy(np.ndarray.flatten(y)).float()
    noise = torch.from_numpy(noise).float()
    mask = ~missing_indices
    input_vector = np.array(kernel)

    # tensorflow
    input_vector = np.expand_dims(input_vector, axis=0)  # for the number of examples which is one here
    input_vector = np.expand_dims(input_vector, axis=1)  # for the number of channels which is one here
    input_vector = torch.from_numpy(np.array(input_vector)).float()

    # tensorflow
    conv_layer = torch.nn.Conv2d(1, 1, (N, N), bias=False, dilation=1, padding=0)
    conv_layer.weight = torch.nn.Parameter(
        torch.from_numpy(np.reshape(np.random.normal(0, 1, (N, N)), (1, 1, N, N))).float())

    loss_plot = []

    y_prime = conv_layer(input_vector)

    # loss
    v_prime = conv_layer.weight.data
    v = v_prime.view(N * N)
    v = torch.flip(v, dims=(0,))
    y_prime = y_prime.view(N * N)

    temp = y_flat - y_prime
    first = 0.5 * (1 / NOISE) * torch.dot(temp, temp)
    second = 0.5 * torch.dot(v, y_prime)
    loss = (first + second)
    # print(y_flat.size(), y_prime.size(), v.size())

    loss_plot.append(loss)
    if count % 100 == 0:
        print_vec(v.detach().numpy(), y_prime.detach().numpy(), loss.detach().numpy())

    adam = optim.Adam([conv_layer], lr=0.1)
    for i in range(800):
        adam.zero_grad()
        loss.backward()
        adam.step()

        img = np.reshape(y_prime.detach().numpy(), (N, N))
        se = np.sum((y[mask]--img[mask])**2)
        print('loss:', se)
        plot(img)
        print(np.allclose(y, img, atol=1e-15))
