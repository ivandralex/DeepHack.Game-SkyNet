require "initenv"

function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    -- net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
    --                     args.filter_size[1], args.filter_size[1],
    --                     args.filter_stride[1], args.filter_stride[1],1))
    -- net:add(args.nl())

    -- -- Add convolutional layers
    -- for i=1,(#args.n_units-1) do
    --     -- second convolutional layer
    --     net:add(convLayer(args.n_units[i], args.n_units[i+1],
    --                         args.filter_size[i+1], args.filter_size[i+1],
    --                         args.filter_stride[i+1], args.filter_stride[i+1]))
    --     net:add(args.nl())
    -- end

    --module = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
    --module = nn.SpatialMaxPooling(kW, kH [, dW, dH, padW, padH])


    net:add(nn.SpatialConvolution(args.hist_len*args.ncols, 32, 8, 8, 4, 4, 1, 1))
    net:add(args.nl())
    net:add(nn.SpatialConvolution(32, 64, 4, 4, 2, 2))
    net:add(args.nl())
    net:add(nn.SpatialConvolution(64, 64, 3, 3))
    net:add(args.nl())


    -- net:add(nn.SpatialConvolution(args.hist_len*args.ncols, 32, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


    -- net:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


    -- net:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


    -- net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- net:add(args.nl())
    -- net:add(nn.SpatialMaxPooling(2, 2, 2, 2))


    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    -- net:add(nn.Linear(nel, args.n_hid[1]))
    -- net:add(args.nl())
    -- local last_layer_size = args.n_hid[1]

    -- for i=1,(#args.n_hid-1) do
    --     -- add Linear layer
    --     last_layer_size = args.n_hid[i+1]
    --     net:add(nn.Linear(args.n_hid[i], last_layer_size))
    --     net:add(args.nl())
    -- end

    -- -- add the last fully connected layer (to actions)
    -- net:add(nn.Linear(last_layer_size, args.n_actions))

    net:add(nn.Linear(nel, 1024))
    net:add(args.nl())
    net:add(nn.Dropout(0.5))    
    net:add(nn.Linear(1024, 512))
    net:add(args.nl())
    net:add(nn.Linear(512, args.n_actions))


    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
