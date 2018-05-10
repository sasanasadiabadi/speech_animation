# speech animation
for p in ("Knet","ArgParse","JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using JLD

atype = gpu()>=0 ? KnetArray{Float32}:Array{Float32}

batchsize = 128
epochs = 10
srand = 123
Kx = 5  # sliding window of size (2Kx+1) at input sequence
Ky = 2  # sliding window size of (2Ky+1) at output sequence
num_phn = 41
num_pca = 16
mode = 0 # 0 for MLP/ 1 for CNN

path = "--path to raw data"

#read data from path
files = readdir(path)

Xtrn = Any[]
ytrn = Any[]

function make_data(md)
    Xtrn = Any[]
    ytrn = Any[]
    for k in files
    	dt = readdlm(joinpath(path,k), ',')
    	# seperate input/output sequences
    	X = dt[1:41,:]
    	y = dt[42:end,:]

    	# pad data to input/output sequences
    	X = hcat(repmat(X[:,1],1,Kx),X,repmat(X[:,end],1,Kx))
    	y = hcat(repmat(y[:,1],1,Ky),y,repmat(y[:,end],1,Ky))

    	# create dataset using sliding window
    	for i=Kx+1:size(X,2)-Kx
    		tmp = X[:,i-Kx:i+Kx]

    		append!(Xtrn,[tmp])
    	end

    	for j=Ky+1:size(y,2)-Ky
    		tmp = y[:,j-Ky:j+Ky]
    		append!(ytrn,[tmp])
    	end
    end
    #Xt = reduce(hcat,Xtrn)
    # flatten data
    N = size(Xtrn,1)
    if (md==0)
        X_train = zeros((2*Kx+1)*num_phn,N)
        y_train = zeros((2*Ky+1)*num_pca,N)

        for i=1:N
        	X_train[:,i] = Xtrn[i][:]
        	y_train[:,i] = ytrn[i][:]
        end
    elseif (md==1)
        X_train = zeros(num_phn,1,(2*Kx+1),N)
        y_train = zeros((2*Ky+1)*num_pca,N)

        for i=1:N
        	X_train[:,:,:,i] = reshape(Xtrn[i],num_phn,1,2Kx+1)
        	y_train[:,i] = ytrn[i][:]
        end
    end
    return X_train, y_train
end

X_train, y_train = make_data(mode)


# minibatch data
function mini_batch(X, Y, bs)
    data = Any[]
    for i=1:bs:size(X, 2)
        bl = min(i+bs-1, size(X, 2))
        push!(data, (X[:, i:bl], Y[:, i:bl]))
    end
    return data
end

# predict ouput values
function predict(w,x,md)
    if (md==0)
        # 3 fully-connected layers
    	x1 = tanh.(w[1]*mat(x) .+ w[2])
    	x2 = tanh.(w[3]*x1 .+ w[4])
    	x3 = tanh.(w[5]*x2 .+ w[6])
        # output regression layer
    	x5 = w[7]*x3 .+ w[8]

    elseif (md==1)
        x1 = pool(tanh.(conv4(w[1],x) .+ w[2]))
        x2 = pool(tanh.(conv4(w[3],x1) .+ w[4]))

        x3 = tanh.(w[5]*mat(x2) .+ w[6])
        x4 = tanh.(w[7]*x3 .+ w[8])
        x5 = w[9]*x4 .+ w[10]
    end

    return x5
end

# initilize weight matrix
function init_weight(xtype,md)
    w = Any[]
    if (md==0)
    	push!(w,xavier(3000,451))
    	push!(w,zeros(3000,1))
    	push!(w,xavier(3000,3000))
    	push!(w,zeros(3000,1))
    	push!(w,xavier(3000,3000))
    	push!(w,zeros(3000,1))
    	push!(w,xavier(80,3000))
    	push!(w,zeros(80,1))

    elseif (md==1)
        push!(w, xavier(7,1,11,256))
        push!(w,zeros(1,1,256,1))
        push!(w, xavier(5,1,256,512))
        push!(w,zeros(1,1,512,1))
        push!(w,xavier(3000,3072))
        push!(w,zeros(3000,1))
        push!(w,xavier(3000,3000))
        push!(w,zeros(3000,1))
        push!(w,xavier(80,3000))
        push!(w,zeros(80,1))
    end
    return map(xtype,w)
end


# MSE loss
loss(w,x,y) = mean(abs2,y .- convert(Array{Float32}, predict(w,x,mode)))

lossgradient = grad(loss)

# weight update
function train!(w, data,xtype)
    for (x,y) in data
	g = lossgradient(w,x,y)
	g = map(xtype,g)
	#opts = map(x->Sgd(), w)
	opts = map(x->Adam(), w)
        update!(w, g, opts)
    end
    return w
end

########## main function ##########

#dtrn = mini_batch(X_train,y_train,batchsize)
dtrn = minibatch(X_train, y_train, batchsize, xtype=atype)

w = init_weight(atype,mode)

t0 = now()
for i=1:epochs
    w = train!(w,dtrn,atype)
    trnloss = 0
    count = 0 # counts number of batches
    for (x,y) in dtrn
        trnloss += loss(w,x,y)
        count += 1
    end
    trnloss = trnloss/count

    t1 = now()
    train_time = Int((t1-t0).value)*0.001

    println("elapsed time: ", train_time)
    println("(epoch $i/$epochs : train_loss: $trnloss)")
    t0 = t1

    bst_loss = Inf
    if trnloss<bst_loss
        bst_loss = trnloss
        save("Weights.jld","w",map(wi->Array(wi),w))
        println("model saved!")
    end
end



