# speech animation
for p in ("Knet","ArgParse","CSV")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

using Knet
using CSV

atype = gpu()>=0 ? KnetArray{Float32}:Array{Float32}

batchsize = 10
epochs = 10
srand = 123
Kx = 5  # sliding window of size 11 at input sequence
Ky = 2  # sliding window size of 5 at output sequence
num_class = 41
num_pca = 6

path = "--path to data"

#read data from path
files = readdir(path)

Xtrn = Any[]
ytrn = Any[]
for k in files
	dt = readdlm(joinpath(path,k), ',')
    #dt = CSV.read("bbaf2n.csv")
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
# flatten data
N = size(Xtrn,1)
X_train = zeros((2*Kx+1)*num_class,N)
y_train = zeros((2*Ky+1)*num_pca,N)

for i=1:N
	X_train[:,i] = Xtrn[i][:]
	y_train[:,i] = ytrn[i][:]
end

println("train size",size(X_train))
println("label size",size(y_train))

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
function predict(w,x)
    # 3 feed forward dense layers
	x1 = sigm.(w[1]*x .+ w[2])
	x2 = sigm.(w[3]*x1 .+ w[4])
	x3 = sigm.(w[5]*x2 .+ w[6])
    # output regression layer
	x4 = w[7]*x3 .+ w[8]

	return x4
end

# initilize weight matrix
function init_weight()
	w = Any[]
	push!(w,xavier(1024,451))
	push!(w,zeros(1024,1))
	push!(w,xavier(1024,1024))
	push!(w,zeros(1024,1))
	push!(w,xavier(1024,1024))
	push!(w,zeros(1024,1))
	push!(w,xavier(30,1024))
	push!(w,zeros(30,1))
	return map(atype,w)
end

# MSE loss
loss(w,x,y) = mean(abs2,y .- predict(w,x))

lossgradient = grad(loss)

# weight update
function train!(w, data)
    for (x,y) in data
        g = lossgradient(w,x,y)
		g = map(atype,g)
		#opts = map(x->Sgd(), w)
		opts = map(x->Adam(), w)
        update!(w, g)
    end
	return w
end

# main function
dtrn = mini_batch(X_train,y_train,batchsize)

w = init_weight()

for i=1:epochs
    w = train!(w,dtrn)
    trnloss = 0
    count = 0 # counts number of batches
    for (x,y) in dtrn
        trnloss += loss(w,x,y)
        count += 1
    end
    trnloss = trnloss/count
    println("(epoch $i/$epochs : train_loss: $trnloss)")
end


