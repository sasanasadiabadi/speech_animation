# apply dimensionality reduction on data

path = "--path to data"
save_path = "-- path to saving folder"

#read data from path
files = readdir(path)

X = Any[]
for k in files
	dt = readdlm(joinpath(path,k), ',')
	push!(X,dt)
end
X = reduce(vcat,X)
X = X'
println("data size: ", size(X))

# mean substraction
X_bar = mean(X,2)
Xs = X .- X_bar
# eigenvalue decomposiiton for the covraiance matrix
C = Xs*Xs'./(size(X,2))
e_val, e_vec = eig(C)
e_val = sort(e_val, rev=true)
# get minimum number of parameters to explain variation
tlist = cumsum(e_val).>0.98*sum(e_val)
t = findfirst(tlist)
println("number of pca parameters: ", t)

evect = zeros(size(e_vec,1),t)
for i=1:t
    evect[:,i] = e_vec[:,end-i+1]
end

# project data into pca space
X_pca = zeros(t,size(X,2))

for i=1:size(X,2)
    X_pca[:,i] = evect'*(X[:,i] - X_bar)
end

# write pca data into csv files
j = 0
for k in files
	filname = joinpath(save_path,k)
    writecsv(filname, X_pca[:,(j*75)+1:(j+1)*75])
    j = j + 1
end
