using Images
using PyPlot
using Clustering
using MultivariateStats

include("Common.jl")

# typealiases for arrays of images and features
typealias ImageList Array{Array{Float64,2},1}
typealias FeatureList Array{Array{Float64,2},1}

# type to store the input images and their labels
type Dataset
  images::Array{Array{Float64,2},1}
  labels::Array{Float64,1}
  n::Int
end

# length method for type Dataset
import Base.length
function length(x::Dataset)
  @assert length(x.images) == length(x.labels) == x.n "The length of the dataset is inconsistent."
  return x.n
end

# type to store parameters
type Parameters
  fsize::Int
  sigma::Float64
  threshold::Float64
  boundary::Int
end

# concatenate two datasets
function concat(d1::Dataset, d2::Dataset )
  return Dataset([d1.images; d2.images], [d1.labels; d2.labels], d1.n + d2.n)
end

# create a train / test split of a dataset
function traintestsplit(d::Dataset, p::Float64)
  ntrain = Int(floor(d.n * p))
  ntest = d.n - ntrain
  permuted_idx = randperm(d.n)

  train = Dataset(d.images[permuted_idx[1:ntrain]], d.labels[permuted_idx[1:ntrain]], ntrain)
  test = Dataset(d.images[permuted_idx[1+ntrain:end]], d.labels[permuted_idx[1+ntrain:end]], ntest)
  return train, test
end

# Create input data by separating planes and bikes randomly into two equally sized sets.
function loadimages()

  nbikes = 106 # number of planes
  nplanes = 134 # number of bikes
  implanes = ImageList(nplanes)
  for i = 1:nplanes
    implanes[i] = PyPlot.imread(@sprintf("../data-julia/planes/%03i.png",i))
  end
  imbikes = ImageList(nbikes)
  for i = 1:nbikes
    imbikes[i] = PyPlot.imread(@sprintf("../data-julia/bikes/%03i.png",i))
  end
  bikes = Dataset(imbikes, ones(Float64, nbikes), nbikes)
  planes = Dataset(implanes, zeros(Float64, nplanes), nplanes)

  trainbikes, testbikes = traintestsplit(bikes, 0.5)
  trainplanes, testplanes = traintestsplit(planes, 0.5)

  trainingset = concat(trainbikes, trainplanes)
  testingset = concat(testbikes, testplanes)

  @assert length(trainingset) == 120
  @assert length(testingset) == 120
  return trainingset::Dataset, testingset::Dataset
end

# Extract features for all images using im2feat for each individual image
function extractfeatures(images::ImageList,params::Parameters)
  n = length(images)
  features = FeatureList(n)
  for i = 1:n
  features[i] = im2feat(images[i], params.fsize, params.sigma, params.threshold, params.boundary)
  end
  @assert length(features) == length(images)
  return features::FeatureList
end

# Extract features for a single image by applying Harris detection to find interest points
# and SIFT to compute the features at these points.
function im2feat(im::Array{Float64,2},fsize::Int,sigma::Float64,threshold::Float64,boundary::Int)
  mask,_ = Common.harris(im, sigma, fsize, threshold)
  mask = padarray(mask[1+boundary:end-boundary, 1+boundary:end-boundary],
  [boundary, boundary], [boundary, boundary], "value", false)
  y, x = findn(mask)
  F = Common.sift([x y], im, sigma)
  @assert size(F,1) == 128
  return F::Array{Float64,2}
end

# Build a concatenated feature matrix from all given features
function concatenatefeatures(features::FeatureList)
  X = cat(2, features...)
  @assert size(X,1) == 128
  return X::Array{Float64,2}
end

# Build a codebook for a given feature matrix by k-means clustering with K clusters
function computecodebook(X::Array{Float64,2},K::Int)
  clusterresult = kmeans(X,K)
  codebook = clusterresult.centers
  @assert size(codebook) == (size(X,1),K)
  return codebook::Array{Float64,2}
end

# Compute a histogram over the codebook for all given features
function computehistogram(features::FeatureList,codebook::Array{Float64,2},K::Int)
  N = length(features)
  H = zeros(K,N)
  for i = 1:N
    F = features[i]
    D = (sum(F.^2, 1) .+ sum(codebook.^2, 1)')' - (2 * F' * codebook)
    _,minidx = findmin(D, 2)
    _,idx = ind2sub(size(D), minidx[:])
    H[:,i] = [count(x->x==k, idx) for k=1:K] ./ size(D, 1)
  end
  @assert size(H) == (K,length(features))
  return H::Array{Float64,2}
end

# Visualize a feature matrix by projection to the first two principal components.
# Points get colored according to class labels y
function visualizefeatures(X::Array{Float64,2}, y)
  pca = fit(PCA, X)
  Xt = transform(pca, X)
  figure()
  # Plot airplanes
  idx = (y.==0)
  scatter(Xt[1,idx], Xt[2,idx], color="b")
  # Plot bikes
  idx = (y.==1)
  scatter(Xt[1,idx], Xt[2,idx], color="r")
  title("Projection of data onto PC1 and PC2")
  xlabel("PC1")
  ylabel("PC2")
  legend(["airplanes", "bicycles"])
  return nothing
end

# Problem 1: Bag of Words Model: Codebook

function problem1()
  # parameters
  params = Parameters(15,1.4,1e-7,10)
  K = 50

  # load trainging and testing data
  traininginputs,testinginputs = loadimages()

  # extract features from images
  trainingfeatures = extractfeatures(traininginputs.images,params)
  testingfeatures = extractfeatures(testinginputs.images,params)

  # construct feature matrix from the training features
  X = concatenatefeatures(trainingfeatures)

  # write codebook
  codebook = computecodebook(X,K)

  # compute histogram
  traininghistogram = computehistogram(trainingfeatures,codebook,K)
  testinghistogram = computehistogram(testingfeatures,codebook,K)

  # visualize training features
  visualizefeatures(traininghistogram, traininginputs.labels)

  return
end
