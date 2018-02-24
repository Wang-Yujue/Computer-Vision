# 5-layer deep neural networks
# when extend the neural networs to 5 layers which is coded in problem2_deep.jl,
# basicly adding more layers and adding more neurons play same role in improving nural networs prediction,
# in this problem with 50 inputs, under this condition
# I find 5-layer deep neural networks produce more stable test error result than 3-layer
# adding more layers with relative small numbers of neurons in each hidden layer is more efficient than adding more neurons in
# the hidden layer in the previous 3-layer neural networks, for exemple the layer structure [50,25,1] and [50,3,3,3,1] result similar error
# however layer structure [50,3,3,3,1] is faster than [50,25,1]
# moreover, incearse neurons in hidden layers until computing [50,9,9,9,1] same training error: 0.0% testing error: 1.67%
# however computing [50,10,10,10,1] gives a 0.0% training error but worse test error rate, probably overfitting,
# furthermore it is also computational expensive for a PC,
# same for the task1 problem 5-layer, there are not many improvements on classsification
# and I think with 3 layer networks, it is enough to fullfil this classification problem.
# overall considering the tradeoff between the computational cost and the prediction performance
# I would say that for this problem testing error 1.67% could be the best for now though with different layers and different neurons
# it is sufficient of using 3 layers with 10 neurons in the hidden layer [50,10,1] or 5 layers with 3 neurons in hidden layers [50,3,3,3,1]
# but I would definitely prefer 5 layers with 3 neurons in hidden layers [50,3,3,3,1],
# because it gives decent error rate and more stable prediction and is more computational efficient

using Images
using PyPlot
using JLD
using Optim

include("Common.jl")

# Load features and labels from file
function loaddata(path::ASCIIString)
  features = load(path, "features")
  labels = load(path, "labels")
  @assert length(labels) == size(features,1)
  return features::Array{Float64,2}, labels::Array{Float64,1}
end

# Show a 2-dimensional plot for the given features with different colors according
# to the labels.
function showbefore(features::Array{Float64,2},labels::Array{Float64,1})
  idx0 = find(labels.==0)
  idx1 = find(labels.==1)
  figure()
  scatter(features[idx0,1], features[idx0,2], s=10pi, c="blue")
  scatter(features[idx1,1], features[idx1,2], s=10pi, c="red")
  legend(["Class 0", " Class 1"])
  return nothing::Void
end

# Show a 2-dimensional plot for the given features along with the decision boundary
function showafter(features::Array{Float64,2},labels::Array{Float64,1},Ws::Vector{Any}, bs::Vector{Any}, netdefinition::Array{Int,1})
  X = rand(-2.5:1e-4:3,5000,2) # take a grid of random genarated points and feed it in the neural networks
  _,c = predict(X, Ws, bs, netdefinition)
  idx0 = find(labels.==0)
  idx1 = find(labels.==1)
  figure()
  scatter(features[idx0,1], features[idx0,2], s=10pi, c="blue")
  scatter(features[idx1,1], features[idx1,2], s=10pi, c="red")
  idx0 = find(c.==0)
  idx1 = find(c.==1)
  scatter(X[idx0,1], X[idx0,2], s=2pi, c="y", marker="o") # scatter two classes
  scatter(X[idx1,1], X[idx1,2], s=2pi, c="m", marker="x") # boundary of two labeled area is the neural networks decision boundary
  legend(["Class 0", "Class 1", "Class 0", " Class 1"])
  axis([-2.5, 3, -2.5, 3])
  return nothing::Void
end

# Implement the sigmoid function
function sigmoid(z)
  s = 1.0 ./ (1.0 + exp(-z))
  return s
end

# Implement the derivative of the sigmoid function
function dsigmoid_dz(z)
  dz = sigmoid(z) .* (1 - sigmoid(z))
  return dz
end

# To avoid redundant codes
# return the hypothesis which is the output
function hypothesis(X::Array{Float64,2}, theta::Vector{Float64}, netdefinition::Array{Int,1})
  theta1, theta2, theta3, theta4 = parameters(theta, netdefinition)
  m = size(X, 1)
  a1 = [ones(m, 1) X]
  a2 = [ones(m, 1) sigmoid(a1 * theta1')]
  a3 = [ones(m, 1) sigmoid(a2 * theta2')]
  a4 = [ones(m, 1) sigmoid(a3 * theta3')]
  h = sigmoid(a4 * theta4')
  return m, h
end

# Evaluates the loss function of the MLP
function nnloss(theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  m, h = hypothesis(X, theta, netdefinition)
  J = 1/m * sum(sum((-y .* log(h) - (1 - y) .* log(1 - h))))
  loss = J
  return loss::Float64
end

# Evaluate the gradient of the MLP loss w.r.t. Ws and Bs
# The gradient should be stored in the vector 'storage'
function nnlossgrad(theta::Array{Float64,1}, storage::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  # Computes the gradient using "finite differences" and gives a numerical estimate of the gradient
  numgrad = storage
  perturb = zeros(size(theta))
  e = 1e-4
  for p = 1:length(theta)
    # Set perturbation vector
    perturb[p] = e
    loss1 = nnloss(theta - perturb, X, y, netdefinition)
    loss2 = nnloss(theta + perturb, X, y, netdefinition)
    # Compute Numerical Gradient
    numgrad[p] = (loss2 - loss1) / (2*e)
    perturb[p] = 0
  end
  return storage::Array{Float64,1}
end

# Use LBFGS to optmize w and b of the MLP loss
function train(trainfeatures::Array{Float64,2}, trainlabels::Array{Float64,1}, netdefinition::Array{Int, 1})
  sigmaW = 1.0 # Initialize sigma to get WS, bs
  sigmaB = 1.0
  Ws,bs = initWeights(netdefinition, sigmaW, sigmaB)
  theta = weightsToTheta(Ws, bs)
  f(theta) = nnloss(theta, trainfeatures, trainlabels, netdefinition)
  g(theta, storage) = nnlossgrad(theta, storage, trainfeatures, trainlabels, netdefinition)
  res = optimize(f, g, theta, LBFGS()) # use grad calculated through numerical method
                                        # train error: 0.0% test error: 0.83%
  theta = res.minimum
  Ws, bs = thetaToWeights(theta, netdefinition)
  return Ws::Vector{Any}, bs::Vector{Any}
end

# To avoid redundant codes add following functions
#caculate parameters rolled for three layer neural networks
function parameters(theta::Vector{Float64}, netdefinition::Array{Int,1})
  bs, Ws1, Ws2, Ws3, Ws4 = thetaToWsbs(theta, netdefinition)
  theta1 = [bs[1:netdefinition[2]] Ws1]
  theta2 = [bs[netdefinition[2]+1:netdefinition[3]+netdefinition[2]] Ws2]
  theta3 = [bs[netdefinition[3]+netdefinition[2]+1:netdefinition[4]+netdefinition[3]+netdefinition[2]] Ws3]
  theta4 = [bs[netdefinition[4]+netdefinition[3]+netdefinition[2]+1:end] Ws4]
  return theta1, theta2, theta3, theta4
end

# convert unrolled theta to Ws1 Ws2 and bs
function thetaToWsbs(theta::Vector{Float64}, netdefinition::Array{Int,1})
  n1, n2, n3, n4, n5 = N(netdefinition)
  Ws1 = reshape(theta[(n1+1):(n1+n2)], netdefinition[2], netdefinition[1])
  Ws2 = reshape(theta[(n1+n2+1):(n1+n2+n3)], netdefinition[3], netdefinition[2])
  Ws3 = reshape(theta[(n1+n2+n3+1):(n1+n2+n3+n4)], netdefinition[4], netdefinition[3])
  Ws4 = reshape(theta[(n1+n2+n3+n4+1):(n1+n2+n3+n4+n5)], netdefinition[5], netdefinition[4])
  bs = theta[1:n1]
  return bs, Ws1, Ws2, Ws3, Ws4
end

#return neural networks dimensions according to the netdefinition
function N(netdefinition::Array{Int,1})
  n1 = netdefinition[2] + netdefinition[3] + netdefinition[4] + netdefinition[5]
  n2 = netdefinition[2] * netdefinition[1]
  n3 = netdefinition[3] * netdefinition[2]
  n4 = netdefinition[4] * netdefinition[3]
  n5 = netdefinition[5] * netdefinition[4]
  return n1, n2, n3, n4, n5
end

# Predict the classes of the given data points using Ws and Bs.
# p, N x 1 array of Array{Float,2}, contains the output class scores (continuous value) for each input feature.
# c, N x 1 array of Array{Float,2}, contains the output class label (either 0 or 1) for each input feature.
function predict(X::Array{Float64,2}, Ws::Vector{Any}, bs::Vector{Any}, netdefinition::Array{Int,1})
  theta = weightsToTheta(Ws, bs)
  _, h = hypothesis(X, theta, netdefinition)
  p = h
  c = convert(Array{Float64,2}, p .>= 0.5)
  return p::Array{Float64,2}, c::Array{Float64,2}
end

# A helper function which concatenates weights and biases into a variable theta
function weightsToTheta(Ws::Vector{Any}, bs::Vector{Any})
  theta = [bs[:]; Ws[:]]
  theta = convert(Vector{Float64},theta)
  return theta::Vector{Float64}
end

# A helper function which decomposes and reshapes weights and biases from the variable theta
function thetaToWeights(theta::Vector{Float64}, netdefinition::Array{Int,1})
  bs, Ws1, Ws2, Ws3, Ws4 = thetaToWsbs(theta, netdefinition)
  Ws = [Ws1[:]; Ws2[:]; Ws3[:]; Ws4[:]]
  Ws = convert(Vector{Any},Ws)
  bs = convert(Vector{Any},bs)
  return Ws::Vector{Any}, bs::Vector{Any}
end

# Initialize weights and biases from Gaussian distributions
function initWeights(netdefinition::Array{Int,1}, sigmaW::Float64, sigmaB::Float64)
  n1, n2, n3, n4, n5 = N(netdefinition)
  Ws = randn(n2+n3+n4+n5) * sigmaW
  bs = randn(n1) * sigmaB
  Ws = convert(Vector{Any},Ws)
  bs = convert(Vector{Any},bs)
  return Ws::Vector{Any}, bs::Vector{Any}
end

# Problem 2: Multilayer Perceptron
function problem2()

  # LINEAR SEPARABLE DATA
  # load data
  features,labels = loaddata("../data-julia/separable.jld")

  # show data points
  showbefore(features,labels)
  title("Data for Separable Case")

  # train MLP
  Ws,bs = train(features,labels, [2,4,4,4,1])

  # show optimum and plot decision boundary
  showafter(features,labels,Ws,bs, [2,4,4,4,1])
  title("Learned Decision Boundary(5-layer) for Separable Case")


  ## LINEAR NON-SEPARABLE DATA
  # load data
  features,labels = loaddata("../data-julia/nonseparable.jld")

  # show data points
  showbefore(features,labels)
  title("Data for Non-Separable Case")

  # train MLP
  Ws,bs = train(features,labels, [2,4,4,4,1])

  # show optimum and plot decision boundary
  showafter(features,labels,Ws,bs, [2,4,4,4,1])
  title("Learned Decision Boundary(5-layer) for Non-Separable Case")


  # PLANE-BIKE-CLASSIFICATION FROM PROBLEM 1
  # load data
  trainfeatures,trainlabels = loaddata("../data-julia/imgstrain.jld")
  testfeatures,testlabels = loaddata("../data-julia/imgstest.jld")

  # train SVM and predict classes
  Ws,bs = train(trainfeatures,trainlabels, [50,3,3,3,1])
  _,trainpredictions = predict(trainfeatures, Ws, bs, [50,3,3,3,1])
  _,testpredictions = predict(testfeatures, Ws, bs, [50,3,3,3,1])

  # show error
  trainerror = sum(trainpredictions.!=trainlabels)/length(trainlabels)
  testerror = sum(testpredictions.!=testlabels)/length(testlabels)
  println("Training Error Rate: $(round(100*trainerror,2))%")
  println("Testing Error Rate: $(round(100*testerror,2))%")

  return
end
