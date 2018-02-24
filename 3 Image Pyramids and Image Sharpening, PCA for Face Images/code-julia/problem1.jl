using Images
using PyPlot


# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  if length(size)==1
    size=[size,size]
  end
  rx=(size[2]-1/2)
  dx=(-rx:rx)'
  ry =(size[1]-1)/2
  dy=-ry:ry
  D= dy.^2 .+ dx.^2
  G=exp(-0.5.*D./sigma^2)
  f=G./sum(G)
  return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
  function binomialfactors(n)
    a=[binomial(n-1,i) for i=0:n-1]
    return a
  end

  if length(size)==1
    size=[size,size]
  end
  bx= binomialfactors(size[2])'
  by= binomialfactors(size[1])
  B=by*bx
  f=B./sum(B)
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
  D=A[1:2:end,1:2:end]
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})
  res= zeros(2*size(A,1), 2*size(A,2))
  res[1:2:end,1:2:end]=A
  filt=makebinomialfilter(fsize)
  U=4*imfilter(res,filt,"symmetric")
  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)
  filt=makegaussianfilter(fsize,sigma)
  G=Array(Array{Float64,2},nlevels)
  G[1]=im
  for i=2:nlevels
    G[i]=downsample2(imfilter(G[i-1],filt,"symmetric"))
  end
  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})
  im = (P[1].-minimum(P[1]))./(maximum(P[1]).-minimum(P[1]))
  for i = 2:length(P)
    im = [im [((P[i].-minimum(P[i]))./(maximum(P[i]).-minimum(P[i]))); zeros(size(im,1)-size(P[i],1),size(P[i],2))]]
  end
  figure()
  imshow(im,"gray",interpolation="none")
  axis("off")
  
  return nothing::Void
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})
  L=Array(Array{Float64,2},nlevels)
  L[end]=G[end]
  for i=nlevels-1:-1:1
    L[i]=G[i]-upsample2(G[i+1],fsize)
  end
  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})
  A=deepcopy(L)
  A[1]*=1.5
  A[2]*=1.9
  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})
  function clipimage(I, mi=0.0, ma=1.0)
    I[I.>ma]=ma
    I[I.<mi]=mi
    return I
  end

  im=L[end]
  for i= length(L)-1:-1:1
    im=L[i]+upsample2(im,fsize)
  end
  im = clipimage(im)
  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening

function problem1()
  # parameters
  fsize = [5 5]
  sigma = 1.5
  nlevels = 6

  # load image
  im = PyPlot.imread("../data-julia/a2p1.png")

  # create gaussian pyramid
  G = makegaussianpyramid(im,nlevels,fsize,sigma)

  # display gaussianpyramid
  displaypyramid(G)
  title("Gaussian Pyramid")

  # create laplacian pyramid
  L = makelaplacianpyramid(G,nlevels,fsize)

  # dispaly laplacian pyramid
  displaypyramid(L)
  title("Laplacian Pyramid")

  # amplify finest 2 subands
  L_amp = amplifyhighfreq2(L)

  # reconstruct image from laplacian pyramid
  im_rec = reconstructlaplacianpyramid(L_amp,fsize)

  # display original and reconstructed image
  figure()
  subplot(131)
  imshow(im,"gray",interpolation="none")
  axis("off")
  title("Original Image")
  subplot(132)
  imshow(im_rec,"gray",interpolation="none")
  axis("off")
  title("Reconstructed Image")
  subplot(133)
  imshow(im-im_rec,"gray",interpolation="none")
  axis("off")
  title("Difference")
  gcf()

  return
end
