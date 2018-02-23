using Images
using PyPlot

include("Common.jl")

# Load the rgb image from a2p3.png and convert it to a normalized floating point image.
# Then convert it to a grayscale image.
function loadimage()
  rgb=PyPlot.imread("../data-julia/a2p3.png")
  im=Common.rgb2gray(rgb)
  return im::Array{Float32,2},rgb::Array{Float32,3}
end

# Calculate the structure tensor for the Harris detector.
# Replicate boundaries for filtering.
function computetensor(im::Array{Float32,2},sigma::Float64,fsize::Int)
  g=gaussian2d(sigma,[fsize,fsize])
  g2=gaussian2d(1.6*sigma,[fsize,fsize])
  d=[0.5 0 -0.5]
  smoothed=imfilter(im,g)
  dx=imfilter(smoothed,d)
  dy=imfilter(smoothed,d')
  dx2=imfilter(dx.^2,g2)
  dy2=imfilter(dy.^2,g2)
  dxdy=imfilter(dx.*dy,g2)
  return dx2::Array{Float64,2},dy2::Array{Float64,2},dxdy::Array{Float64,2}
end

# Compute Harris function values from the structure tensor
function computeharris(dx2::Array{Float64,2},dy2::Array{Float64,2},dxdy::Array{Float64,2},sigma::Float64)
  d=dx2.*dy2-dxdy.^2
  t=dx2+dy2
  harris=sigma^4*(d-0.06*t.^2)
  return harris::Array{Float64,2}
end

# Apply non-maximum suppression on the harris function result to extract local maxima
# with a 5x5 window. Allow multiple points with equal values within the same window
# and apply thresholding with the given threshold value.
function nonmaxsupp(harris::Array{Float64,2}, thresh::Float64)
  h_max=Common.imfilter_max(harris,5)
  maxpad=padarray(h_max[3:end-2,3:end-2],[2,2],[2,2],"value",Inf)
  supp=(harris.>=maxpad)&(harris.>thresh)
  py,px=findn(supp)
  return px::Array{Int,1},py::Array{Int,1}
end


# Problem 1: Harris Detector

function problem1()
  # parameters
  sigma = 2.4
  threshold = 1e-6
  fsize = 25

  # load image as color and grayscale images
  im,rgb = loadimage()

  # calculate structure tensor
  dx2,dy2,dxdy = computetensor(im,sigma,fsize)

  # compute harris function
  harris = computeharris(dx2,dy2,dxdy,sigma)

  # display harris images
  figure()
  imshow(harris,"jet",interpolation="none")
  axis("off")
  title("Harris function values")
  gcf()

  # threshold harris function values
  mask = harris .> threshold
  y,x = findn(mask)
  figure()
  imshow(rgb)
  plot(x,y,"xy")
  axis("off")
  title("Harris Interest Points without Non-maximum Suppression")
  gcf()

  # apply non-maxumum suppression
  x,y = nonmaxsupp(harris,threshold)

  # display points ontop of rgb image
  figure()
  imshow(rgb)
  plot(x,y,"xy")
  axis("off")
  title("Harris Interest Points after non-maximum suppression")
  gcf()

  return
end
