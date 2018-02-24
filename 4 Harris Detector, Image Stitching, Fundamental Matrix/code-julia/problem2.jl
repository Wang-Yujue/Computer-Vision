using Images
using PyPlot
using Grid

include("Common.jl")

# Load Harris interest points of both images
function loadkeypoints(path::ASCIIString)
  data = load("../data-julia/keypoints.jld")
  keypoints1 = data["keypoints1"]
  keypoints2 = data["keypoints2"]
  @assert size(keypoints1,2) == 2 # Nx2
  @assert size(keypoints2,2) == 2 # Kx2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end

# Compute pairwise squared euclidean distances for given features
function euclideansquaredist(f1::Array{Float64,2},f2::Array{Float64,2})
  D = (sum(f1.^2,1)' .+ sum(f2.^2,1)) - 2*(f1'*f2)
  @assert size(D) == (size(f1,2),size(f2,2))
  return D::Array{Float64,2}
end

# Find pairs of corresponding interest points based on the distance matrix D.
# p1 is a Nx2 and p2 a Mx2 vector describing the coordinates of the interest
# points in the two images.
# The output should be a min(N,M)x4 vector such that each row holds the coordinates of an
# interest point in p1 and p2.
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
  if size(p1,1) <= size(p2,1)
     _,lidx = findmin(D,2)
     idx = zeros(Integer,size(lidx,1))
     for i = 1:length(lidx)
       _,tmp = ind2sub(size(D),lidx[i])
       idx[i] = tmp
     end
     pairs = [p1 p2[idx,:]]
   else
     _,lidx = findmin(D,1)
     idx = zeros(Integer,size(lidx,1))
     for i = 1:length(lidx)
       tmp,_ = ind2sub(size(D),lidx[i])
       idx[i] = tmp
     end
     pairs = [p1[idx,:] p2]
   end
  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end

# Show given matches on top of the images in a single figure, in a single plot.
# Concatenate the images into a single array.
function showmatches(im1::Array{Float32,2},im2::Array{Float32,2},pairs::Array{Int,2})
  figure()
  imshow([im1 im2],"gray",interpolation="none")
  axis("off")
  c = size(im1,2)
  n = size(pairs,1)
  for i = 1:n
    plot([pairs[i,1], pairs[i,3] + c], [pairs[i,2], pairs[i,4]])
  end
  return nothing::Void
end

# Compute the estimated number of iterations for RANSAC
function computeransaciterations(p::Float64,k::Int,z::Float64)
  n = ceil(log(1-z)/log(1-p^k))
  n = Int(n)
  return n::Int
end

# Randomly select k corresponding point pairs
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
  idx = randperm(size(points1,1))[1:k]
  sample1 = points1[idx,:]
  sample2 = points2[idx,:]
  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end

# Apply conditioning.
# Return the conditioned points and the condition matrix.
function condition(points::Array{Float64,2})
  P = points[1:2,:]./points[3,:]'
  t = mean(P,2)
  s = 0.5 * maximum(abs(P))
  T = [[eye(2) -t]./s; 0 0 1]
  U = T * points
  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end

# Estimate the homography from the given correspondences
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
  points1 = Common.cart2hom(points1')
  points2 = Common.cart2hom(points2')

  x1 = points1[1,:]
  y1 = points1[2,:]
  x2 = points2[1,:]
  y2 = points2[2,:]
  O = zeros(size(x1,1))
  I = ones(size(x1,1))
  A2 = [O O O x1 y1 I -x1.*y2 -y1.*y2 -y2;-x1 -y1 -I O O O x1.*x2 y1.*x2 x2]
  U1,T1 = condition(points1)
  U2,T2 = condition(points2)
  x1 = U1[1,:]
  y1 = U1[2,:]
  x2 = U2[1,:]
  y2 = U2[2,:]
  O = zeros(size(x1,1))
  I = ones(size(x1,1))
  A = [O O O x1 y1 I -x1.*y2 -y1.*y2 -y2;-x1 -y1 -I O O O x1.*x2 y1.*x2 x2]

  @printf("Condition number before:%.2f\t after %.2f\n",cond(A2),cond(A))

  _,_,V = svd(A,thin=false)
  Hb = reshape(V[:,end],(3,3))'
  H = T2\Hb*T1
  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end

function hom2cartcorrect(points::Array{Float64,2})
  points_cart = points[1:end-1,:] ./ (points[end,:]).'
  # If the last corordinates are not ones,
  # first dividing all corordinates with the value of the last,
  # then remove the last coordinates to get Cartesian
  # If the last coordinates are all ones, simply remove them to get Cartesian
  return points_cart::Array{Float64,2}
end

# Compute distances for keypoints after transformation with the given homography
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
  x1 = Common.cart2hom(points1')
  x2 = Common.cart2hom(points2')
  x1t = H\x2
  x2t = H*x1
  x1t = Common.cart2hom(hom2cartcorrect(x1t))
  x2t = Common.cart2hom(hom2cartcorrect(x2t))
  d2 = sum((x1-x1t).^2 + (x2-x2t).^2,1)
  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end

# Compute the inliers for a given homography distance and threshold
function findinliers(distance::Array{Float64,2},thresh::Float64)
  indices = find(distance .< thresh)
  n = length(indices)
  return n::Int,indices::Array{Int,1}
end

# RANSAC algorithm
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)
  points1 = pairs[:,1:2]
  points2 = pairs[:,3:4]
  bestinliers = []
  bestpairs = []
  bestH = []
  for i = 1:n
    sample1,sample2 = picksamples(points1, points2, 4)
    H = computehomography(sample1, sample2)
    d2 = computehomographydistance(H, points1, points2)
    ninliers,indices = findinliers(d2, thresh)
    if ninliers > length(bestinliers)
      bestinliers = indices
      bestpairs = [sample1 sample2]
      bestH = H
    end
  end
  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end

# Recompute the homography based on all inliers
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})
  inliers1 = pairs[inliers,1:2]
  inliers2 = pairs[inliers,3:4]
  H = computehomography(inliers1,inliers2)
  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end

# Show panorama stitch of both images using the given homography.
function showstitch(im1::Array{Float32,2},im2::Array{Float32,2},H::Array{Float64,2})
  im2interp = InterpGrid(im2,0,InterpLinear)
  height,width = size(im1)
  warped = [im1[:,1:300] zeros(height,400)]
  for x = 301:700
    for y = 1:size(im1,1)
      ix,iy = Common.hom2cart(H*[x;y;1])
      warped[y,x] = im2interp[iy,ix]
    end
  end
  figure()
  imshow(warped,"gray",interpolation="none")
  axis("off")
  title("Stitched Image")
  return nothing::Void
end

# Problem 2: Image Stitching

function problem2()
  #SIFT Parameters
  sigma = 1.4              # standard deviation
  # RANSAC Parameters
  ransac_threshold = 50.0   # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("../data-julia/a3p1a.png") # left image
  im2 = PyPlot.imread("../data-julia/a3p1b.png") # right image

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("../data-julia/keypoints.jld")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute squared euclidean distance matirx
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return
end
