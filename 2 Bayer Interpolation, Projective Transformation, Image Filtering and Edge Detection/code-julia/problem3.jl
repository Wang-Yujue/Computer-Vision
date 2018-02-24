using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
  dx = [-0.5 0 0.5] # note : imfilter uses correlation
  gy = gaussian2d(1, [3,1])
  fx = gy*dx
  fy = fx'
  return fx::Array{Float64,2},fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I,fx)
  Iy = imfilter(I,fy)
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end

# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2}, Iy::Array{Float64,2}, thr::Float64)
  edges = sqrt(Ix.^2 + Iy.^2)
  edges[edges.<thr] = 0
  return edges::Array{Float64,2}
end

# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
  orientation = atan(Iy./Ix)
  edges2 = copy(edges)
  r = 2:size(edges, 1) + 1
  c = 2:size(edges, 2) + 1

  padedges = padarray(edges, [1 ,1], [1 ,1], "value", 0)

  # left to right edges
  xedges = (orientation .<= pi/8) & (orientation .>- pi/8)
  xnonmax = padedges[r,c] .< max(padedges[r,c-1], padedges[r,c+1])
  edges2[xedges & xnonmax] = 0

  # top -to - bottom edges
  yedges = (orientation .> 3*pi/8) | (orientation .<= -3*pi/8)
  ynonmax = padedges[r,c] .< max(padedges[r-1,c] ,padedges[r+1,c])
  edges2[yedges & ynonmax] = 0

  # bottomleft to topright edges
  xyedges = (orientation .<= 3*pi/8) & (orientation .> pi/8)
  xynonmax = padedges[r,c] .< max(padedges[r-1,c-1], padedges[r+1,c+1])
  edges2[xyedges & xynonmax] = 0

  # topleft to bottomright edges
  yxedges = (orientation .> -3*pi/8) & (orientation .<= -pi/8)
  yxnonmax = padedges[r,c] .< max(padedges[r+1,c-1], padedges[r-1,c+1])
  edges2[yxedges & yxnonmax] = 0

  edges = edges2.>0
  return edges::BitArray{2}
end


#= Problem 3
Image Filtering and Edge Detection =#

function problem3()
  # load image
  img = PyPlot.imread("../data-julia/a1p3.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  Ix, Iy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(Ix, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(Iy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt(Ix.^2 + Iy.^2), "gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 0.1
  edges = detectedges(Ix,Iy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,Ix,Iy)
  figure()
  imshow(edges2.>0, "gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return

  # Small thr leads to more edges but meanwhile introduces more noises.
  # Big thr leads to less edges which is not good for detection.
  # Experiment with various threshold values from above
  # step and choose one that shows the “important" image edges.

end
