using Images
using PyPlot
using JLD
using Base.Test

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
  points_hom = [points; ones(1, size(points,2))]
  return points_hom::Array{Float64,2}
end

# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
  points_cart = points[1:end-1,:] ./ (points[end,:]).'
  # If the last corordinates are not ones,
  # first dividing all corordinates with the value of the last,
  # then remove the last coordinates to get Cartesian
  # If the last coordinates are all ones, simply remove them to get Cartesian
  return points_cart::Array{Float64,2}
end

# Translation by v
function gettranslation(v::Array{Float64,1})
  T = eye(4)
  T[1:3,4] = v
  return T::Array{Float64,2}
end

# Rotation of d degrees around x axis
function getxrotation(d::Int)
  r = deg2rad(d)
  Rx = eye(4) ;
  Rx[2:3,2:3] = [cos(r) -sin(r); sin(r) cos(r)]
  return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)
  r = deg2rad(d)
  Ry = eye(4) ;
  Ry[[1 3],[1 3]] = [cos(r) sin(r); -sin(r) cos(r)]
  return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)
  r = deg2rad(d)
  Rz = eye(4) ;
  Rz[1:2,1:2] = [cos(r) -sin(r); sin(r) cos(r)]
  return Rz::Array{Float64,2}
end

# Central projection matrix
function getprojection(principal::Array{Int,1}, focal::Int)
  P = [focal 0 principal[1] 0; 0 focal principal[2] 0; 0 0 1 0]
  P = Array{Float64,2}(P)
  return P::Array{Float64,2}
end

# Return full projection matrix C and full model transformation matrix M
function getfull(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
  M = Rz*Rx*Ry*T
  C = V*M
  return C::Array{Float64,2},M::Array{Float64,2}
end

# Load 2D points
function loadpoints()
  points = load("../data-julia/obj_2d.jld","x")
  return points::Array{Float64,2}
end

# Load z-coordintes
function loadz()
  z = load("../data-julia/zs.jld","Z")
  return z::Array{Float64,2}
end

# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
  P3d = P[:,1:3] \ (cart2hom(P2d).*z)
  return P3d::Array{Float64,2}
end

# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
  X = A \ cart2hom(P3d)
  return X::Array{Float64,2}
end

# Plot 2D points
function displaypoints2d(points::Array{Float64,2})
  plot(points[1,:], points[2,:],".b")
  xlabel("Screen X")
  ylabel("Screen Y")
  return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})
  scatter3D(points[1,:], points[2,:], points[3,:],".b")
  xlabel("World X")
  ylabel("World Y")
  zlabel("World Z")
  return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(C::Array{Float64,2}, X::Array{Float64,2})
  P2d = hom2cart(C*cart2hom(X))
  return P2d::Array{Float64,2}
  #return gcf()::Figure
end


#= Problem 2
Projective Transformation =#

function problem2()
  # parameters
  t               = [-27.1; -2.9; -3.2]
  principal_point = [8; -10]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(135)
  Rx = getxrotation(-30)
  Rz = getzrotation(90)

  # central projection
  P = getprojection(principal_point,focal_length)

  # full projection and model matrix
  C,M = getfull(T,Rx,Ry,Rz,P)

  # load data and plot it
  points = loadpoints()
  figure()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  Xt = invertprojection(P,points,z)
  Xh = inverttransformation(M,Xt)
  worldpoints = hom2cart(Xh)
  figure()
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(C,worldpoints)
  figure()
  displaypoints2d(points2)

  @test_approx_eq points points2
  return

  # Use the obtained 3D points, recompute the projected 2D points using the camera C in projectpoints and show them
  # in a new figure. Get the same 2D points as given!
  # The z coordinates store depth information of points.
  # Transformations(translation and rotation) are not commutative because they are matrix multiplications which are not commutative.

end

#close("all")
