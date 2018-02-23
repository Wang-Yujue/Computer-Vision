module Common

using Images

export
  rgb2gray,
  sift,
  cart2hom,
  hom2cart,
  sift

function rgb2gray{T<:AbstractFloat}(A::Array{T,3})
  r,c,d = size(A)
  if d != 3
    throw(DimensionMismatch("Input array must be of size NxMx3."))
  end
  gray = similar(A,r,c)
  for j = 1:c
    for i = 1:r
      @inbounds gray[i,j] = 0.299*A[i,j,1] + 0.587*A[i,j,2] + 0.114 *A[i,j,3]
    end
  end
  return gray
end

function cart2hom(points)
  return [points; ones(1,size(points,2))]
end

function hom2cart(points)
  res = points[1:end-1,:] ./ points[end,:]
  return res
end

function imfilter_ord{T<:AbstractFloat}(A::Array{T,2}, fsize::Int, rank::Int)
  r,c = size(A)
  res = similar(A)
  p = div(fsize,2)
  pad = padarray(A,[p,p],[p,p],"symmetric")
  for j = 1:c
    for i = 1:r
      @inbounds patch = pad[i:i+p+p,j:j+p+p]
      @inbounds res[i,j] = select(patch[:],rank)
    end
  end
  return res
end

function imfilter_max{T<:AbstractFloat}(A::Array{T}, fsize::Int)
  fsize = div(fsize,2)*2+1
  rank = fsize*fsize
  res = imfilter_ord(A,fsize,rank)
  return res
end

function sift(points,im,sigma)
  px = points[:,1]
  py = points[:,2]
  n = length(px)
  res = zeros(128,n)
  d = [0.5 0 -0.5]
  g = gaussian2d(sigma,[25 25])
  smoothed = imfilter(im,g)
  dx = imfilter(smoothed,d)
  dy = imfilter(smoothed,d')
  for i = 1:n
    # get patch
    r1 = [Int32(x) for x = (py[i]-7):(py[i]+8)]
    r2 = [Int32(x) for x = (px[i]-7):(px[i]+8)]
    dxp = dx[r1,r2]
    dyp = dy[r1,r2]
    # im2col adaption
    dxc = zeros(16,16)
    dyc = zeros(16,16)
    for c = 1:4
      for r = 1:4
        dxc[:,r+4*(c-1)] = dxp[1+4*(c-1):4*c,1+4*(r-1):4*r][:]
        dyc[:,r+4*(c-1)] = dyp[1+4*(c-1):4*c,1+4*(r-1):4*r][:]
      end
    end
    # compute histogram
    hist8 = zeros(8,16)
    hist8[1,:] = sum(dxc.*(dxc.>0),1) # 0°
    hist8[3,:] = sum(dyc.*(dyc.>0),1) # 90°
    hist8[5,:] = sum(-dxc.*(dxc.<0),1) # 180°
    hist8[7,:] = sum(-dyc.*(dyc.<0),1) # 270°
    idx = dyc .> -dxc
    hist8[2,:] = sum((dyc.*idx .+ dxc.*idx) ./sqrt(2),1) # 45°
    idx = dyc .> dxc
    hist8[4,:] = sum((dyc.*idx .- dxc.*idx) ./sqrt(2),1) # 135°
    idx = dyc .< -dxc
    hist8[6,:] = sum((-dyc.*idx .- dxc.*idx) ./sqrt(2),1) # 225°
    idx = dyc .< dxc
    hist8[8,:] = sum((-dyc.*idx .+ dxc.*idx) ./sqrt(2),1) # 315°
    res[:,i] = hist8[:]
  end
  # normalization
  res = res ./ sqrt(sum(res.^2,1))
  return res
end
end # module
