using PyPlot
using JLD

# load and return the provided image
function loadImage()

    img = imread("C:/Users/Yujue/Documents/Courses/Computer Vision/assignment 0/data-julia/a0p2.png")

  return img::Array{Float32, 3}
end

# save the image as a .jld file
function saveFile(img::Array{Float32, 3})

    save("C:/Users/Yujue/Documents/Courses/Computer Vision/assignment 0/data-julia/img.jld", "img", img)
end

# load the .jld file and return the image
function loadFile()

  img = load("C:/Users/Yujue/Documents/Courses/Computer Vision/assignment 0/data-julia/img.jld", "img")

  return img::Array{Float32, 3}
end

# create and return a horizontally mirrored image
function mirrorHorizontal(img::Array{Float32, 3})

  N = size(img)
  imgMirrored = Array(Float32, N)
    for m in 1:N[1]
      for n in 1:N[2]
        imgMirrored[m, n, :] = img[N[1] - m + 1, n, :]
      end
    end

  return imgMirrored::Array{Float32, 3}
end

# display the original and the mirrored image in one plot
function displayImages(img1::Array{Float32, 3}, img2::Array{Float32, 3})

   figure()
   subplot(2, 1, 1)
   imshow(img1)
   axis("off")
   subplot(2, 1, 2)
   imshow(img2)
   axis("off")

end

#= Problem 2
Load and Display Image =#

function problem2()

  img1 = loadImage()

  saveFile(img1)

  img2 = loadFile()

  img2 = mirrorHorizontal(img2)

  displayImages(img1, img2)
end

problem2()
