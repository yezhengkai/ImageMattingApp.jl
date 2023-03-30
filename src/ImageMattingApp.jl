module ImageMattingApp

using Downloads
import ONNXRunTime as OX
using Images
using FileIO
using GLMakie
using VideoIO

export matting, webcam_matting

include("modnet.jl")
include("video_matting.jl")

end
