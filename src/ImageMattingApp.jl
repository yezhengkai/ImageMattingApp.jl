module ImageMattingApp

using Downloads
import ONNXRunTime as OX
using Images
using FileIO

export matting

include("modnet.jl")

end
