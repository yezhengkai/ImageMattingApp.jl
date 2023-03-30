MODNET_MODEL_URL = "https://drive.google.com/u/0/uc?id=1IxxExwrUe4_yQnlEx389tmQI8luX7z5m&export=download"
MODNET_MODEL_PATH = "model/modnet_photographic_portrait_matting.onnx"
REF_SIZE = Ref{Int}(512)
INFERENCE_SESSION = Ref{OX.InferenceSession}()

"""
    download_moddnet_model()

Download ONNX version of MODNet to "model/modnet_photographic_portrait_matting.onnx".
"""
function download_modnet_model()
    mkpath("model")
    Downloads.download(MODNET_MODEL_URL, MODNET_MODEL_PATH)
    @info "Successfully downloaded MODNet ONNX model."
    return nothing
end

function get_scale_factor(img_h, img_w, ref_size)
    if max(img_h, img_w) < ref_size || min(img_h, img_w) > ref_size
        if img_w >= img_h
            im_rh = ref_size
            im_rw = trunc(Int, img_w / img_h * ref_size)
        elseif img_w < img_h
            im_rw = ref_size
            im_rh = trunc(Int, img_h / img_w * ref_size)
        end
    else
        im_rh = img_h
        im_rw = img_w
    end
    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32

    x_scale_factor = im_rw / img_w
    y_scale_factor = im_rh / img_h

    return x_scale_factor, y_scale_factor
end

"""
    preprocessing(img)

Preprocessing for MODNet input.

# Arguments
- `img`: the image to process

# Returns
- `img`: processed image
- `img_h`: the height of input image
- `img_w`: the width of input image
"""
function preprocessing(img)
    img = collect(channelview(img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img = collect(rawview(img))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)

    # unify image channels to 3
    if ndims(img) == 2
        # Change size (hight, width) to (1, hight, width)
        img =reshape(img, 1, size(img)...)
    end
    if size(img, 1) == 1
        # Change size (1, hight, width) to (3, hight, width)
        img = repeat(img, 3)
    elseif size(img, 1) == 4
        # Change size (4, hight, width) to (3, hight, width)
        img = img[1:3, :, :]
    end

    # normalize values to scale it between -1 to 1
    img = (img .- 127.5) / 127.5

    _, img_h, img_w = size(img)
    x, y = get_scale_factor(img_h, img_w, REF_SIZE[])

    # resize image
    img = imresize(img, ratio=(1, y, x))  # ISSUE: interpolation method is different

    # prepare input shape
    img = reshape(img, 1, size(img)...)
    img = float32.(img)

    return img, img_h, img_w
end

"""
    postprocessing(matte, img_h, img_w)

Postprocessing for MODNet output.

# Arguments
- `matte`: MODNet's output
- `img_h`: the height of input image
- `img_w`: the width of input image

# Returns
- `matte`: Processed matte
"""
function postprocessing(matte, img_h, img_w)
    # refine matte
    matte = dropdims(matte; dims=(1, 2))
    matte = imresize(matte, img_h, img_w)
    return matte
end

"""
    matting(img)

Obtain a matte portrait from the input image.

# Arguments
- `img`: the image for image matting

# Returns
- `matte`: MODNet's output

# Examples
```julia
julia> matte = matting(img);
```
"""
function matting(img)

    if !isfile(MODNET_MODEL_PATH)
        download_modnet_model()
    end

    img, img_h, img_w = preprocessing(img)

    if !isassigned(INFERENCE_SESSION)
        INFERENCE_SESSION[] = OX.load_inference(MODNET_MODEL_PATH)
    end
    # Initialize session and get prediction
    model = INFERENCE_SESSION[]
    # input size: (batch_size, 3, height, width)
    input = Dict("input" => img)
    # output size: (batch_size, 3, height, width)
    result = model(input)

    matte = result["output"]
    matte = postprocessing(matte, img_h, img_w)
    return matte
end
