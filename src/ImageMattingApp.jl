module ImageMattingApp

using Downloads
import ONNXRunTime as OX
using Images
using FileIO

# export download_modnet_model
# export preprocessing
# export postprocessing
export inference

MODNET_MODEL_URL = "https://drive.google.com/u/0/uc?id=1IxxExwrUe4_yQnlEx389tmQI8luX7z5m&export=download"
MODNET_MODEL_PATH = "model/modnet_photographic_portrait_matting.onnx"
REF_SIZE = 512

"""
    download_moddnet_model()

Download ONNX version of MODNet to "model/modnet_photographic_portrait_matting.onnx".
"""
function download_modnet_model()
    mkpath("model")
    Downloads.download(MODNET_MODEL_URL, MODNET_MODEL_PATH)
    @info "Successful download MODNet ONNX model."
    return nothing
end


function preprocessing(img)
    img = collect(channelview(img))  # type: Array {N0f8, 3}, size: (channel(RGB), hight, width)
    img = collect(rawview(img))  # type: Array {UInt8, 3}, size: (channel(RGB), hight, width)
    # im = cv2.imread(args.image_path)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # unify image channels to 3
    if ndims(img) == 2
        # Change size (hight, width) to (1, hight, width)
        img =reshape(img, 1, size(img)...)
    end
    if size(img, 1) == 1
        # Change size (1, hight, width) to (3, hight, width)
        img = repeat(img, 3)
    end
    if size(img, 1) == 4
        # Change size (1, hight, width) to (3, hight, width)
        img = img[1:3, :, :]
    end
    # if len(im.shape) == 2:
    #     im = im[:, :, None]
    # if im.shape[2] == 1:
    #     im = np.repeat(im, 3, axis=2)
    # elif im.shape[2] == 4:
    #     im = im[:, :, 0:3]

    # normalize values to scale it between -1 to 1
    img = (img .- 127.5) / 127.5
    # im = (im - 127.5) / 127.5

    _, img_h, img_w = size(img)
    x, y = get_scale_factor(img_h, img_w, REF_SIZE)
    # im_h, im_w, im_c = im.shape
    # x, y = get_scale_factor(im_h, im_w, ref_size)

    # resize image
    img = imresize(img, ratio=(1, y, x))  # ISSUE: interpolation method is different
    # im = cv2.resize(im, None, fx = x, fy = y, interpolation = cv2.INTER_AREA)

    # prepare input shape
    img = reshape(img, 1, size(img)...)
    img = float32.(img)
    # im = np.transpose(im)
    # im = np.swapaxes(im, 1, 2)
    # im = np.expand_dims(im, axis = 0).astype('float32')

    return img, img_h, img_w
end

function postprocessing(matte, img_h, img_w)
    # refine matte
    matte = dropdims(matte; dims=(1, 2))
    matte = imresize(matte, img_h, img_w)
    # matte = (np.squeeze(result[0]) * 255).astype('uint8')
    # matte = cv2.resize(matte, dsize=(im_w, im_h), interpolation = cv2.INTER_AREA)
    return matte
end

function inference(img)

    if !isfile(MODNET_MODEL_PATH)
        download_modnet_model()
    end

    # img = load(image_path)
    img, img_h, img_w = preprocessing(img)

    # Initialize session and get prediction
    # model_path = OX.testdatapath(MODNET_MODEL_PATH)
    model = OX.load_inference(MODNET_MODEL_PATH)
    # input size: (batch_size, 3, height, width)
    input = Dict("input" => img)
    # output size: (batch_size, 3, height, width)
    result = model(input)

    matte = result["output"]
    matte = postprocessing(matte, img_h, img_w)
    return matte
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

end
