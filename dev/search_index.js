var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = ImageMattingApp","category":"page"},{"location":"#ImageMattingApp","page":"Home","title":"ImageMattingApp","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ImageMattingApp.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ImageMattingApp]","category":"page"},{"location":"#ImageMattingApp.download_modnet_model-Tuple{}","page":"Home","title":"ImageMattingApp.download_modnet_model","text":"download_moddnet_model()\n\nDownload ONNX version of MODNet to \"model/modnetphotographicportrait_matting.onnx\".\n\n\n\n\n\n","category":"method"},{"location":"#ImageMattingApp.inference-Tuple{Any}","page":"Home","title":"ImageMattingApp.inference","text":"inference(img)\n\nObtain a matte portrait from the input image.\n\nArguments\n\nimg: the image for image matting\n\nReturns\n\nmatte: MODNet's output\n\nExamples\n\njulia> matte = inference(img);\n\n\n\n\n\n","category":"method"},{"location":"#ImageMattingApp.postprocessing-Tuple{Any, Any, Any}","page":"Home","title":"ImageMattingApp.postprocessing","text":"postprocessing(matte, img_h, img_w)\n\nPostprocessing for MODNet output.\n\nArguments\n\nmatte: MODNet's output\nimg_h: the height of input image\nimg_w: the width of input image\n\nReturns\n\nmatte: Processed matte\n\n\n\n\n\n","category":"method"},{"location":"#ImageMattingApp.preprocessing-Tuple{Any}","page":"Home","title":"ImageMattingApp.preprocessing","text":"preprocessing(img)\n\nPreprocessing for MODNet input.\n\nArguments\n\nimg: the image to process\n\nReturns\n\nimg: processed image\nimg_h: the height of input image\nimg_w: the width of input image\n\n\n\n\n\n","category":"method"}]
}