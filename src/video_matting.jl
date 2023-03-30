function update_img(mode, img, bg)
    matte = matting(img)
    if mode == 0
        return GLMakie.rotr90(img)
    elseif mode == 1
        new_bg = (
            bg .* (1 .- matte)
            +
            img .* matte
        )
        return GLMakie.rotr90(new_bg)
    end
end


function generate_background(background, img_size)
    isnothing(background) || return background
    return ones(RGB, img_size)
end

function keyboard_event!(scene, mode)
    on(events(scene).keyboardbutton) do event
        if event.action == Keyboard.press || event.action == Keyboard.repeat
            if event.key == Keyboard._0
                mode[] = 0
            elseif ispressed(scene, Keyboard._1)
                mode[] = 1
            end
        end
    end
end
function webcam_matting(background=nothing)
    cam = VideoIO.opencamera()
    try
        img = read(cam)
        bg = generate_background(background, size(img))
        mode = GLMakie.Observable(0)
        obs_img = GLMakie.Observable(update_img(mode[], img, bg))

        scene = GLMakie.Scene(camera=GLMakie.campixel!, resolution=reverse(size(img)))
        GLMakie.image!(scene, obs_img)
        keyboard_event!(scene, mode)

        display(scene)

        fps = VideoIO.framerate(cam)
        while GLMakie.isopen(scene)
            img = read(cam)
            obs_img[] = update_img(mode[], img, bg)
            sleep(1 / fps)
        end
    finally
        close(cam)
    end

end
