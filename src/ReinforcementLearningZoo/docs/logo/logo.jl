using Javis

function ground(args...)
    background("white")
    sethue("white")
end

video = Video(400, 450)

U = 50
SHIFT = Point(0, -1.7U)

function tangram(obj)
    if obj == :triangle1
        sethue("#C9E891")
        [O, Point(2U, 0), Point(0, 2U)]
    elseif obj == :triangle2
        sethue("#69C0D2")
        [O, Point(-2U, 0), Point(0, 2U)]
    elseif obj == :triangle3
        sethue("#F8CB2D")
        [O, Point(-U, U), Point(U, U)]
    elseif obj == :triangle4
        sethue("#F195C8")
        [O, Point(-U, 0), Point(0, -U)]
    elseif obj == :triangle5
        sethue("#F9F224")
        [O, Point(U, 0), Point(0, -U)]
    elseif obj == :box
        sethue("#A796C2")
        [O, Point(0, -U), Point(U, -U), Point(U, 0)]
    elseif obj == :parallelogram
        sethue("#EF3E62")
        [O, Point(U, -U), Point(0, -U), Point(-U, 0)]
    end
end

javis(
    video,
    [
        BackgroundAction(1:400, ground),
        Action(
            1:400,
            (args...) -> poly(tangram(:triangle1), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(50:60, Translation(O, Point(0, 2U))),
                SubAction(60:70, Rotation(0.0, -π / 2)),
                SubAction(70:80, Translation(O, Point(-2U, 0))),
                SubAction(80:90, Translation(O, Point(U, -U))),
            ],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:triangle2), :fill, close = true);
            subactions = [SubAction(1:1, Translation(O, SHIFT))],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:triangle3), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(1:1, Translation(O, Point(0, -2U))),
                SubAction(100:110, Translation(O, Point(-3U, 0))),
                SubAction(110:120, Rotation(0.0, -π / 2)),
                SubAction(120:130, Translation(O, Point(-5.5U, 0))),
                SubAction(130:140, Translation(O, Point(0, U))),
            ],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:triangle5), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(1:1, Translation(O, Point(U, 0))),
                SubAction(150:160, Translation(O, Point(0, 4.5 * U))),
                SubAction(160:170, Rotation(0.0, -3π / 4)),
                SubAction(
                    170:180,
                    Translation(
                        O,
                        Point(sqrt((2 - √2 / 2)^2 / 2) * U, -sqrt((2 - √2 / 2)^2 / 2) * U),
                    ),
                ),
            ],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:parallelogram), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(1:1, Translation(O, Point(-U, 0))),
                SubAction(200:210, Translation(O, Point(U, -U))),
                SubAction(210:220, Translation(O, Point(U, 0))),
                SubAction(220:230, Rotation(0.0, π / 2)),
                SubAction(230:240, Translation(O, Point(2U, 0))),
                SubAction(240:250, Translation(O, Point(0, U))),
            ],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:triangle4), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(250:260, Translation(O, Point(0, -U))),
                SubAction(260:270, Translation(O, Point(2U, 0))),
                SubAction(270:280, Rotation(0.0, -π / 2)),
                SubAction(280:290, Translation(O, Point(-4U, 0))),
                SubAction(290:300, Translation(O, Point(0.5 * U, -0.5 * U))),
            ],
        ),
        Action(
            1:400,
            (args...) -> poly(tangram(:box), :fill, close = true);
            subactions = [
                SubAction(1:1, Translation(O, SHIFT)),
                SubAction(300:310, Translation(O, Point(-U, 0))),
                SubAction(310:320, Rotation(0.0, -π / 6)),
            ],
        ),
    ];
    pathname = "logo.gif",
)
