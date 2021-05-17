using Test

using MotionCorrespondence


@testset "first" begin

    points_a = [1 6; 2 6; 3 0; 4 5]
    points_b = [1 0; 2 0; 3 0; 4 6]
    points_c = [1 3; 2 3; 3 3; 4 0]

    points_stacked = cat(points_a, points_b, points_c, dims=3)
    correspondence_initial = [1, 2, 3]

    assignment_expected = [
        1 2 3
        1 2 3
        3 1 2
        2 1 3
    ]

    @test establish_correspondence(points_stacked, correspondence_initial) == assignment_expected
end