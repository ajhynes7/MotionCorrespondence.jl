using LinearAlgebra


function proximal_uniformity(X_frames, ϕₖ₋₁, indices_points)
    
    n_points = length(ϕₖ₋₁)
    (Xₖ₋₁, Xₖ, Xₖ₊₁) = X_frames
    
    C₁ = zeros(n_points, n_points)
    C₂ = zeros(n_points, n_points)

    for x = 1:n_points
        for z = 1:n_points

            point_a = Xₖ₋₁[x, :]
            point_b = Xₖ[ϕₖ₋₁[x], :]
            point_c = Xₖ₊₁[z, :]

            vector_ab = point_b - point_a
            vector_bc = point_c - point_b

            C₁[x, z] = norm(vector_ab - vector_bc)
            C₂[x, z] = norm(vector_bc)
        end
    end

    cost₁ = sum(C₁)
    cost₂ = sum(C₂)
    
    p, q, r = indices_points

    point_p = Xₖ₋₁[p, :]
    point_q = Xₖ[q, :]
    point_r = Xₖ₊₁[r, :]

    vector_pq = point_q - point_p
    vector_qr = point_r - point_q

    return norm(vector_pq - vector_qr) / cost₁ + norm(vector_qr) / cost₂
end


function frame_correspondence(X_frames, ϕₖ₋₁)
    
    n_points = length(ϕₖ₋₁)

    M = Matrix{Union{Missing,Float64}}(undef, n_points, n_points)

    for i = 1:n_points
        for j = 1:n_points
            for p = 1:n_points

                if ϕₖ₋₁[p] == 1
                    M[i, j] = proximal_uniformity(X_frames, ϕₖ₋₁, (p, i, j))
                end
            end
        end
    end

    # Compute ϕₖ, the assignment of points in frame k to points in frame k + 1
    ϕₖ = zero(ϕₖ₋₁)

    for _ = 1:n_points

        # Construct priority matrix B.
        B = Matrix{Union{Missing,Float64}}(undef, n_points, n_points)

        for i = 1:n_points

            if all(ismissing, M[i, :])
                # This row has been already masked with NaN.
                continue
            end
    
            # Find the minimum column of M.
            lᵢ = argmin(skipmissing(M[i, :]))

            # Sum of row i of M, excluding column lᵢ.
            sum_row = sum(skipmissing(M[i, :])) - M[i, lᵢ]

            # Sum of column lᵢ of M, excluding row i.
            sum_col = sum(skipmissing(M[:, lᵢ])) - M[i, lᵢ]

            B[i, lᵢ] = sum_row + sum_col
        end

        row_max, col_max = Tuple(argmax(skipmissing(B)))

        ϕₖ[row_max] = col_max

        M[row_max, :] .= missing
        M[:, col_max] .= missing
    end
        
    return ϕₖ
end


function establish_correspondence(points_stacked, correspondence_initial)

    n_frames, _, n_points = size(points_stacked)

    # Begin with previously known correspondence between frames.
    Φ = [Vector{Int}(undef, n_points) for _ = 1:n_frames - 1]
    Φ[1] = correspondence_initial

    for k  = 2:n_frames - 1

        # Points from previous, current, and next frame.
        Xₖ₋₁ = points_stacked[k - 1, :, :]'
        Xₖ = points_stacked[k, :, :]'
        Xₖ₊₁ = points_stacked[k + 1, :, :]'

        X_frames = (Xₖ₋₁, Xₖ, Xₖ₊₁)

        Φ[k] = frame_correspondence(X_frames, Φ[k - 1])
    end

    # Assign labels to the points.
    assignment = zeros(Int, n_frames, n_points)
    assignment[1, :] = 1:n_points

    for k = 2:n_frames
        correspondence_prev = Φ[k - 1]
        assignment[k, :] = assignment[k - 1, correspondence_prev]
    end

    return assignment
end
