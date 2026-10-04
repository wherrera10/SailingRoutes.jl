"""
    SailingRoutes

A Julia module for weather routing for sailboats.

# Direction conventions
- Angles are measured in degrees, following rather standard nautical bearing convention:
- Wind direction (`winddeg`) is the direction in degrees from north that the wind blows *from*.
- Current direction (`currentdeg`) is the direction in degrees from north that the current flows *toward*.
- Given a nautical bearing, the sine and cosine of the bearing is used to decompose
  that angle direction into its north-south (y) and east-west (x) components, respectively.

# Architecture
- `SailingPolar`: Represents boat performance data with variations in wind
- `SurfaceParameters`: Wind and current conditions for the problem
- `RoutingProblem`: The optimization problem, with time variant conditions for wind and current
- `minimumtimeroute`: A* search implementation for weather routing

# Usage Example
```julia
polar = getpolardata("polar.csv")
rp = createroutingproblem(...)
result = solveroutingproblem(polar, rp)
"""
module SailingRoutes

using DelimitedFiles
using DataStructures
using Interpolations

# Type Exports
export SailingPolar, SurfaceParameters, Position, GridPosition, GridPoint
export TimeSlice, TimedPath, RoutingProblem

# Earth Geometry Exports
export getpolardata, deg2rad, rad2deg, angledifference, normalizeangledegrees
export cartesian2polar, polar2cartesian, haversine, inverse_haversine

# Sailing Performance Exports
export boatspeed, bestvectorspeed, vmg, sailsegmenttime
export interpolatepolar, findbesttack

# Weather Routing Exports
export closestpoint, minimumtimeroute, gettimeslice
export solveroutingproblem, createroutingproblem, validateroutingproblem

# Constants
const EARTH_RADIUS_M = 6_372_800.0
const KNOT = 0.514444444444
const SECONDS_PER_MINUTE = 60.0

# Type aliases
const Latitude = Float64
const Longitude = Float64
const Degrees = Float64
const Knots = Float64

"""
    Position(lat, lon)

Represents a geographic position with latitude and longitude in degrees.
The two-argument constructor converts real values to `Float64`; tuple and
vector constructors accept `(latitude, longitude)` values in that order.
"""
struct Position
    lat::Latitude
    lon::Longitude
end

Position(lat::Real, lon::Real) = Position(Float64(lat), Float64(lon))
Position(p::Tuple) = Position(p[1], p[2])
Position(p::AbstractVector) = Position(p[1], p[2])

Base.show(io::IO, p::Position) = print(io, "Position($(p.lat), $(p.lon))")

"""
    SailingPolar

Polar sailing performance data indexed by wind angle and wind speed.
"""
struct SailingPolar{I}
    winds::Vector{Float64}
    degrees::Vector{Float64}
    speeds::Matrix{Float64}
    maxboatspeed::Float64
    interpolator::I

    function SailingPolar(winds, degrees, speeds)
        winds = Float64.(winds)
        degrees = Float64.(degrees)
        speeds = Float64.(speeds)

        if length(winds) != size(speeds, 2)
            throw(
                ArgumentError(
                    "Number of wind speeds ($(length(winds))) must match columns in speeds matrix ($(size(speeds, 2)))",
                ),
            )
        end
        if length(degrees) != size(speeds, 1)
            throw(
                ArgumentError(
                    "Number of degrees ($(length(degrees))) must match rows in speeds matrix ($(size(speeds, 1)))",
                ),
            )
        end
        if any(<(0), winds) || any(<(0), degrees)
            throw(ArgumentError("Wind speeds and degrees must be non-negative"))
        end
        if !issorted(winds) || !issorted(degrees)
            throw(ArgumentError("Winds and degrees must be sorted in ascending order"))
        end

        itp = interpolate((degrees, winds), speeds, Gridded(Linear()))
        new{typeof(itp)}(winds, degrees, speeds, maximum(speeds), itp)
    end
end

"""
    SurfaceParameters

Wind and current direction/velocity for a position.
Angles in degrees, velocities in knots.

Following standard nautical convention, `winddeg` is the direction the wind
blows *from* (e.g. a "north wind" is 0°/360° and blows toward the south),
while `currentdeg` is the *set* — the direction the current flows *toward*
(e.g. a current with a set of 090° moves water eastward).
"""
Base.@kwdef struct SurfaceParameters
    winddeg::Float64 = 0.0
    windkts::Float64 = 0.0
    currentdeg::Float64 = 0.0
    currentkts::Float64 = 0.0
end

function Base.show(io::IO, sp::SurfaceParameters)
    print(
        io,
        "SurfaceParameters(wind=$(sp.winddeg)° @ $(sp.windkts) kts, current=$(sp.currentdeg)° @ $(sp.currentkts) kts)",
    )
end

"""
    GridPosition

A grid cell identified by its one-based `(row, column)` indices.
The two-argument constructor converts integer indices to `Int`.
"""
const GridPosition = Tuple{Int, Int}
@inline GridPosition(r::Integer, c::Integer) = (Int(r), Int(c))

"""
    GridPoint

A geographic grid cell with its position and the wind and current conditions
at that location. The six-argument constructor takes latitude, longitude,
wind direction, wind speed, current direction, and current speed, in that order.
Directions are in degrees and speeds are in knots.
"""
struct GridPoint
    pt::Position
    sp::SurfaceParameters
end

GridPoint(lat::Real, lon::Real, winddeg::Real, windkts::Real, currentdeg::Real, currentkts::Real) =
    GridPoint(
        Position(lat, lon),
        SurfaceParameters(Float64(winddeg), Float64(windkts), Float64(currentdeg), Float64(currentkts)),
    )

Base.show(io::IO, gp::GridPoint) = print(io, "GridPoint($(gp.pt), $(gp.sp))")

"""
    TimeSlice

A matrix of `GridPoint`s representing positions and conditions at one time.
"""
const TimeSlice = Matrix{GridPoint}

"""
    TimedPath(duration, path)

A route with elapsed travel time in minutes and a vector of grid positions.
The duration must be nonnegative and the path must contain at least one point.
"""
struct TimedPath
    duration::Float64
    path::Vector{GridPosition}

    function TimedPath(duration, path)
        duration = Float64(duration)
        duration < 0 && throw(ArgumentError("Duration cannot be negative"))
        isempty(path) && throw(ArgumentError("Path cannot be empty"))
        new(duration, path)
    end
end

Base.isless(a::TimedPath, b::TimedPath) = a.duration < b.duration

function Base.show(io::IO, tp::TimedPath)
    print(io, "TimedPath($(tp.duration) minutes, $(length(tp.path)) points)")
end

"""
    getpolardata(filename)

Read a sailing polar CSV file, consisting of separated values, but using ';' not ','
as the delimiter. Throws an ArgumentError if file cannot be read or has invalid content.
"""
function getpolardata(filename)
    try
        datacells, headercells = readdlm(filename, ';', header = true)
        winds = parse.(Float64, strip.(headercells[2:end]))
        degrees = Float64.(datacells[:, 1])
        speeds = Float64.(datacells[:, 2:end])
        return SailingPolar(winds, degrees, speeds)
    catch e
        throw(ArgumentError("Failed to read valid polar data from file '$filename': $e"))
    end
end

const ANGLE_TOL = 1e-10
"""
    normalizeangledegrees(angle)

Normalize an angle in degrees to the range [0, 360).
"""
@inline function normalizeangledegrees(angle::Float64)
    a = mod(angle, 360.0)
    # Snap values extremely close to 0° or 360° to 0°
    a < ANGLE_TOL || 360.0 - a < ANGLE_TOL ? 0.0 : a
end

"""
    cartesian2polar(x, y)

Convert x, y coordinates to polar coordinates with angle in degrees.
"""
@inline function cartesian2polar(x, y)
    r = hypot(x, y)
    θ = rad2deg(atan(y, x))
    return r, mod(θ + 360.0, 360.0)
end

"""
    polar2cartesian(r, deg)

Convert polar coordinates in degrees to cartesian x, y coordinates.
"""
@inline function polar2cartesian(r, deg)
    s, c = sincosd(deg)
    return r * c, r * s
end

"""
    haversine(lat1, lon1, lat2, lon2)

Return distance in meters and initial bearing in degrees.
"""
@inline function haversine(lat1, lon1, lat2, lon2)
    lat1 = deg2rad(lat1)
    lon1 = deg2rad(lon1)
    lat2 = deg2rad(lat2)
    lon2 = deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
    clamped_a = clamp(a, 0.0, 1.0)
    c = 2 * atan(sqrt(clamped_a), sqrt(1 - clamped_a))

    y = sin(dlon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    θ = mod(rad2deg(atan(y, x)) + 360.0, 360.0)

    distance = EARTH_RADIUS_M * c
    return distance, θ
end

@inline haversine(p1::Position, p2::Position) = haversine(p1.lat, p1.lon, p2.lat, p2.lon)

"""
    inverse_haversine(lat1, lon1, distance, direction)

Return the destination latitude and longitude in degrees.
Longitude is normalized to [-180, 180).
"""
function inverse_haversine(lat1_degrees, lon1_degrees, distance, direction)
    lat1 = deg2rad(lat1_degrees)
    lon1 = deg2rad(lon1_degrees)
    dir = deg2rad(direction)
    dist = distance / EARTH_RADIUS_M

    lat2 = asin(sin(lat1) * cos(dist) + cos(lat1) * sin(dist) * cos(dir))
    lon2 = lon1 + atan(sin(dir) * sin(dist) * cos(lat1),
        cos(dist) - sin(lat1) * sin(lat2))

    lon_deg = rad2deg(lon2)
    lon_deg = mod(lon_deg + 180.0, 360.0) - 180.0
    return rad2deg(lat2), lon_deg
end

"""
    interpolatepolar(polar, windangle, windspeed)

Bilinear interpolation over polar data.
"""
function interpolatepolar(polar::SailingPolar, windangle::Float64, windspeed::Float64)
    windangle = clamp(windangle, polar.degrees[1], polar.degrees[end])
    windspeed = clamp(windspeed, polar.winds[1], polar.winds[end])
    return polar.interpolator(windangle, windspeed)
end

"""
    boatspeed(polar, windangle, windspeed)

Return the boat's speed in knots at the given wind angle and wind speed,
using the polar's interpolated performance data. Values outside the polar's
data ranges are clamped to the nearest boundary.
"""
@inline boatspeed(polar::SailingPolar, windangle::Float64, windspeed::Float64) =
    interpolatepolar(polar, windangle, windspeed)

"""
    angledifference(a, b)

Smallest absolute difference between two compass angles in degrees,
correctly handling wraparound at 0°/360° (e.g. the difference between
350° and 10° is 20°, not 340°). Result is in [0, 180].
"""
@inline function angledifference(a, b)
    d = mod(a - b, 360.0)
    return d > 180.0 ? 360.0 - d : d
end

"""
    vmg(speed, heading, target)

Calculate the velocity made good (VMG) toward a target direction.
`speed` is the boat speed, `heading` is the current heading in degrees,
and `target` is the target direction in degrees.
"""
@inline vmg(speed, heading, target) = speed * cosd(angledifference(heading, target))

"""
    findbesttack(polar, targetangle, windspeed)

Return the best angle and VMG for a target point of sail.
"""
function findbesttack(polar::SailingPolar, targetangle::Float64, windspeed::Float64)
    bestangle = targetangle
    bestvmg = -Inf

    for angle in polar.degrees
        speed = boatspeed(polar, clamp(angle, 0.0, 180.0), windspeed)
        speed <= 0 && continue
        calculatedvmg = speed * cosd(angledifference(angle, targetangle))
        if calculatedvmg > bestvmg
            bestvmg = calculatedvmg
            bestangle = angle
        end
    end

    return bestangle, bestvmg
end

"""
    bestvectorspeed(polar, dirtravel, dirwind, windspeed, dircur, currentspeed)

Determine the boat heading that maximizes progress toward `dirtravel`,
taking current into account.

Returns `(heading, sog)` where

- `heading` is the recommended boat heading through the water.
- `sog` is the resulting speed over ground.

The optimization criterion is velocity made good (VMG) toward the desired
track, while the returned speed is the actual speed over ground.
"""
function bestvectorspeed(
    polar::SailingPolar,
    dirtravel::Float64,
    dirwind::Float64,
    windspeed::Float64,
    dircur::Float64,
    currentspeed::Float64,
)
    isempty(polar.degrees) && return dirtravel, currentspeed

    # Unit vector toward desired travel direction
    tx, ty = polar2cartesian(1.0, dirtravel)

    # Current vector
    curx, cury = polar2cartesian(currentspeed, dircur)

    bestheading = dirtravel
    bestsog = currentspeed
    bestvmg = -Inf

    for relangle in polar.degrees

        boatspd = boatspeed(polar, relangle, windspeed)
        boatspd <= 1e-9 && continue

        for sign in (-1.0, 1.0)
            heading = normalizeangledegrees(dirwind - sign * relangle)
            boatx, boaty = polar2cartesian(boatspd, heading)
            gx = boatx + curx
            gy = boaty + cury
            sog = hypot(gx, gy)
            sog <= 0 && continue
            # Projection of ground velocity onto desired track
            vmg = gx * tx + gy * ty

            if vmg > bestvmg
                bestvmg = vmg
                bestheading = heading
                bestsog = sog
            end
        end
    end

    # No propulsion: simply drift with the current.
    if bestvmg == -Inf
        return dirtravel, currentspeed
    end

    return bestheading, bestsog
end

"""
    sailsegmenttime(polar, p, lat1, lon1, lat2, lon2)

Trip time in minutes for one segment.
"""
function sailsegmenttime(polar::SailingPolar, p::SurfaceParameters, lat1, lon1, lat2, lon2)
    distance, dir = haversine(lat1, lon1, lat2, lon2)
    _, vel = bestvectorspeed(polar, dir, p.winddeg, p.windkts, p.currentdeg, p.currentkts)
    vel <= 0 && return Inf
    return distance / (vel * KNOT) / SECONDS_PER_MINUTE
end

"""
    RoutingProblem(timeinterval, timeframe, obstacleindices, start, finish)

Describes a route-search problem on a time-varying grid. `timeinterval` is
the duration in minutes represented by each time slice, `timeframe` is a
nonempty vector of equally sized `TimeSlice`s, and `obstacleindices` marks
blocked cells. `start` and `finish` are `(row, column)` grid positions.
The constructor checks grid dimensions, bounds, and that the endpoints are
not obstacles.
"""
mutable struct RoutingProblem
    timeinterval::Float64
    timeframe::Vector{TimeSlice}
    obstacleindices::BitMatrix
    start::GridPosition
    finish::GridPosition

    function RoutingProblem(timeinterval, timeframe, obstacleindices, start, finish)
        timeinterval = Float64(timeinterval)
        timeinterval > 0 || throw(ArgumentError("timeinterval must be positive"))
        isempty(timeframe) && throw(ArgumentError("timeframe cannot be empty"))

        if !(start isa Tuple && length(start) == 2 && finish isa Tuple && length(finish) == 2)
            throw(ArgumentError("start and finish must be 2-tuples of grid indices"))
        end

        t0 = timeframe[1]
        if !checkbounds(Bool, t0, start[1], start[2]) ||
           !checkbounds(Bool, t0, finish[1], finish[2])
            throw(ArgumentError("Start and finish positions must be within grid bounds"))
        end

        refsize = size(timeframe[1])
        for slice in timeframe
            size(slice) == refsize ||
                throw(ArgumentError("All time slices must have identical dimensions"))
        end

        obs_matrix = obstacleindices isa BitMatrix ? obstacleindices : BitMatrix(obstacleindices)

        if size(obs_matrix) != size(t0)
            throw(ArgumentError("obstacleindices must have the same size as each time slice"))
        end
        if obs_matrix[start[1], start[2]] || obs_matrix[finish[1], finish[2]]
            throw(ArgumentError("Start and finish positions cannot be obstacles"))
        end

        new(timeinterval, timeframe, obs_matrix, (Int(start[1]), Int(start[2])), (Int(finish[1]), Int(finish[2])))
    end
end

"""
    gettimeslice(rp, duration)

Get the appropriate time slice for a given elapsed duration.
"""
function gettimeslice(rp::RoutingProblem, duration::Float64)
    idx = Int(floor(duration / rp.timeinterval)) + 1
    idx = clamp(idx, 1, length(rp.timeframe))
    return rp.timeframe[idx]
end

"""
    closestpoint(p, mat)

Return the closest GridPoint in a matrix to position `p`.
"""
function closestpoint(p::Position, mat::Matrix{GridPoint})
    mindist = Inf
    minidx = (1, 1)

    for i in axes(mat, 1), j in axes(mat, 2)
        q = mat[i, j].pt
        dlat = p.lat - q.lat
        latmean = (p.lat + q.lat)/2
        dlon = (p.lon - q.lon) * cosd(latmean)
        dist = dlat * dlat + dlon * dlon
        if dist < mindist
            mindist = dist
            minidx = (i, j)
        end
    end

    return mat[minidx...]
end

"""
    heuristiccostestimate(rp, polar, from, to)

Conservative lower-bound heuristic for A*. Uses the theoretical maximum boat speed
without a favorable current, so the heuristic remains admissible regardless of current.
"""
function heuristiccostestimate(
    rp::RoutingProblem,
    polar::SailingPolar,
    from::GridPosition,
    to::GridPosition,
)
    p1 = rp.timeframe[1][from...].pt
    p2 = rp.timeframe[1][to...].pt
    distance, _ = haversine(p1, p2)
    return distance / (max(polar.maxboatspeed, 1e-9) * KNOT * SECONDS_PER_MINUTE)
end

const NEIGHBOR_OFFSETS = (
    GridPosition(-1, -1),
    GridPosition(-1, 0),
    GridPosition(-1, 1),
    GridPosition(0, -1),
    GridPosition(0, 1),
    GridPosition(1, -1),
    GridPosition(1, 0),
    GridPosition(1, 1),
)

"""
    minimumtimeroute(rp, polar; verbose=false, maxiterations=10_000)

The optimization for which all the other types and functions in the module are defined.
Finds the minimum time route for a sailing problem using A* search. Returns a `TimedPath`
object containing the total travel time and the path as array of `GridPosition` objects.
"""
function minimumtimeroute(
    rp::RoutingProblem,
    polar::SailingPolar;
    verbose::Bool = false,
    maxiterations::Int = 10_000,
)
    start = rp.start
    goal = rp.finish
    start == goal && return TimedPath(0.0, [start])

    nrows, ncols = size(rp.obstacleindices)
    openset = PriorityQueue{GridPosition, Float64}()
    openset[start] = heuristiccostestimate(rp, polar, start, goal)
    closed = falses(nrows, ncols)
    gscore = fill(Inf, nrows, ncols)
    gscore[start...] = 0.0
    arrivaltime = fill(Inf, nrows, ncols)
    arrivaltime[start...] = 0.0

    parent = fill(GridPosition(0, 0), nrows, ncols)
    iterations = 0
    while !isempty(openset)
        iterations += 1
        iterations > maxiterations && break
        current = dequeue!(openset)
        closed[current...] && continue

        if current == goal
            path = GridPosition[]
            node = goal
            while node != GridPosition(0, 0)
                push!(path, node)
                node = parent[node...]
            end
            reverse!(path)
            return TimedPath(gscore[goal...], path)
        end

        closed[current...] = true

        if verbose && iterations % 1000 == 0
            println("Iteration $iterations   Open=$(length(openset))")
        end

        slice = gettimeslice(rp, arrivaltime[current...])
        gp1 = slice[current...]

        @inbounds for off in NEIGHBOR_OFFSETS
            rr = current[1] + off[1]
            cc = current[2] + off[2]
            if !(1 <= rr <= nrows && 1 <= cc <= ncols)
                continue
            end
            rp.obstacleindices[rr, cc] && continue
            closed[rr, cc] && continue
            gp2 = slice[rr, cc]
            dt = sailsegmenttime(
                polar,
                gp1.sp,
                gp1.pt.lat,
                gp1.pt.lon,
                gp2.pt.lat,
                gp2.pt.lon,
            )
            isfinite(dt) || continue
            tentative = gscore[current...] + dt
            tentative >= gscore[rr, cc] && continue
            parent[rr, cc] = current
            gscore[rr, cc] = tentative
            arrivaltime[rr, cc] = tentative
            neighbor = (rr, cc)
            openset[neighbor] = tentative + heuristiccostestimate(rp, polar, neighbor, goal)
        end
    end

    # If got here we did not find any route
    verbose && println("No route found after $iterations iterations.")
    return TimedPath(Inf, [start])
end

"""
    validateroutingproblem(rp)

Validate that a `RoutingProblem` has consistent time-slice and obstacle
dimensions, in-bounds endpoints, and unblocked start and finish cells.
Returns `(true, message)` if valid, or `(false, explanation)` otherwise.
"""
function validateroutingproblem(rp::RoutingProblem)
    isempty(rp.timeframe) && return false, "Timeframe is empty"

    refsize = size(rp.timeframe[1])
    for (idx, slice) in enumerate(rp.timeframe)
        if size(slice) != refsize
            return false, "Time slice $idx has different size than first slice"
        end
    end

    if size(rp.obstacleindices) != refsize
        return false, "Obstacle matrix size does not match time slice size"
    end

    if !checkbounds(Bool, rp.timeframe[1], rp.start[1], rp.start[2])
        return false, "Start position is outside grid bounds"
    end
    if !checkbounds(Bool, rp.timeframe[1], rp.finish[1], rp.finish[2])
        return false, "Finish position is outside grid bounds"
    end

    if rp.obstacleindices[rp.start[1], rp.start[2]]
        return false, "Start position is an obstacle"
    end
    if rp.obstacleindices[rp.finish[1], rp.finish[2]]
        return false, "Finish position is an obstacle"
    end

    return true, "Problem is valid"
end

"""
    createroutingproblem(latgrid, longrid, windfunction, currentfunction, timesteps, obstacles, start, finish, timeinterval=10.0)

Create a `RoutingProblem` from latitude and longitude grid coordinates.
`windfunction(lat, lon, timestep)` and `currentfunction(lat, lon, timestep)`
must each return a `(direction, speed)` pair; directions are in degrees and
speeds are in knots. The number of generated time slices is `timesteps`.
`obstacles` is a Boolean grid, and `start` and `finish` are `(row, column)`
indices. `timeinterval` gives the duration in minutes represented by each
slice and defaults to 10 minutes.
"""
function createroutingproblem(
    latgrid,
    longrid,
    windfunction,
    currentfunction,
    timesteps,
    obstacles,
    start,
    finish,
    timeinterval = 10.0,
)
    grid = [
        begin
            w = windfunction(lat, lon, 0)
            c = currentfunction(lat, lon, 0)
            GridPoint(lat, lon, w[1], w[2], c[1], c[2])
        end
        for lat in latgrid, lon in longrid
    ]

    timeslices = [copy(grid) for _ in 1:timesteps]

    for (idx, slice) in enumerate(timeslices)
        for I in eachindex(slice)
            gp = slice[I]
            lat, lon = gp.pt.lat, gp.pt.lon
            w = windfunction(lat, lon, idx)
            c = currentfunction(lat, lon, idx)
            slice[I] = GridPoint(gp.pt, SurfaceParameters(w[1], w[2], c[1], c[2]))
        end
    end

    return RoutingProblem(timeinterval, timeslices, obstacles, start, finish)
end

"""
    solveroutingproblem(sp, rp; verbose = false)
Alias for minimumtimeroute(). Uses A* search to find the minimum time route.
Arguments:
- `sp`: A `SailingPolar` object containing the boat's polar performance data.
- `rp`: A `RoutingProblem` object defining the routing problem, including the grid,
  wind and current conditions, obstacles, start and finish positions, and time interval.
- `verbose`: Named optional boolean flag to enable verbose output during the search.
Returns:
    Tuple of a `TimedPath` object containing the total travel time as well as the
    optimal path found as a vector of `GridPosition` objects ((x, y) tuples).
"""
solveroutingproblem(sp::SailingPolar, rp::RoutingProblem; verbose = false) =
    minimumtimeroute(rp, sp; verbose = verbose)

    
end # module SailingRoutes
