# Data

Generated datasets need no files (`--points uniform:10M`, `clustered:10M:64`,
`scene:10M`). Put real point clouds here; this folder is git-ignored except for
this note. Supported formats (`--points file:PATH`):

| Extension | Layout |
|---|---|
| `.bin` | raw little-endian float32 `x y z` triples (also what `--save-points` writes) |
| `.kitti` | raw float32 `x y z intensity` (KITTI Velodyne scans; rename `.bin` to `.kitti`) |
| `.ply` | ASCII or binary little-endian, vertex `x y z` as float or double |
| `.xyz` `.txt` `.csv` `.pts` | one point per line, first three numbers |

Examples of public point clouds that work well: the Stanford 3D scanning
repository (Lucy, Asian Dragon), KITTI odometry Velodyne scans, Semantic3D.
