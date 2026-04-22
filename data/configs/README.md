# Scene Configuration Files

This directory contains the JSON scene files used by the `instantvnr` apps
(`vnr_int_dual`, `vnr_int_single`, `vnr_cmd_train`, `vnr_cmd_render`, ...).
A scene file describes **where the raw volume lives on disk**, **how to
interpret the bytes**, and **how to render it** (camera, transfer function,
isosurfaces, lighting, ...).

Scene files are parsed by
`base/ovr/serializer/serializer_json.cpp`
(`ovr::scene::create_json_scene`). Any field not listed below is either
ignored or consumed by legacy tooling; the sections marked *optional* reflect
what the current loader actually reads.

> JSON with `//` comments is accepted (the loader enables lenient parsing),
> so you can comment out fields without breaking the file.

## Top-Level Layout

```json
{
  "dataSource" : [ /* one or more volume entries */ ],
  "snapshot"   : [ /* unused, keep empty */ ],
  "view"       : { /* camera, volume, transfer function, ... */ }
}
```

| Field        | Required | Description                                                                                     |
| ------------ | :------: | ----------------------------------------------------------------------------------------------- |
| `dataSource` |    yes   | Array of volumes referenced by the scene. Each entry gets an integer `id` starting at 1.        |
| `view`       |    yes   | Rendering description (camera, lights, main volume, transfer function, optional isosurfaces).   |
| `snapshot`   |    no    | Legacy field retained for forward-compatibility; always an empty array in the provided configs. |
| `cache`      |    no    | Legacy NN-cache / out-of-core streaming config. Ignored by the current JSON scene loader.       |
| `macrocell`  |    no    | Legacy top-level macrocell path/hints. Ignored by the current JSON scene loader.                |

## `dataSource[]`

Each entry describes one 3D scalar field stored as a raw binary file.

```json
{
  "id"            : 1,
  "name"          : "vorts1.data",
  "format"        : "REGULAR_GRID_RAW_BINARY",
  "fileName"      : [
    "data/data/vorts1.data",
    "/media/data/volume/vorts1.data"
  ],
  "dimensions"    : { "x": 128, "y": 128, "z": 128 },
  "type"          : "FLOAT",
  "offset"        : 0,
  "endian"        : "LITTLE_ENDIAN",
  "fileUpperLeft" : false
}
```

| Field           | Required | Default          | Description                                                                                                                                       |
| --------------- | :------: | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`        |    yes   | —                | Only `REGULAR_GRID_RAW_BINARY` is currently implemented.                                                                                          |
| `fileName`      |    yes   | —                | Path to the raw file. May be a **single string** or an **array of candidate paths**: the loader picks the first path that exists (handy for sharing configs across machines). Relative paths are resolved against the directory containing the scene JSON. |
| `dimensions`    |    yes   | —                | Voxel counts along `x`, `y`, `z`.                                                                                                                 |
| `type`          |    yes   | —                | Scalar value type. One of `BYTE`, `UNSIGNED_BYTE`, `SHORT`, `UNSIGNED_SHORT`, `INT`, `UNSIGNED_INT`, `FLOAT`, `DOUBLE`.                           |
| `offset`        |    no    | `0`              | Byte offset to the first voxel in the file (useful when reading a single field out of a packed multi-variable dump).                              |
| `endian`        |    no    | `LITTLE_ENDIAN`  | `LITTLE_ENDIAN` or `BIG_ENDIAN`.                                                                                                                  |
| `fileUpperLeft` |    no    | `false`          | If `true`, the Y axis is flipped on load (for data authored with origin at the upper-left of each slice).                                         |
| `scales`        |    no    | `{1, 1, 1}`      | Anisotropic voxel spacing, e.g. `{ "x": 1.0, "y": 1.0, "z": 2.5 }`. Applied as `grid_spacing` on the structured-regular volume.                    |
| `id`            |    no    | array index      | Stable identifier used by `view.volume.dataId`, `view.isosurfaces.dataId`, etc. Starts at `1`.                                                    |
| `name`          |    no    | —                | Human-readable label; not used by the loader.                                                                                                     |

## `view`

The `view` block controls the camera, the main volume model, optional
geometric models (isosurfaces, spheres), and lighting.

### Camera (`view.camera`)

```json
"camera": {
  "eye"    : { "x": 63.5, "y": 63.5, "z": 400 },
  "center" : { "x": 63.5, "y": 63.5, "z": 63.5 },
  "up"     : { "x": 0,    "y": 1,    "z": 0   },
  "fovy"   : 45,
  "projectionMode": "PERSPECTIVE"
}
```

Only `eye`, `center`, `up`, and `fovy` are read. `fovy` is in degrees.
`projectionMode`, `zNear`, and `zFar` are ignored by the current loader.

### Lighting

```json
"lightSource": {
  "type"     : "DIRECTIONAL_LIGHT",
  "position" : { "x": 0, "y": 0, "z": 1, "w": 0 },
  "diffuse"  : { "r": 1, "g": 1, "b": 1, "a": 1 }
}
```

- `lightSource` is a single directional light. `position` is interpreted as
  the light **direction** (xyz), `diffuse` as the light color (rgb).
- `additionalLightSources` (optional array) follows the same schema and
  appends to the scene's light list.
- If neither field is present, a default white directional light pointing at
  `(1, 1, 1)` is inserted automatically.

Fields such as `ambient`, `specular`, `globalAmbient`, `globalLighting`,
`lighting`, and `lightingSide` are retained for compatibility but are not
consumed by the current loader.

### Main volume (`view.volume`)

This block selects one of the `dataSource` entries and describes how to
sample and shade it.

```json
"volume": {
  "dataId"          : 1,
  "visible"         : true,
  "sampleDistance"  : 0.25,
  "scalarMappingRange"             : { "minimum": 0.0, "maximum": 12.14 },
  "scalarMappingRangeUnnormalized" : { "minimum": 0,   "maximum": 3276  },
  "transferFunction": { /* see below */ }
}
```

| Field                            | Required | Description                                                                                                                                                                                                                                                                                                            |
| -------------------------------- | :------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataId`                         |    no    | Selects the backing volume from `dataSource`. Defaults to the first entry.                                                                                                                                                                                                                                             |
| `visible`                        |    no    | If `false`, the main volumetric model is not added to the scene (you can still render isosurfaces or spheres off the same data). Defaults to `true`.                                                                                                                                                                   |
| `sampleDistance`                 |    yes   | Step size used to derive the volume sampling rate (`volume_sampling_rate = 1 / sampleDistance`). Typical value: `0.25`.                                                                                                                                                                                                |
| `scalarMappingRange`             | see note | Value range the transfer function is defined over, **in the OpenGL-normalized domain** (`[0, 1]` for floats, `[0, max_int]` scaled for integer types). The loader rescales it for integer volumes automatically.                                                                                                       |
| `scalarMappingRangeUnnormalized` | see note | Value range **already in the native units of the data**. Preferred for integer volumes, since it skips the implicit OpenGL normalization. If both are present, `scalarMappingRangeUnnormalized` wins.                                                                                                                  |

> **Note:** at least one of `scalarMappingRange` /
> `scalarMappingRangeUnnormalized` should be provided. Without them the
> loader falls back to the full float range and prints a warning for
> integer-typed volumes; results are likely to look wrong.

#### `view.volume.transferFunction`

```json
"transferFunction": {
  "resolution"    : 1024,
  "alphaArray"    : { "data": "<base64 of N float32 alpha samples>", "encoding": "BASE64" },
  "colorControls" : [
    { "position": 0.0, "color": { "r": 0, "g": 0.36, "b": 1 } },
    { "position": 1.0, "color": { "r": 1, "g": 0,    "b": 1 } }
  ],
  "gaussianObjects": [
    { "mean": 0.46, "sigma": 0.012, "heightFactor": 0.007 }
  ]
}
```

- `resolution` — number of samples along the 1D transfer function (typically
  `1024`).
- `alphaArray.data` — base64-encoded little-endian `float32` array of length
  `resolution`. Each entry is the opacity at the corresponding position in
  `[0, 1]`. The terminal samples (`alpha[0]`, `alpha[resolution-1]`) are
  clamped to `0` when they fall below `0.01` to prevent leaking color into
  transparent regions.
- `colorControls[]` — list of RGB color stops at normalized positions in
  `[0, 1]`. The loader samples this into the transfer function texture.
- `gaussianObjects[]` (optional) — compact representation of Gaussian alpha
  bumps. Used when the TF was authored in an editor that supports them; they
  are already baked into `alphaArray` and are kept for round-tripping.

The easiest way to produce this block is to let one of the GUI apps
(`vnr_int_dual`, `vnr_int_single`) save a scene after you edit the transfer
function.

### Isosurfaces (`view.isosurfaces`, optional)

```json
"isosurfaces": {
  "dataId"    : 1,
  "visible"   : true,
  "isovalues" : [ 5.0, 8.0 ],
  "aoSamples" : 32,
  "transferFunctionDataId" : 1,
  "scalarMappingRange"     : { "minimum": -0.11, "maximum": 0.096 },
  "transferFunction"       : { /* same schema as view.volume.transferFunction */ }
}
```

| Field                    | Required | Description                                                                                                                                    |
| ------------------------ | :------: | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `isovalues`              |    yes   | Array of scalar iso-values at which to extract surfaces.                                                                                       |
| `dataId`                 |    no    | Volume the iso-surface is extracted from. Defaults to the first data source.                                                                   |
| `visible`                |    no    | If `false`, the iso-surface geometry is parsed but not added to the scene. Defaults to `true`.                                                 |
| `aoSamples`              |    no    | Number of ambient-occlusion samples used when shading the iso-surface. Defaults to `0` (AO disabled).                                          |
| `transferFunction`       |    no    | If provided, the iso-surface is colored by a volume-mapped transfer function; otherwise it reuses the main volume's transfer function.         |
| `transferFunctionDataId` |    no    | Volume sampled by the iso-surface's color transfer function (defaults to `dataId`). Lets you color surface A by variable B in multi-field data.|

### Spheres (`view.spheres`, optional)

```json
"spheres": {
  "visible"   : true,
  "positions" : [
    { "x": 0.5, "y": 127.5, "z": 0.5, "radius": 1.0 }
  ]
}
```

A small point primitive used mostly in debug/test scenes. Each entry is
converted into a sphere with unit radius (the `radius` key is reserved for
future use).

## Fields Retained for Compatibility

The following keys appear in the provided configs but are **not consumed by
the current scene loader**. They are left in place so that the files remain
compatible with the legacy VIDI/OVR tools that authored them:

`backgroundColor`, `globalAmbient`, `globalLighting`, `lighting`,
`lightingSide`, `method`, `tfPreIntegration`, `projectionMode`, `zNear`,
`zFar`, `boundingBox`, `boundingBoxColor`, `boundingBoxVisible`,
`clippingBox`, `textureBox`, `interpolationType`, `material`,
`opacityUnitDistance`, `slice`, `transferFunctionType`,
`lightSource.ambient`, `lightSource.specular`, top-level `cache`,
top-level `macrocell`.

Feel free to strip them when writing a new config from scratch — a minimal
config only needs `dataSource`, `view.camera`, and
`view.volume.transferFunction` (plus a scalar range).

## Minimal Example

The smallest config that the loader accepts looks like this:

```json
{
  "dataSource": [
    {
      "format"     : "REGULAR_GRID_RAW_BINARY",
      "fileName"   : "data/data/vorts1.data",
      "dimensions" : { "x": 128, "y": 128, "z": 128 },
      "type"       : "FLOAT"
    }
  ],
  "view": {
    "camera": {
      "eye"    : { "x": 63.5, "y": 63.5, "z": 400 },
      "center" : { "x": 63.5, "y": 63.5, "z": 63.5 },
      "up"     : { "x": 0,    "y": 1,    "z": 0   },
      "fovy"   : 45
    },
    "volume": {
      "dataId"             : 1,
      "sampleDistance"     : 0.25,
      "scalarMappingRange" : { "minimum": 0.0, "maximum": 12.14 },
      "transferFunction"   : {
        "resolution"   : 1024,
        "alphaArray"   : { "encoding": "BASE64", "data": "..." },
        "colorControls": [
          { "position": 0.0, "color": { "r": 0, "g": 0, "b": 1 } },
          { "position": 1.0, "color": { "r": 1, "g": 0, "b": 0 } }
        ]
      }
    }
  }
}
```

For richer, ready-to-run examples see the other `scene_*.json` files in this
directory — `scene_vorts1.json` is the canonical small test volume,
`scene_fullbody_ct.json` exercises integer data with
`scalarMappingRangeUnnormalized`, and `scene_supernova.json` demonstrates
gaussian-based transfer functions.
