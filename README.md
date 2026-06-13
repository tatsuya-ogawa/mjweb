# MJLab Web Play

Browser-based MuJoCo locomotion playground for Unitree G1 and Go1 policies. The app runs MuJoCo WASM, ONNX Runtime Web, Three.js rendering, and Gaussian-splat-derived height fields directly in the browser.

## Features

- MuJoCo WASM simulation with ONNX policy inference in the browser
- Unitree G1 and Go1 walk environments on flat and rough terrain
- Go1 Gaussian support terrain generated from bundled or local Gaussian sources
- Navigation mode with goal picking, route preview, route following, and adjustable path-planning limits
- Manual command controls for forward, lateral, and yaw velocity
- Local-only custom Gaussian loading for `.sog`, `.spz`, `.splat`, and `.ply` files

## Requirements

- Node.js 22 or newer is recommended
- npm
- uv, for regenerating MuJoCo scene assets
- A browser with WebAssembly and WebGL/WebGPU support

The Vite dev server sets `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers required by the runtime. Use the project scripts instead of opening `index.html` directly.

## Quick Start

```bash
npm install
npm run prepare:assets
npm run dev
```

Open the printed Vite URL, usually `http://localhost:5173/`.

For a production build:

```bash
npm run build
npm run preview
```

## Controls

- `Control`: drive the robot manually with the D-pad and velocity sliders.
- `Navigation`: build route maps, pick goals, preview paths, and follow a planned route.
- `Pick Spawn`: place the robot spawn point on supported terrain.
- `Pause`: pause or resume the simulation.
- `Reset`: reset the current environment to its initial state.
- `Contacts`, `Meshes`, `Skeleton`: toggle debug and rendering overlays.

## Gaussian Terrain

The `Unitree Go1 Gaussian Support` environment can use bundled Gaussian sources or a local custom source.

- `Terrain Preset`: choose a bundled Gaussian preset.
- `Load Custom Gaussian`: use a local `.sog`, `.spz`, `.splat`, or `.ply` source for the selected preset.
- `Clear Custom Gaussian`: remove the loaded custom source and return to the bundled source.
- `Source Scale`: adjust source-to-world scale before regenerating the height field.

Custom Gaussian files are read by the browser from your local machine. The app does not upload them to a server.

## Optional Garden Demo

The Garden demo source is large and intentionally ignored by Git. To download it locally:

```bash
npm run setup:garden
npm run dev:local-gaussian
```

The script places the source under `public/envs/go1_gaussian/splats/garden/` and writes the transform config needed by the app.

## Project Scripts

- `npm run prepare:assets`: clone or reuse `mjlab`, restore source mesh assets for tracked scenes, regenerate optimized render assets, and regenerate the Gaussian source manifest.
- `npm run prepare:assets:ci`: run the same preparation for CI and prune source STL assets from `public/` before Vite copies files to `dist/`.
- `npm run setup:garden`: download the optional Garden Gaussian demo source and write its transform config.
- `npm run dev`: generate the Gaussian source manifest and start Vite.
- `npm run dev:local-gaussian`: include ignored local Gaussian files, then start Vite.
- `npm run build`: type-check and build the production bundle.
- `npm run build:local-gaussian`: include ignored local Gaussian files, then build.
- `npm run preview`: serve the production build locally.
- `npm run security:audit`: run `npm audit --audit-level=moderate`.
- `npm run verify:render`: verify MuJoCo/render mesh mapping.
- `npm run optimize:render-assets`: regenerate optimized render assets.

## Repository Hygiene

Generated and local-only paths are ignored, including:

- `node_modules/`
- `dist/`
- `src/generated/`
- optimized render outputs under `public/render_assets/`
- regenerated scene source assets under `public/envs/*/assets/`
- regenerated render manifests and optimized scene XML files
- `.env` and `.env.*`
- Python virtual environments such as `.venv/`
- local MJLab/MuJoCo checkouts
- large optional Gaussian splats outside explicitly bundled presets

Before publishing or cutting a release:

```bash
npm run prepare:assets
npm run build
npm run security:audit
git status --short
```

GitHub Actions uses `npm run prepare:assets:ci` before building. If `tatsuya-ogawa/mjlab` is private for a runner, set the repository secret `MJLAB_REPO_TOKEN` with read access to that repository.

## Security Notes

- The app is a static browser app; it does not include a backend service.
- Custom Gaussian files stay local to the browser session.
- Do not commit `.env` files, local model exports, or downloaded third-party datasets.
- Dependency audit currently passes with `npm audit --audit-level=moderate`.

## License

No project license has been selected yet. Choose and add a `LICENSE` file before accepting external contributions or relying on redistribution terms.
