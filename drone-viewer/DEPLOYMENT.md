# Deployment

## Quick Deploy (Vercel)

1. Install Vercel CLI: `npm i -g vercel`
2. From this folder: `vercel`
3. Or connect this repo at [vercel.com](https://vercel.com) and deploy.

`vercel.json` is already configured for SPA routing and long-lived caching for `/models/*`.

## Netlify

1. Connect the repo at [netlify.com](https://netlify.com).
2. Build command: `npm run build`
3. Publish directory: `build`
4. Add a redirect: `/* /index.html 200` (SPA).

## Optional: Mesh compression (Draco)

To reduce GLB size before deploy:

```bash
npm install -g gltf-pipeline
gltf-pipeline -i public/models/drone_mesh.glb -o public/models/drone_mesh_compressed.glb --draco.compressionLevel 10
```

Then point the viewer at `/models/drone_mesh_compressed.glb` and ensure the app loads the Draco decoder (e.g. `DRACOLoader` from three.js) when using Draco-compressed GLB.

## Gaussian Splatting

To add a Gaussian Splatting viewer later, place an exported `.splat` (or equivalent) in `public/models/` and use a viewer such as [antimatter15/splat](https://github.com/antimatter15/splat) or [playcanvas/supersplat](https://github.com/playcanvas/supersplat).
