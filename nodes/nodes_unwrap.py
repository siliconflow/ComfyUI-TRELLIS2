"""Modular mesh processing nodes for TRELLIS.2."""
import gc
import os
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .utils import logger
import comfy.model_management


def _log_vram(label):
    import sys
    import torch
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        peak = torch.cuda.max_memory_allocated() / 1024**2
        msg = f"[VRAM] {label}: alloc={alloc:.0f}MB reserved={reserved:.0f}MB peak={peak:.0f}MB"
        logger.info(msg)
        print(msg, file=sys.stderr, flush=True)



class Trellis2Simplify(io.ComfyNode):
    """Simplify mesh to target face count using CuMesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2Simplify",
            display_name="TRELLIS.2 Simplify Mesh",
            category="TRELLIS2",
            description="""Simplify mesh to target face count.

Uses CuMesh for GPU-accelerated simplification.

Parameters:
- target_face_count: Target number of faces
- fill_holes: Fill small holes before simplifying
- fill_holes_perimeter: Max hole perimeter to fill
- remesh: Apply dual-contouring remesh for cleaner topology
- remesh_band: Remesh band width""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("target_face_count", default=500000, min=1000, max=5000000, step=1000),
                io.Boolean.Input("fill_holes", default=True, optional=True),
                io.Float.Input("fill_holes_perimeter", default=0.03, min=0.001, max=0.5, step=0.001, optional=True),
                io.Boolean.Input("remesh", default=False, optional=True),
                io.Float.Input("remesh_band", default=1.0, min=0.1, max=5.0, step=0.1, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        target_face_count=500000,
        fill_holes=True,
        fill_holes_perimeter=0.03,
        remesh=False,
        remesh_band=1.0,
    ):
        import torch
        import cumesh as CuMesh
        import trimesh as Trimesh

        logger.info(f"Simplify: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces -> {target_face_count} target")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        torch.cuda.reset_peak_memory_stats()
        _log_vram("Simplify Start")

        # Convert to torch tensors
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        # Undo coordinate conversion if needed (Z-up back to Y-up)
        vertices_orig = vertices.clone()
        vertices_orig[:, 1], vertices_orig[:, 2] = vertices[:, 2].clone(), -vertices[:, 1].clone()

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_orig, faces)
        logger.info(f"Initial: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
        _log_vram("After CuMesh.init")

        # Fill holes
        if fill_holes:
            cumesh.fill_holes(max_hole_perimeter=fill_holes_perimeter)
            logger.info(f"After fill holes: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            _log_vram("After fill_holes")

        # Optional remesh
        if remesh:
            curr_verts, curr_faces = cumesh.read()
            bvh = CuMesh.cuBVH(curr_verts, curr_faces)
            _log_vram("After BVH build")

            # Estimate grid parameters
            aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device=device)
            center = aabb.mean(dim=0)
            scale = (aabb[1] - aabb[0]).max().item()
            resolution = 512  # Default resolution for remeshing

            cumesh.init(*CuMesh.remeshing.remesh_narrow_band_dc(
                curr_verts, curr_faces,
                center=center,
                scale=(resolution + 3 * remesh_band) / resolution * scale,
                resolution=resolution,
                band=remesh_band,
                project_back=0.0,
                verbose=True,
                bvh=bvh,
            ))
            logger.info(f"After remesh: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
            _log_vram("After DC remesh")
            # Clean up BVH after remesh
            del bvh, curr_verts, curr_faces
            _log_vram("After BVH cleanup")

        # Unify face orientations before simplify
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (pre-simplify)")
        _log_vram("After unify (pre-simplify)")

        # Simplify
        cumesh.simplify(target_face_count, verbose=True)
        logger.info(f"After simplify: {cumesh.num_vertices} vertices, {cumesh.num_faces} faces")
        _log_vram("After simplify")

        # Unify face orientations again after simplify (simplify can break it)
        cumesh.unify_face_orientations()
        logger.info("Unified face orientations (post-simplify)")
        _log_vram("After unify (post-simplify)")

        # Read result
        out_vertices, out_faces = cumesh.read()
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()

        # Convert back to Z-up
        vertices_np[:, 1], vertices_np[:, 2] = -vertices_np[:, 2].copy(), vertices_np[:, 1].copy()

        # Build new trimesh
        result = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            process=False
        )

        logger.info(f"Simplify complete: {len(result.vertices)} vertices, {len(result.faces)} faces")

        # Clean up GPU memory
        del vertices, faces, vertices_orig, out_vertices, out_faces, cumesh
        gc.collect()
        comfy.model_management.soft_empty_cache()
        _log_vram("After cleanup")

        return io.NodeOutput(result)


class Trellis2UVUnwrap(io.ComfyNode):
    """UV unwrap mesh using CuMesh/xatlas. No texture baking."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2UVUnwrap",
            display_name="TRELLIS.2 UV Unwrap",
            category="TRELLIS2",
            description="""UV unwrap mesh using xatlas.

Just creates UVs - no texture baking. Use Rasterize PBR node after this.

Parameters:
- chart_cone_angle: UV chart clustering threshold (degrees)
- chart_refine_iterations: Refine UV charts
- chart_global_iterations: Global UV optimization passes
- chart_smooth_strength: UV smoothing strength

TIP: Simplify mesh first! UV unwrapping 10M faces takes forever.""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("chart_cone_angle", default=90.0, min=0.0, max=359.9, step=1.0, optional=True),
                io.Int.Input("chart_refine_iterations", default=0, min=0, max=10, optional=True),
                io.Int.Input("chart_global_iterations", default=1, min=0, max=10, optional=True),
                io.Int.Input("chart_smooth_strength", default=1, min=0, max=10, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        chart_cone_angle=90.0,
        chart_refine_iterations=0,
        chart_global_iterations=1,
        chart_smooth_strength=1,
    ):
        import torch
        import cumesh as CuMesh
        import trimesh as Trimesh

        logger.info(f"UV Unwrap: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        # Convert to torch
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)

        # Undo coord conversion (Z-up back to Y-up)
        vertices_orig = vertices.clone()
        vertices_orig[:, 1], vertices_orig[:, 2] = vertices[:, 2].clone(), -vertices[:, 1].clone()

        chart_cone_angle_rad = np.radians(chart_cone_angle)

        # Initialize CuMesh
        cumesh = CuMesh.CuMesh()
        cumesh.init(vertices_orig, faces)

        # UV Unwrap
        logger.info("Unwrapping UVs...")
        out_vertices, out_faces, out_uvs, out_vmaps = cumesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": chart_cone_angle_rad,
                "refine_iterations": chart_refine_iterations,
                "global_iterations": chart_global_iterations,
                "smooth_strength": chart_smooth_strength,
            },
            return_vmaps=True,
            verbose=True,
        )

        out_vertices = out_vertices.cpu().numpy()
        out_faces = out_faces.cpu().numpy()
        out_uvs = out_uvs.cpu().numpy()

        # Compute normals
        cumesh.compute_vertex_normals()
        out_normals = cumesh.read_vertex_normals()[out_vmaps.to(device)].cpu().numpy()

        # Convert to Z-up
        out_vertices[:, 1], out_vertices[:, 2] = -out_vertices[:, 2].copy(), out_vertices[:, 1].copy()
        out_normals[:, 1], out_normals[:, 2] = -out_normals[:, 2].copy(), out_normals[:, 1].copy()
        out_uvs[:, 1] = 1 - out_uvs[:, 1]

        # Build trimesh with UVs
        result = Trimesh.Trimesh(
            vertices=out_vertices,
            faces=out_faces,
            vertex_normals=out_normals,
            process=False,
        )
        # Attach UVs as visual
        result.visual = Trimesh.visual.TextureVisuals(uv=out_uvs)

        logger.info(f"UV Unwrap complete: {len(result.vertices)} vertices, {len(result.faces)} faces")

        # Clean up GPU memory
        del vertices, faces, vertices_orig, cumesh
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(result)


class Trellis2RasterizePBR(io.ComfyNode):
    """Rasterize PBR textures from voxel data onto UV-mapped mesh."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2RasterizePBR",
            display_name="TRELLIS.2 Rasterize PBR",
            category="TRELLIS2",
            description="""Bake PBR textures from voxel data onto UV-mapped mesh.

Takes a mesh WITH UVs and bakes color/metallic/roughness from the VOXELGRID.

Input mesh MUST have UVs (use UV Unwrap node first).

Parameters:
- texture_size: Resolution of baked textures (512-16384px)""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Custom("TRELLIS2_VOXELGRID").Input("voxelgrid"),
                io.Int.Input("texture_size", default=2048, min=512, max=16384, step=512),
                io.Boolean.Input("use_pytorch3d", default=False),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="trimesh"),
            ],
        )

    @classmethod
    def execute(
        cls,
        trimesh,
        voxelgrid,
        texture_size=2048,
        use_pytorch3d=False,
    ):
        import torch
        import cv2
        import cumesh as CuMesh
        from flex_gemm.ops.grid_sample import grid_sample_3d
        import trimesh as Trimesh

        # Check for UVs
        if not hasattr(trimesh.visual, 'uv') or trimesh.visual.uv is None:
            raise ValueError("Input mesh has no UVs! Use UV Unwrap node first.")

        # Check for voxel data
        if 'attrs' not in voxelgrid:
            raise ValueError("VoxelGrid has no PBR attributes.")

        logger.info(f"Rasterize PBR: {len(trimesh.vertices)} vertices, texture {texture_size}px")

        comfy.model_management.throw_exception_if_processing_interrupted()

        device = comfy.model_management.get_torch_device()

        # Get mesh data
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32).to(device)
        faces = torch.tensor(trimesh.faces, dtype=torch.int32).to(device)
        uvs = torch.tensor(trimesh.visual.uv, dtype=torch.float32).to(device)

        # Undo Z-up to Y-up for voxel sampling
        vertices_yup = vertices.clone()
        vertices_yup[:, 1], vertices_yup[:, 2] = vertices[:, 2].clone(), -vertices[:, 1].clone()

        # Get voxel data from dict
        attr_volume = voxelgrid['attrs']
        if isinstance(attr_volume, np.ndarray):
            attr_volume = torch.from_numpy(attr_volume)
        attr_volume = attr_volume.to(device)

        coords = voxelgrid['coords']
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)
        coords = coords.to(device)

        voxel_size = voxelgrid['voxel_size']
        attr_layout = voxelgrid['layout']

        orig_vertices = voxelgrid['original_vertices']
        if isinstance(orig_vertices, np.ndarray):
            orig_vertices = torch.from_numpy(orig_vertices)
        orig_vertices = orig_vertices.to(device)

        orig_faces = voxelgrid['original_faces']
        if isinstance(orig_faces, np.ndarray):
            orig_faces = torch.from_numpy(orig_faces)
        orig_faces = orig_faces.to(device)

        # AABB
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device)

        # Grid size
        if voxel_size is not None:
            if isinstance(voxel_size, float):
                voxel_size = torch.tensor([voxel_size] * 3, device=device)
            elif isinstance(voxel_size, (list, tuple, np.ndarray)):
                voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=device)
            grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
        else:
            grid_size = torch.tensor([1024, 1024, 1024], dtype=torch.int32, device=device)
            voxel_size = (aabb[1] - aabb[0]) / grid_size

        # Build BVH from original mesh for accurate attribute lookup
        logger.info("Building BVH...")
        bvh = CuMesh.cuBVH(orig_vertices, orig_faces)

        logger.info("Rasterizing in UV space...")

        if use_pytorch3d:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

            # UV-space vertices: treat UVs as 2D positions with z=0
            verts_uv = torch.cat([uvs * 2 - 1, torch.zeros_like(uvs[:, :1])], dim=-1).float()

            rast_mask = torch.zeros(texture_size, texture_size, dtype=torch.bool, device=device)
            rast_face_ids = torch.full((texture_size, texture_size), -1, dtype=torch.long, device=device)
            rast_bary = torch.zeros(texture_size, texture_size, 3, dtype=torch.float32, device=device)

            # Rasterize in chunks
            chunk_size = 100000
            for i in range(0, faces.shape[0], chunk_size):
                comfy.model_management.throw_exception_if_processing_interrupted()
                chunk_faces = faces[i:i+chunk_size].long()
                meshes = Meshes(verts=[verts_uv], faces=[chunk_faces])
                pix_to_face, _, bary_coords, _ = rasterize_meshes(
                    meshes, image_size=texture_size,
                    faces_per_pixel=1, perspective_correct=False,
                    cull_backfaces=False,
                )
                chunk_hit = pix_to_face[0, :, :, 0] >= 0
                chunk_face_ids = pix_to_face[0, :, :, 0] + i
                rast_face_ids = torch.where(chunk_hit, chunk_face_ids, rast_face_ids)
                rast_bary[chunk_hit] = bary_coords[0, :, :, 0][chunk_hit]
                rast_mask |= chunk_hit
                del meshes, pix_to_face, bary_coords, chunk_hit, chunk_face_ids

            del verts_uv
            comfy.model_management.soft_empty_cache()

            mask = rast_mask

            # Manual barycentric interpolation (replaces dr.interpolate)
            face_verts = vertices_yup[faces[rast_face_ids[mask]].long()]  # [N, 3, 3]
            valid_pos = (face_verts * rast_bary[mask].unsqueeze(-1)).sum(dim=1)
            del rast_face_ids, rast_bary, rast_mask, face_verts
        else:
            import nvdiffrast.torch as dr

            # Setup nvdiffrast
            ctx = dr.RasterizeCudaContext()

            # Prepare UVs for rasterization
            uvs_rast = torch.cat([
                uvs * 2 - 1,
                torch.zeros_like(uvs[:, :1]),
                torch.ones_like(uvs[:, :1])
            ], dim=-1).unsqueeze(0)

            rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)

            # Rasterize in chunks
            chunk_size = 100000
            for i in range(0, faces.shape[0], chunk_size):
                comfy.model_management.throw_exception_if_processing_interrupted()
                rast_chunk, _ = dr.rasterize(
                    ctx, uvs_rast, faces[i:i+chunk_size],
                    resolution=[texture_size, texture_size],
                )
                mask_chunk = rast_chunk[..., 3:4] > 0
                rast_chunk[..., 3:4] += i
                rast = torch.where(mask_chunk, rast_chunk, rast)
                del rast_chunk, mask_chunk

            del ctx, uvs_rast
            comfy.model_management.soft_empty_cache()

            mask = rast[0, ..., 3] > 0

            # Interpolate 3D positions
            pos = dr.interpolate(vertices_yup.unsqueeze(0), rast, faces)[0][0]
            valid_pos = pos[mask]

        # Map to original mesh
        _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
        orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

        # Map vertex positions to original mesh
        logger.info("Mapping vertices to original mesh...")
        _, vert_face_id, vert_uvw = bvh.unsigned_distance(vertices_yup, return_uvw=True)
        vert_orig_tris = orig_vertices[orig_faces[vert_face_id.long()]]
        vertices_mapped = (vert_orig_tris * vert_uvw.unsqueeze(-1)).sum(dim=1)

        del bvh, face_id, uvw, orig_tri_verts, vert_face_id, vert_uvw, vert_orig_tris, vertices_yup
        comfy.model_management.soft_empty_cache()

        # Sample voxel attributes for texture
        logger.info("Sampling voxel attributes...")
        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        # Sample PBR attributes at mapped vertex positions
        logger.info("Sampling vertex PBR attributes...")
        vertex_pbr_attrs = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((vertices_mapped - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )[0]

        logger.info("Building PBR textures...")

        del valid_pos, attr_volume, coords, vertices_mapped
        comfy.model_management.soft_empty_cache()

        mask_np = mask.cpu().numpy()

        # Extract PBR channels
        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        del attrs, mask
        gc.collect()
        comfy.model_management.soft_empty_cache()

        # Inpaint UV seams
        mask_inv = (~mask_np).astype(np.uint8)
        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        # Create PBR material
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([
                np.zeros_like(metallic),
                roughness,
                metallic
            ], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=False,
        )

        # Build result
        result = Trimesh.Trimesh(
            vertices=trimesh.vertices,
            faces=trimesh.faces,
            vertex_normals=trimesh.vertex_normals if hasattr(trimesh, 'vertex_normals') else None,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=trimesh.visual.uv, material=material)
        )

        # Attach PBR vertex attributes
        result.vertex_attributes = {}
        for attr_name, attr_slice in attr_layout.items():
            values = vertex_pbr_attrs[:, attr_slice].clamp(0, 1).cpu().numpy()
            if values.shape[1] == 1:
                result.vertex_attributes[attr_name] = values[:, 0].astype(np.float32)
            else:
                result.vertex_attributes[f'{attr_name}_r'] = values[:, 0].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_g'] = values[:, 1].astype(np.float32)
                result.vertex_attributes[f'{attr_name}_b'] = values[:, 2].astype(np.float32)

        logger.info(f"Rasterize complete: {texture_size}x{texture_size} PBR textures")

        del vertices, faces, uvs, orig_vertices, orig_faces, vertex_pbr_attrs
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(result)


def remesh_narrow_band_dc_lowmem(
    vertices, faces, center, scale, resolution,
    band=1, project_back=0, verbose=False, bvh=None,
    topo_chunk=500_000, tri_chunk=500_000,
    remove_inner_faces=False,
):
    """Low-memory version of cumesh.remeshing.remesh_narrow_band_dc.

    Same algorithm but chunks the topology generation and triangle splitting
    steps to avoid materializing huge intermediate tensors.
    """
    import torch
    from cumesh import _C
    from cumesh.bvh import cuBVH
    from cumesh.remeshing import _init_hashmap
    from tqdm import tqdm

    device = vertices.device

    # --- Constants ---
    edge_neighbor_voxel_offset = torch.tensor([
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ], dtype=torch.int32, device=device).unsqueeze(0)  # (1, 3, 4, 3)

    quad_split_1_n = torch.tensor([0, 1, 2, 0, 2, 3], dtype=torch.long, device=device)
    quad_split_1_p = torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long, device=device)
    quad_split_2_n = torch.tensor([0, 1, 3, 3, 1, 2], dtype=torch.long, device=device)
    quad_split_2_p = torch.tensor([0, 3, 1, 3, 2, 1], dtype=torch.long, device=device)

    OFFSETS = torch.tensor([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
    ], dtype=torch.int32, device=device)

    # --- 1. Build BVH ---
    if bvh is None:
        if verbose:
            print("Building BVH...")
        bvh = cuBVH(vertices, faces)

    eps = band * scale / resolution

    # --- 2. Sparse Grid Construction ---
    base_resolution = resolution
    while base_resolution > 32:
        assert base_resolution % 2 == 0
        base_resolution //= 2

    coords = torch.stack(torch.meshgrid(
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        torch.arange(base_resolution, device=device),
        indexing='ij',
    ), dim=-1).int().reshape(-1, 3)

    pbar = tqdm(
        total=int(torch.log2(torch.tensor(resolution // base_resolution)).item()) + 1,
        desc="Building Sparse Grid", disable=not verbose,
    )

    while True:
        cell_size = scale / base_resolution
        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center
        distances = bvh.unsigned_distance(pts)[0]
        distances -= eps
        distances = torch.abs(distances)
        subdiv_mask = distances < 0.87 * cell_size
        coords = coords[subdiv_mask]
        if base_resolution >= resolution:
            break
        base_resolution *= 2
        coords *= 2
        coords = (coords.unsqueeze(1) + OFFSETS.unsqueeze(0)).reshape(-1, 3)
        pbar.update(1)

    Nvox = coords.shape[0]
    if verbose:
        print(f"Sparse grid: {Nvox:,} voxels")

    # --- 3. Hashmaps + DC vertices ---
    hashmap_vox = _init_hashmap(resolution, 2 * Nvox, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vox,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1),
        resolution, resolution, resolution,
    )

    coords = coords.contiguous()
    grid_verts = _C.get_sparse_voxel_grid_active_vertices(
        *hashmap_vox, coords, resolution, resolution, resolution,
    )
    Nvert = grid_verts.shape[0]

    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center
    distances_vert = bvh.unsigned_distance(pts_vert)[0]
    distances_vert -= eps

    pbar.update(1)
    pbar.close()

    if verbose:
        print("Running Dual Contouring...")

    hashmap_vert = _init_hashmap(resolution + 1, 2 * Nvert, device)
    _C.hashmap_insert_3d_idx_as_val_cuda(
        *hashmap_vert,
        torch.cat([torch.zeros_like(grid_verts[:, :1]), grid_verts], dim=1),
        resolution + 1, resolution + 1, resolution + 1,
    )

    dual_verts, intersected = _C.simple_dual_contour(
        *hashmap_vert, coords, distances_vert,
        resolution + 1, resolution + 1, resolution + 1,
    )

    # Free hashmap_vert — no longer needed
    del hashmap_vert, grid_verts, distances_vert, pts_vert
    torch.cuda.empty_cache()

    # --- 4. Chunked Topology Generation ---
    if verbose:
        print(f"Topology generation (chunked, {topo_chunk:,} voxels/chunk)...")

    R = resolution
    all_quad_indices = []
    all_intersected_dirs = []
    for start in range(0, Nvox, topo_chunk):
        end = min(start + topo_chunk, Nvox)
        c = coords[start:end]
        inter = intersected[start:end]
        chunk_n = c.shape[0]

        # (chunk, 3, 4, 3)
        neighbors = c.reshape(chunk_n, 1, 1, 3) + edge_neighbor_voxel_offset
        mask = inter != 0
        connected = neighbors[mask]  # (M, 4, 3)
        dirs = inter[mask]           # (M,)
        M = connected.shape[0]
        if M == 0:
            del neighbors, connected, dirs
            continue

        hash_key = torch.cat([
            torch.zeros((M * 4, 1), dtype=torch.int, device=device),
            connected.reshape(-1, 3),
        ], dim=1)
        indices = _C.hashmap_lookup_3d_cuda(
            *hashmap_vox, hash_key, R, R, R,
        ).reshape(M, 4).int()
        valid = (indices != 0xffffffff).all(dim=1)
        if valid.any():
            all_quad_indices.append(indices[valid])
            all_intersected_dirs.append(dirs[valid].int())

        del neighbors, connected, dirs, hash_key, indices, valid

    quad_indices = torch.cat(all_quad_indices)
    intersected_dir = torch.cat(all_intersected_dirs)
    del all_quad_indices, all_intersected_dirs, intersected
    L = quad_indices.shape[0]

    if verbose:
        print(f"  {L:,} quads")

    # --- 5. Remove unreferenced vertices ---
    unique_verts = torch.unique(quad_indices.reshape(-1))
    dual_verts = dual_verts[unique_verts]
    vert_map = torch.zeros((Nvox,), dtype=torch.int32, device=device)
    vert_map[unique_verts] = torch.arange(unique_verts.shape[0], dtype=torch.int32, device=device)
    quad_indices = vert_map[quad_indices]
    del vert_map, unique_verts

    mesh_vertices = (dual_verts / resolution - 0.5) * scale + center

    # --- 5b. Quad-level inner face removal ---
    if remove_inner_faces:
        if verbose:
            print(f"Removing inner quads from {quad_indices.shape[0]:,} quads...")
        inner_chunk = 524_288
        quad_centers = torch.empty((quad_indices.shape[0], 3), dtype=torch.float32, device=device)
        for i in range(0, quad_indices.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_indices.shape[0])
            q = quad_indices[i:end].long()
            quad_centers[i:end] = (
                mesh_vertices[q[:, 0]] + mesh_vertices[q[:, 1]] +
                mesh_vertices[q[:, 2]] + mesh_vertices[q[:, 3]]
            ) * 0.25

        sdf = torch.empty(quad_centers.shape[0], dtype=torch.float32, device=device)
        for i in range(0, quad_centers.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_centers.shape[0])
            sdf[i:end] = bvh.signed_distance(quad_centers[i:end], mode='raystab')[0]

        is_outer = sdf >= -eps * 0.1
        n_removed = (~is_outer).sum().item()
        quad_indices = quad_indices[is_outer]
        intersected_dir = intersected_dir[is_outer]

        # Re-index vertices to remove unused
        used = torch.zeros(mesh_vertices.shape[0], dtype=torch.bool, device=device)
        used[quad_indices.flatten()] = True
        new_idx = torch.full((mesh_vertices.shape[0],), -1, dtype=torch.int32, device=device)
        new_idx[used] = torch.arange(used.sum(), dtype=torch.int32, device=device)
        mesh_vertices = mesh_vertices[used]
        for i in range(0, quad_indices.shape[0], inner_chunk):
            end = min(i + inner_chunk, quad_indices.shape[0])
            quad_indices[i:end] = new_idx[quad_indices[i:end]]
        L = quad_indices.shape[0]

        if verbose:
            print(f"  Removed {n_removed:,} inner quads, {L:,} remaining")
        del quad_centers, sdf, is_outer, used, new_idx

    # --- 6. Chunked Triangle Splitting ---
    if verbose:
        print(f"Triangle splitting (chunked, {tri_chunk:,} quads/chunk)...")

    all_triangles = []
    for start in range(0, L, tri_chunk):
        end = min(start + tri_chunk, L)
        qi = quad_indices[start:end]
        idir = intersected_dir[start:end]
        is_pos = (idir == 1).unsqueeze(1)

        # Split 1
        t0 = torch.where(is_pos, qi[:, quad_split_1_p], qi[:, quad_split_1_n])
        n0a = torch.linalg.cross(
            mesh_vertices[t0[:, 1]] - mesh_vertices[t0[:, 0]],
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 0]],
        )
        n0b = torch.linalg.cross(
            mesh_vertices[t0[:, 2]] - mesh_vertices[t0[:, 1]],
            mesh_vertices[t0[:, 3]] - mesh_vertices[t0[:, 1]],
        )
        align0 = (n0a * n0b).sum(dim=1).abs()

        # Split 2
        t1 = torch.where(is_pos, qi[:, quad_split_2_p], qi[:, quad_split_2_n])
        n1a = torch.linalg.cross(
            mesh_vertices[t1[:, 1]] - mesh_vertices[t1[:, 0]],
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 0]],
        )
        n1b = torch.linalg.cross(
            mesh_vertices[t1[:, 2]] - mesh_vertices[t1[:, 1]],
            mesh_vertices[t1[:, 3]] - mesh_vertices[t1[:, 1]],
        )
        align1 = (n1a * n1b).sum(dim=1).abs()

        selected = torch.where((align0 > align1).unsqueeze(1), t0, t1)
        all_triangles.append(selected)
        del qi, idir, t0, t1, n0a, n0b, n1a, n1b, align0, align1, selected

    mesh_triangles = torch.cat(all_triangles).reshape(-1, 3)
    del all_triangles, quad_indices, intersected_dir

    # --- 7. Project back ---
    if project_back > 0:
        if verbose:
            print("Projecting back to original mesh...")
        _, face_id, uvw = bvh.unsigned_distance(mesh_vertices, return_uvw=True)
        orig_tri_verts = vertices[faces[face_id.long()]]
        projected_verts = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        mesh_vertices -= project_back * (mesh_vertices - projected_verts)

    if verbose:
        print(f"  {mesh_vertices.shape[0]:,} vertices, {mesh_triangles.shape[0]:,} faces")

    return mesh_vertices, mesh_triangles.int()


def _batched_unsigned_distance(bvh, positions, batch_size=500_000, return_uvw=False):
    """Batch unsigned_distance queries to avoid GPU kernel timeout on large meshes."""
    import torch
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)
    distances_list, face_id_list, uvw_list = [], [], []
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        d, f, u = bvh.unsigned_distance(positions[i:end], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)
    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None,
    )


class Trellis2ExportGLB(io.ComfyNode):
    """Voxel data -> simplify -> UV unwrap -> bake PBR -> export GLB."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ExportGLB",
            display_name="TRELLIS.2 Export GLB",
            category="TRELLIS2",
            is_output_node=True,
            description="""Textured GLB export from voxel data.

Takes voxel output from "Inference" node and:
1. Optionally remeshes (Dual Contouring)
2. Simplifies to decimation_target faces
3. UV unwraps and bakes PBR textures
4. Exports textured GLB""",
            inputs=[
                io.Custom("TRELLIS2_VOXEL").Input("voxel",
                    tooltip="Voxel data from Inference node."),
                io.Int.Input("decimation_target", default=500000, min=1000, max=5000000, step=1000, optional=True),
                io.Int.Input("texture_size", default=2048, min=512, max=8192, step=512, optional=True),
                io.Boolean.Input("remesh", default=True, optional=True),
                io.Boolean.Input("pre_simplify", default=True, optional=True,
                    tooltip="Pre-simplify mesh before remesh to massively reduce VRAM. May lose very thin features."),
                io.Boolean.Input("use_pytorch3d", default=False, optional=True,
                    tooltip="Use pytorch3d for UV rasterization instead of nvdiffrast."),
                io.Boolean.Input("remove_inner_faces", default=False, optional=True,
                    tooltip="Remove faces whose centers are inside the original mesh (removes internal geometry artifacts)."),
                io.Boolean.Input("double_sided", default=False, optional=True,
                    tooltip="Mark material as double-sided in GLB (renders both front and back faces)."),
                io.String.Input("filename_prefix", default="trellis2", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="glb_path"),
            ],
        )

    @classmethod
    def execute(
        cls,
        voxel,
        decimation_target=500000,
        texture_size=2048,
        remesh=True,
        pre_simplify=True,
        use_pytorch3d=False,
        remove_inner_faces=False,
        double_sided=False,
        filename_prefix="trellis2",
    ):
        import torch
        import cv2
        import cumesh as CuMesh
        from flex_gemm.ops.grid_sample import grid_sample_3d
        import trimesh as Trimesh

        torch.cuda.empty_cache()

        # --- 1. Extract tensors from voxel dict ---
        vertices = voxel['vertices'].float() if torch.is_tensor(voxel['vertices']) else torch.tensor(voxel['vertices']).float()
        faces = voxel['faces'].int() if torch.is_tensor(voxel['faces']) else torch.tensor(voxel['faces']).int()
        coords = voxel['coords'].float() if torch.is_tensor(voxel['coords']) else torch.tensor(voxel['coords']).float()
        attr_volume = voxel['attrs'].float() if torch.is_tensor(voxel['attrs']) else torch.tensor(voxel['attrs']).float()
        voxel_size_f = float(voxel['voxel_size'])
        attr_layout = voxel['layout']

        logger.info(f"ExportGLB: {vertices.shape[0]} verts, {faces.shape[0]} faces, {coords.shape[0]} voxels")

        comfy.model_management.throw_exception_if_processing_interrupted()
        device = comfy.model_management.get_torch_device()

        # Compute grid size from voxel_size
        aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device=device)
        voxel_size = torch.tensor([voxel_size_f] * 3, dtype=torch.float32, device=device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()

        # --- 2. Pre-simplify before remesh ---
        if pre_simplify and remesh and faces.shape[0] > 2_000_000:
            logger.info(f"Pre-simplifying {faces.shape[0]} faces -> 2M before remesh")
            premesh = CuMesh.CuMesh()
            premesh.init(vertices.to(device), faces.to(device))
            premesh.simplify(2_000_000, verbose=True)
            vertices, faces = premesh.read()
            vertices, faces = vertices.cpu(), faces.cpu()
            del premesh
            torch.cuda.empty_cache()
            logger.info(f"Pre-simplified: {vertices.shape[0]} verts, {faces.shape[0]} faces")

        # --- 3. DC remesh ---
        if remesh:
            dc_resolution = grid_size.max().item()
            dc_center = aabb.mean(dim=0)
            dc_scale = (aabb[1] - aabb[0]).max().item()
            remesh_band = 1.0

            logger.info(f"Running low-mem DC: resolution={dc_resolution}")
            new_verts, new_faces = remesh_narrow_band_dc_lowmem(
                vertices.to(device), faces.to(device),
                center=dc_center,
                scale=(dc_resolution + 3 * remesh_band) / dc_resolution * dc_scale,
                resolution=dc_resolution,
                band=remesh_band,
                project_back=0.9,
                verbose=True,
            )
            vertices = new_verts.cpu()
            faces = new_faces.cpu()
            del new_verts, new_faces
            torch.cuda.empty_cache()
            logger.info(f"Remeshed: {vertices.shape[0]} verts, {faces.shape[0]} faces")

        # --- 4. Fill holes + Build BVH from current mesh ---
        vertices = vertices.to(device)
        faces = faces.to(device)

        mesh = CuMesh.CuMesh()
        mesh.init(vertices, faces)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        vertices, faces = mesh.read()
        logger.info(f"After fill holes: {mesh.num_vertices} verts, {mesh.num_faces} faces")

        # --- 5. Build BVH from original (pre-simplify) mesh for accurate attribute lookup ---
        orig_vertices = vertices.clone()
        orig_faces = faces.clone()
        bvh = CuMesh.cuBVH(orig_vertices, orig_faces)

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 6. Simplify + cleanup ---
        if not remesh:
            # No-remesh: aggressive simplify → cleanup → final simplify → cleanup
            mesh.simplify(decimation_target * 3, verbose=True)
            logger.info(f"After initial simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.simplify(decimation_target, verbose=True)
            logger.info(f"After final simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")
            mesh.remove_duplicate_faces()
            mesh.repair_non_manifold_edges()
            mesh.remove_small_connected_components(1e-5)
            mesh.fill_holes(max_hole_perimeter=3e-2)
            mesh.unify_face_orientations()
        else:
            # Remesh: just simplify (DC already cleaned topology)
            mesh.simplify(decimation_target, verbose=True)
            logger.info(f"After simplify: {mesh.num_vertices} verts, {mesh.num_faces} faces")

        # --- 7. Remove inner faces (optional) ---
        if remove_inner_faces:
            logger.info("Removing inner faces...")
            out_v, out_f = mesh.read()
            face_centers = (out_v[out_f[:, 0].long()] + out_v[out_f[:, 1].long()] + out_v[out_f[:, 2].long()]) / 3.0
            chunk_size = 524_288
            signed_dists = torch.empty(face_centers.shape[0], dtype=torch.float32, device=device)
            for ci in range(0, face_centers.shape[0], chunk_size):
                end = min(ci + chunk_size, face_centers.shape[0])
                signed_dists[ci:end] = bvh.signed_distance(face_centers[ci:end], mode='raystab')[0]
            eps = (aabb[1] - aabb[0]).max().item() / grid_size.max().item()
            is_outer = signed_dists >= -eps * 0.1
            n_removed = (~is_outer).sum().item()
            if n_removed > 0:
                mesh.remove_faces(~is_outer)
                mesh.remove_unreferenced_vertices()
            logger.info(f"Removed {n_removed} inner faces, {mesh.num_faces} remaining")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 8. UV unwrap ---
        logger.info("UV unwrapping...")
        out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
            compute_charts_kwargs={
                "threshold_cone_half_angle_rad": np.radians(90.0),
                "refine_iterations": 0,
                "global_iterations": 1,
                "smooth_strength": 1,
            },
            return_vmaps=True,
            verbose=True,
        )
        out_vertices = out_vertices.to(device)
        out_faces = out_faces.to(device)
        out_uvs = out_uvs.to(device)
        out_vmaps = out_vmaps.to(device)
        mesh.compute_vertex_normals()
        out_normals = mesh.read_vertex_normals()[out_vmaps]
        logger.info(f"UV unwrap done: {out_vertices.shape[0]} verts, {out_faces.shape[0]} faces")

        comfy.model_management.throw_exception_if_processing_interrupted()

        # --- 9. Rasterize in UV space ---
        logger.info("Rasterizing UV space...")
        if use_pytorch3d:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes

            verts_uv = torch.cat([out_uvs * 2 - 1, torch.ones_like(out_uvs[:, :1])], dim=-1).float()
            rast_mask = torch.zeros(texture_size, texture_size, dtype=torch.bool, device=device)
            rast_face_ids = torch.full((texture_size, texture_size), -1, dtype=torch.long, device=device)
            rast_bary = torch.zeros(texture_size, texture_size, 3, dtype=torch.float32, device=device)

            chunk_size = 100_000
            for i in range(0, out_faces.shape[0], chunk_size):
                comfy.model_management.throw_exception_if_processing_interrupted()
                chunk_faces = out_faces[i:i+chunk_size].long()
                meshes = Meshes(verts=[verts_uv], faces=[chunk_faces])
                pix_to_face, _, bary_coords, _ = rasterize_meshes(
                    meshes, image_size=texture_size,
                    faces_per_pixel=1, perspective_correct=False,
                    cull_backfaces=False, bin_size=0,
                )
                chunk_hit = pix_to_face[0, :, :, 0] >= 0
                chunk_face_ids = pix_to_face[0, :, :, 0] + i
                rast_face_ids = torch.where(chunk_hit, chunk_face_ids, rast_face_ids)
                rast_bary[chunk_hit] = bary_coords[0, :, :, 0][chunk_hit]
                rast_mask |= chunk_hit
                del meshes, pix_to_face, bary_coords, chunk_hit, chunk_face_ids

            del verts_uv
            torch.cuda.synchronize()

            # pytorch3d pixel order is flipped on both axes vs nvdiffrast
            rast_mask = rast_mask.flip(0).flip(1)
            rast_face_ids = rast_face_ids.flip(0).flip(1)
            rast_bary = rast_bary.flip(0).flip(1)

            mask = rast_mask
            face_verts = out_vertices[out_faces[rast_face_ids[mask]].long()]
            valid_pos = (face_verts * rast_bary[mask].unsqueeze(-1)).sum(dim=1)
            del rast_face_ids, rast_bary, rast_mask, face_verts
        else:
            import nvdiffrast.torch as dr

            ctx = dr.RasterizeCudaContext()
            uvs_rast = torch.cat([
                out_uvs * 2 - 1,
                torch.zeros_like(out_uvs[:, :1]),
                torch.ones_like(out_uvs[:, :1]),
            ], dim=-1).unsqueeze(0)
            rast = torch.zeros((1, texture_size, texture_size, 4), device=device, dtype=torch.float32)

            chunk_size = 100_000
            for i in range(0, out_faces.shape[0], chunk_size):
                comfy.model_management.throw_exception_if_processing_interrupted()
                rast_chunk, _ = dr.rasterize(
                    ctx, uvs_rast, out_faces[i:i+chunk_size],
                    resolution=[texture_size, texture_size],
                )
                mask_chunk = rast_chunk[..., 3:4] > 0
                rast_chunk[..., 3:4] += i
                rast = torch.where(mask_chunk, rast_chunk, rast)
                del rast_chunk, mask_chunk

            del ctx, uvs_rast

            mask = rast[0, ..., 3] > 0
            pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
            valid_pos = pos[mask]
            del rast, pos

        comfy.model_management.soft_empty_cache()

        # --- 10. BVH map to original surface ---
        logger.info("Mapping to original surface...")
        _, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)
        orig_tri_verts = orig_vertices[orig_faces[face_id.long()]]
        valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)
        del face_id, uvw, orig_tri_verts

        # --- 11. Sample voxel attributes ---
        logger.info("Sampling PBR attributes...")
        coords = coords.to(device)
        attr_volume = attr_volume.to(device)

        attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device=device)
        attrs[mask] = grid_sample_3d(
            attr_volume,
            torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
            shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
            grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
            mode='trilinear',
        )

        del valid_pos, bvh, orig_vertices, orig_faces, attr_volume, coords
        comfy.model_management.soft_empty_cache()

        # --- 12. Inpaint UV seams ---
        logger.info("Inpainting UV seams...")
        mask_np = mask.cpu().numpy()
        mask_inv = (~mask_np).astype(np.uint8)

        base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

        del attrs, mask
        gc.collect()
        comfy.model_management.soft_empty_cache()

        base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
        metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
        alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

        # --- 13. Build PBR material ---
        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
            baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
            metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
            metallicFactor=1.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
            doubleSided=double_sided,
        )

        # --- 14. Coordinate conversion Y-up → Z-up ---
        vertices_np = out_vertices.cpu().numpy()
        faces_np = out_faces.cpu().numpy()
        uvs_np = out_uvs.cpu().numpy()
        normals_np = out_normals.cpu().numpy()

        vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2].copy(), -vertices_np[:, 1].copy()
        normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2].copy(), -normals_np[:, 1].copy()
        uvs_np[:, 1] = 1 - uvs_np[:, 1]

        # --- 15. Assemble trimesh ---
        textured_mesh = Trimesh.Trimesh(
            vertices=vertices_np,
            faces=faces_np,
            vertex_normals=normals_np,
            process=False,
            visual=Trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
        )

        # --- 16. Export GLB ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.glb"
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, filename)

        textured_mesh.export(output_path, file_type='glb')
        logger.info(f"GLB exported: {output_path}")

        del textured_mesh, out_vertices, out_faces, out_uvs, out_normals, mesh
        gc.collect()
        comfy.model_management.soft_empty_cache()

        return io.NodeOutput(output_path)


class Trellis2ExportTrimesh(io.ComfyNode):
    """Export trimesh to file (GLB, OBJ, PLY, etc.).

    Note: This is NOT isolated because it's just disk I/O.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2ExportTrimesh",
            display_name="TRELLIS.2 Export Trimesh",
            category="TRELLIS2",
            is_output_node=True,
            description="""Export trimesh to various 3D file formats.

Supports: GLB, OBJ, PLY, STL, 3MF, DAE""",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.String.Input("filename_prefix", default="trellis2", optional=True),
                io.Combo.Input("file_format", options=["glb", "obj", "ply", "stl", "3mf", "dae"],
                               default="glb", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="file_path"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, filename_prefix="trellis2", file_format="glb"):
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.{file_format}"

        output_dir = folder_paths.get_output_directory()
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(exist_ok=True)

        trimesh.export(str(output_path), file_type=file_format)

        logger.info(f"Exported to: {output_path}")

        return io.NodeOutput(str(output_path))


NODE_CLASS_MAPPINGS = {
    "Trellis2Simplify": Trellis2Simplify,
    "Trellis2UVUnwrap": Trellis2UVUnwrap,
    "Trellis2RasterizePBR": Trellis2RasterizePBR,
    "Trellis2ExportGLB": Trellis2ExportGLB,
    "Trellis2ExportTrimesh": Trellis2ExportTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Simplify": "TRELLIS.2 Simplify Mesh",
    "Trellis2UVUnwrap": "TRELLIS.2 UV Unwrap",
    "Trellis2RasterizePBR": "TRELLIS.2 Rasterize PBR",
    "Trellis2ExportGLB": "TRELLIS.2 Export GLB",
    "Trellis2ExportTrimesh": "TRELLIS.2 Export Trimesh",
}
