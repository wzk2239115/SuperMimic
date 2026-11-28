"""Render original BVH motion beside a saved Q1 rollout, without simulating."""
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path

os.environ["MUJOCO_GL"] = "disable"
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def read_bvh(path):
    hierarchy, motion = re.split(r'(?m)^\s*MOTION\s*$', path.read_text(), maxsplit=1)
    tokens = iter(re.findall(r'[^\s{}]+|[{}]', hierarchy))
    if next(tokens).upper() != 'HIERARCHY' or next(tokens).upper() != 'ROOT':
        raise ValueError('Invalid BVH hierarchy')
    joints = []

    def visit(name, parent):
        index = len(joints)
        joints.append(None)
        if next(tokens) != '{' or next(tokens) != 'OFFSET':
            raise ValueError('Invalid BVH joint')
        offset = np.array([float(next(tokens)) for _ in range(3)])
        token = next(tokens)
        channels = []
        if token == 'CHANNELS':
            channels = [next(tokens) for _ in range(int(next(tokens)))]
            token = next(tokens)
        joints[index] = name, parent, offset, channels
        while token != '}':
            if token == 'JOINT':
                visit(next(tokens), index)
            elif token == 'End' and next(tokens) == 'Site':
                visit(name + '_tip', index)
            else:
                raise ValueError(f'Unsupported BVH token {token}')
            token = next(tokens)

    visit(next(tokens), -1)
    header = re.search(r'Frames:\s*(\d+)\s+Frame Time:\s*([\d.eE+-]+)', motion)
    if header is None:
        raise ValueError('Missing BVH frame metadata')
    count, dt = int(header[1]), float(header[2])
    values = np.fromstring(motion[header.end():], sep=' ').reshape(count, -1)
    positions = np.empty((count, len(joints), 3))
    rotations = []
    column = 0
    for j, (_, parent, offset, channels) in enumerate(joints):
        translation = np.broadcast_to(offset, (count, 3)).copy()
        axes, angles = [], []
        for channel in channels:
            if channel.endswith('position'):
                translation[:, 'XYZ'.index(channel[0])] += values[:, column]
            elif channel.endswith('rotation'):
                axes.append(channel[0])
                angles.append(values[:, column])
            else:
                raise ValueError(f'Unsupported BVH channel {channel}')
            column += 1
        local = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
        for axis, degrees in zip(axes, angles, strict=True):
            axis_index = 'XYZ'.index(axis)
            first, second = (axis_index + 1) % 3, (axis_index + 2) % 3
            radians = np.deg2rad(degrees)
            component = np.broadcast_to(np.eye(3), local.shape).copy()
            component[:, first, first] = component[:, second, second] = np.cos(radians)
            component[:, second, first] = np.sin(radians)
            component[:, first, second] = -np.sin(radians)
            local = local @ component
        if parent == -1:
            positions[:, j] = translation
            rotations.append(local)
        else:
            positions[:, j] = positions[:, parent] + np.einsum(
                'tij,tj->ti', rotations[parent], translation
            )
            rotations.append(rotations[parent] @ local)
    if column != values.shape[1]:
        raise ValueError('BVH channel count mismatch')
    # LaFAN1 BVH uses centimeters and Y up. Apply a right-handed change of axes.
    positions = positions[:, :, [0, 2, 1]] * 0.01
    positions[:, :, 1] *= -1
    feet = [i for i, (name, *_rest) in enumerate(joints)
            if 'foot' in name.lower() or 'toe' in name.lower()]
    positions[:, :, 2] -= np.percentile(positions[:, feet or [0], 2], 2)
    return joints, positions, dt


def project(points, center, scale, width, height):
    azimuth, elevation = np.deg2rad([125, 12])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    front = np.array([np.cos(azimuth), np.sin(azimuth), 0])
    up = -np.sin(elevation) * front + np.array([0, 0, np.cos(elevation)])
    depth = np.cos(elevation) * front + np.array([0, 0, np.sin(elevation)])
    camera = (points - center) @ np.stack([right, -up, depth], axis=1)
    camera *= scale
    camera[..., :2] += [width / 2, height / 2]
    return camera


def panel(width, height):
    image = Image.new('RGB', (width, height), '#243241')
    draw = ImageDraw.Draw(image)
    return image, draw


def ground(draw, center, scale, width, height):
    for offset in np.arange(-3, 3.1, 0.5):
        x, y = center[:2]
        for a, b in (((x + offset, y - 3, 0), (x + offset, y + 3, 0)),
                     ((x - 3, y + offset, 0), (x + 3, y + offset, 0))):
            screen = project(np.array([a, b]), center, scale, width, height)
            draw.line([tuple(point[:2]) for point in screen], fill='#3d4e5f')


def human_image(joints, points, width, height):
    image, draw = panel(width, height)
    center = np.array([*points[0, :2], 0.9])
    scale = 150
    screen = project(points, center, scale, width, height)
    ground(draw, center, scale, width, height)
    limbs = []
    for index, (name, parent, *_rest) in enumerate(joints):
        if parent >= 0:
            limbs.append((name.lower(), screen[parent], screen[index]))
    for name, a, b in sorted(limbs, key=lambda limb: (limb[1][2] + limb[2][2])):
        if 'head' in name:
            radius, color = 0.085, '#e0a780'
        elif 'spine' in name or 'neck' in name:
            radius, color = 0.115, '#58a2d9'
        elif any(part in name for part in ('arm', 'hand', 'shoulder')):
            radius, color = 0.042, '#e0a780'
        else:
            radius, color = 0.054, '#849eb8'
        r = round(radius * scale)
        draw.line([tuple(a[:2]), tuple(b[:2])], fill=color, width=2 * r)
        for x, y in (a[:2], b[:2]):
            draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    return image


def robot_meshes(model):
    meshes = []
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if model.geom_group[geom] != 1:
            continue
        mesh = model.geom_dataid[geom]
        va, fa = model.mesh_vertadr[mesh], model.mesh_faceadr[mesh]
        vertices = model.mesh_vert[va:va + model.mesh_vertnum[mesh]]
        faces = model.mesh_face[fa:fa + model.mesh_facenum[mesh]]
        # Merge nearby vertices once to keep the CPU renderer fast.
        lower = vertices.min(axis=0)
        cell = max(float(np.ptp(vertices, axis=0).max()) / 12, 1e-6)
        _, inverse = np.unique(np.floor((vertices - lower) / cell),
                               axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        counts = np.bincount(inverse)
        reduced = np.stack([
            np.bincount(inverse, weights=axis) / counts for axis in vertices.T
        ], axis=1)
        faces = inverse[faces]
        valid = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
        faces = faces[valid]
        _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        meshes.append((geom, reduced, faces[unique]))
    if not meshes:
        raise ValueError('No Q1 visual meshes found in geom group 1')
    return meshes


def robot_image(data, meshes, pose, width, height):
    image, draw = panel(width, height)
    center = np.array([*pose[:2], 0.48])
    scale = 240
    ground(draw, center, scale, width, height)
    triangles = []
    for geom, vertices, faces in meshes:
        world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        triangles.append(project(world, center, scale, width, height)[faces])
    faces = np.concatenate(triangles)
    normal = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    length = np.linalg.norm(normal, axis=1)
    shade = np.clip(np.abs(normal[:, 2]) / np.maximum(length, 1e-9), 0, 1)
    pixels = faces[:, :, :2].astype(int).reshape(-1, 6)
    for index in np.argsort(faces[:, :, 2].mean(axis=1)):
        level = int(100 + 120 * shade[index])
        draw.polygon(tuple(pixels[index]), fill=(level, level + 8, level + 18))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bvh', 'rollout', 'xml', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--human-offset', type=float, default=0.0,
                        help='Seconds into the BVH corresponding to rollout time zero')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    joints, human, human_dt = read_bvh(args.bvh)
    with np.load(args.rollout, allow_pickle=False) as trace:
        states = trace['states'][:, :29]
        failed = np.r_[False, trace['step_failures']]
        fps = float(trace['fps'])
    if len(states) != len(failed) or not len(states) or fps <= 0:
        raise ValueError('Invalid Q1 rollout')
    times = np.arange(0, (len(states) - 1) / fps, 1 / 30)
    times = np.r_[times, (len(states) - 1) / fps]
    human_frames = (times + args.human_offset) / human_dt
    if min(human_frames) < 0 or max(human_frames) > len(human) - 1:
        raise ValueError('BVH does not cover the requested rollout time range')

    width, height = 480, 360
    robot = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    robot_data = mujoco.MjData(robot)
    meshes = robot_meshes(robot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp_dir:
        temporary_video = Path(temp_dir) / "render.mp4"
        process = subprocess.Popen([
            'ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{2 * width}x{height + 32}', '-r', '30', '-i', '-',
            '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags',
            '+faststart', str(temporary_video),
        ], stdin=subprocess.PIPE)
        try:
            for frame, (time, source) in enumerate(zip(times, human_frames, strict=True)):
                lo = int(source)
                fraction = source - lo
                pose = human[lo] * (1 - fraction) + human[min(lo + 1, len(human) - 1)] * fraction
                index = min(round(time * fps), len(states) - 1)
                robot_data.qpos[:] = states[index]
                mujoco.mj_kinematics(robot, robot_data)
                image = Image.new('RGB', (width * 2, height + 32), '#19212b')
                image.paste(human_image(joints, pose, width, height), (0, 32))
                image.paste(robot_image(robot_data, meshes, states[index], width, height), (width, 32))
                draw = ImageDraw.Draw(image)
                draw.text((12, 9), f'HUMAN MOCAP  {time:.2f}s', fill='white')
                label = 'FAILED' if failed[index] else 'TRACKING'
                draw.text((width + 12, 9), f'Q1 GPU ROLLOUT  {time:.2f}s  {label}',
                          fill='#ff958a' if failed[index] else 'white')
                process.stdin.write(image.tobytes())
                if frame % 90 == 0 or frame + 1 == len(times):
                    print(f'Rendered {frame + 1}/{len(times)} frames', flush=True)
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg encoding failed')
        temporary_video.replace(args.output)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
