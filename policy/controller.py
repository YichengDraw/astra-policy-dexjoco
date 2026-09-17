import numpy as np
from PIL import Image


def unit(q):
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)


def angle(a, b):
    return 2 * np.arccos(np.clip(np.abs(np.sum(unit(a) * unit(b), axis=-1)), 0, 1))


def quat(v):
    t = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.concatenate((np.cos(t / 2), 0.5 * np.sinc(t / (2 * np.pi)) * v), axis=-1)


def step(observation, memory):
    data = POLICY_DATA['policy_data.npz']
    meta = POLICY_DATA['train_metadata.json']
    meta = meta.get('metadata', meta)
    S, A, F = data['states'], data['actions'], data['features']
    E, T = data['episode_index'], data['frame_index']
    dim, n = int(meta['state_shape'][1]), len(S)
    s = np.asarray(observation['state'], dtype=np.float64)
    if dim not in (23, 46) or s.shape != (dim,) or not np.all(np.isfinite(s)) or not n:
        return {}, dict(memory)
    arms = dim // 23
    if S.shape != (n, dim) or A.shape != (n, 22 * arms) or F.shape[0] != n:
        return {}, dict(memory)
    stride = max(1, int(meta.get('stride', 1)))
    signature = [meta.get('task', ''), meta.get('regime', ''), n, dim]
    memory = dict(memory if memory.get('library') == signature else {})
    # Absolute-only tolerance across all 22 components, never just pose or hand.
    inactive = np.all(np.abs(A.reshape(n, arms, 22)) <= 1e-8, axis=2)
    valid = np.isfinite(S).all(axis=1) & np.isfinite(A).all(axis=1) & np.isfinite(F).all(axis=1)
    valid &= ~inactive.all(axis=1)
    for arm in range(arms):
        p = 7 * arm
        if np.linalg.norm(s[p+3:p+7]) < 1e-8:
            return {}, dict(memory)
        valid &= np.linalg.norm(S[:, p+3:p+7], axis=1) > 1e-8
    # Retire stored selection references to non-executable records.
    for key in ('index', 'intent', 'pending'):
        row = int(memory.get(key, -1))
        if 0 <= row < n and not valid[row]:
            memory[key] = -1
            if key == 'pending':
                memory['votes'] = 0
    if not np.any(valid):
        # A sparse command holds the current observed state in the decoder.
        return {}, dict(memory)
    scales = []
    for arm in range(arms):
        p, h = 7 * arm, 7 * arms + 16 * arm
        scales.append((np.maximum(S[valid, p:p+3].std(axis=0), 0.07),
                       np.maximum(S[valid, h:h+16].std(axis=0), 0.30)))
    images, parts = observation['images'], []
    for camera in meta['camera_keys']:
        key = camera if camera in images else meta.get('camera_mapping', {}).get(camera, camera)
        if key not in images:
            return {}, dict(memory)
        im = np.asarray(images[key])
        if im.shape != (224, 224, 3) or im.dtype != np.uint8:
            return {}, dict(memory)
        # RGB224 already has the official padding/uint8 conversion.
        small = Image.fromarray(im).resize((16, 16), Image.Resampling.BILINEAR)
        parts.append((np.asarray(small, dtype=np.float32) / np.float32(255)).reshape(-1, order='C'))
    f = np.concatenate(parts)
    if f.size != F.shape[1]:
        return {}, dict(memory)

    def proprio(rows, reference):
        out = np.zeros(len(rows), dtype=np.float64)
        for arm, (ps, hs) in enumerate(scales):
            p, h = 7 * arm, 7 * arms + 16 * arm
            dp = (rows[:, p:p+3] - reference[p:p+3]) / ps
            dh = (rows[:, h:h+16] - reference[h:h+16]) / hs
            out += 0.60 * np.mean(np.minimum(dp * dp, 25), axis=1)
            out += 0.20 * (angle(rows[:, p+3:p+7], reference[p+3:p+7]) / 0.7) ** 2
            out += 0.20 * np.mean(np.minimum(dh * dh, 25), axis=1)
        return out / arms

    views, visual_scales = [], []
    for c in range(len(parts)):
        block = F[:, c*768:(c+1)*768]
        scale = np.clip(block[valid].std(axis=0), 0.05, 0.30)
        visual_scales.append(scale)
        z = (block - f[c*768:(c+1)*768]) / scale
        views.append(np.mean(np.minimum(z * z, 16), axis=1))
    views, vscale = np.asarray(views), np.concatenate(visual_scales)
    visual = 0.75 * views.mean(axis=0) + 0.25 * views.max(axis=0)
    cost = 2.0 * visual + proprio(S, s)
    previous = np.asarray(memory.get('state', s), dtype=np.float64)
    previous_image = np.asarray(memory.get('feature', np.rint(f * 255)), dtype=np.float32) / 255
    motion = float(proprio(previous[None], s)[0] + np.mean(np.minimum(((f-previous_image)/vscale)**2, 16)))
    stale = 0 if motion > 0.0003 else int(memory.get('stale', 0)) + 1
    # Observed direction helps distinguish revisited poses, without an execution clock.
    order, pred = np.lexsort((T, E)), np.arange(n)
    order = order[valid[order]]
    before, after = order[:-1], order[1:]
    edges = (E[before] == E[after]) & (T[after] > T[before]) & (T[after]-T[before] <= 2*stride)
    edges &= valid[before] & valid[after]
    pred[after[edges]] = before[edges]
    for arm in range(arms):
        p = 7 * arm
        observed = s[p:p+3] - previous[p:p+3]
        length = np.linalg.norm(observed)
        if length > 0.001:
            velocity = S[:, p:p+3] - S[pred, p:p+3]
            speed = np.linalg.norm(velocity, axis=1)
            direction = (velocity @ observed) / np.maximum(speed * length, 1e-9)
            cost += (0.07 / arms) * np.where(speed > 0.001, 1-np.clip(direction, -1, 1), 0)
    cost[~valid | ~np.isfinite(cost)] = np.inf
    best = int(np.argmin(cost))
    if not np.isfinite(cost[best]):
        return {}, dict(memory)
    old = int(memory.get('index', -1))
    anchored = 0 <= old < n and np.isfinite(cost[old])
    index, pending, votes = best, -1, 0
    if not anchored:
        _, group = np.unique(E, return_inverse=True)
        starts = np.full(int(group.max()) + 1, np.inf)
        np.minimum.at(starts, group[valid], T[valid])
        opening = np.where(valid & (T <= starts[group] + 2*stride), cost, np.inf)
        first = int(np.argmin(opening))
        if opening[first] <= cost[best] + 0.15:
            index = first
    else:
        delta = (T - T[old]) / stride
        local = valid & (E == E[old]) & (delta >= -6) & (delta <= 30)
        score = cost + 0.025 * (delta < 0) + 0.003 * (delta != 0)
        index = int(np.argmin(np.where(local, score, np.inf)))
        margin = (0.008 + 0.04*cost[best]) if stale >= 12 else (0.025 + 0.12*cost[best])
        if best != index and cost[best] + margin < cost[index]:
            candidate = int(memory.get('pending', -1))
            agrees = 0 <= candidate < n and valid[candidate] and E[candidate] == E[best] and abs(T[candidate]-T[best]) <= 9*stride
            pending, votes = best, (int(memory.get('votes', 0)) + 1 if agrees else 1)
            if votes >= 3:
                index, pending, votes = best, -1, 0

    # Lookahead never updates the observation-aligned phase or accumulates with calls.
    intent = index
    future = np.flatnonzero(valid & (E == E[index]) & (T > T[index]) & (T <= T[index]+2*stride))
    for j in future[np.argsort(T[future])]:
        if cost[j] > cost[index] + 0.035 or cost[j] > cost[best] + 0.20:
            break
        scene = float(np.mean(np.minimum(((F[j]-F[index])/vscale)**2, 16)))
        if scene > 0.025 or proprio(S[j:j+1], S[index])[0] > 0.035:
            break
        smooth = True
        for arm in range(arms):
            # Inactive blocks are not targets; do not look ahead across activity changes.
            if inactive[index, arm] != inactive[j, arm]:
                smooth = False
                break
            if inactive[index, arm]:
                continue
            a = 22 * arm
            u, v = A[index, a:a+22], A[j, a:a+22]
            smooth = (smooth and np.linalg.norm(u[:3]-v[:3]) <= 0.08
                      and angle(quat(u[3:6]), quat(v[3:6])) <= 0.30
                      and np.max(np.abs(u[6:]-v[6:])) <= 0.35)
        if not smooth:
            break
        intent = int(j)
    # Preserve absolute command intent, including legitimate demonstrated tracking lag.
    command = {'transition_steps': 3}
    for arm, name in enumerate(('right', 'left')[:arms]):
        if inactive[intent, arm]:
            # Omission holds this arm's current observed TCP, quaternion and hand.
            continue
        goal = A[intent, 22*arm:22*arm+22]
        fingers = ('ff', 'mf', 'rf', 'th') if arm == 0 else ('rf', 'mf', 'ff', 'th')
        command[name] = {'xyz': goal[:3].tolist(), 'rotvec': goal[3:6].tolist(),
                         'hand_target': {finger+'j'+str(j): float(goal[6+4*k+j])
                                         for k, finger in enumerate(fingers) for j in range(4)}}
    new_memory = {'library': signature, 'index': int(index), 'intent': int(intent),
                  'pending': int(pending), 'votes': int(votes), 'stale': int(stale),
                  'state': s.tolist(), 'feature': np.rint(f*255).astype(int).tolist()}
    return command, new_memory
