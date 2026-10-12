"""Window of main.py (own process): the map (nvblox mesh, trajectory, current camera; Follow /
Free / Camera views), the tracked IR image with its ORB features, the FFS depth over color, status,
and the session buttons.

map_q: ("session", out dir), ("reset",), ("map", vertices, triangles, colors, camera centers,
status, (color, depth) or None), ("estimate", seconds) and ("progress", 0..1, text, finished)
of the USD export.
cam_q: ("cam", IR image, keypoints Nx3 (x, y, matched), pose or None, track ms, tracking ok or
"imu" (on the IMU alone), turning deg/s). ui_q (to main.py): "stop", "discard", "new", "usd".
"""
import signal
import threading

import numpy as np

LOW_MATCHES = 50
FAST_DPS = (20, 40)  # turning speed (deg/s): warn, too fast
DEPTH_RANGE = (0.2, 2.5)  # as the live map
STATUS_LINES = 9
STAGES = [("start", "Start"), ("record", "Record"), ("trajectory", "Final trajectory"),
          ("mesh", "Final mesh"), ("save", "Save"), ("done", "Done")]


def draw_camera(img, kps, ok, track_ms, other_map=False, dps=0.0, imu_only=False):
    import cv2
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    matched = int(kps[:, 2].sum()) if len(kps) else 0
    for x, y, m in kps:
        cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0) if m else (255, 150, 0), 1, cv2.LINE_AA)
    border = (255, 0, 0) if not ok or dps > FAST_DPS[1] else (255, 140, 0) if imu_only else \
        (255, 220, 0) if matched < LOW_MATCHES or other_map or dps > FAST_DPS[0] else None
    if border:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), border, 12)

    def put(text, y, color, scale=0.9):
        cv2.putText(out, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    put(("IMU ONLY (not mapped)" if imu_only else "OTHER MAP (until merged)" if other_map
         else f"{matched}/{len(kps)} matched  {track_ms:.0f} ms") if ok else "LOST", 34, border or (255, 255, 255))
    if ok:
        put(f"turning {dps:.0f} deg/s" + ("  SLOW DOWN" if dps > FAST_DPS[0] else ""), 70,
            (255, 0, 0) if dps > FAST_DPS[1] else (255, 220, 0) if dps > FAST_DPS[0] else (255, 255, 255))
    if ok and matched < LOW_MATCHES:
        put("FEW FEATURES: aim at textured things", out.shape[0] - 20, (255, 220, 0), 0.8)
    return np.ascontiguousarray(out), matched


def draw_depth(color, depth):
    import cv2
    near, far = DEPTH_RANGE
    v = np.clip((depth - near) / (far - near), 0, 1)
    turbo = cv2.cvtColor(cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    out = color.copy()
    valid = (depth > near) & (depth <= far)
    out[valid] = (0.35 * color[valid] + 0.65 * turbo[valid]).astype(np.uint8)
    return np.ascontiguousarray(out)


def frustum(T, st, depth=0.4):
    w, h, fx, fy, cx, cy = st["width"], st["height"], st["fx"], st["fy"], st["cx"], st["cy"]
    corners = np.array([[(u - cx) / fx * depth, (v - cy) / fy * depth, depth] for u, v in
                        ((0, 0), (w, 0), (w, h), (0, h))])
    pts = np.vstack([np.zeros(3), corners]) @ T[:3, :3].T + T[:3, 3]
    return pts, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]


GREEN, YELLOW, RED, GREY = (0.35, 0.85, 0.45), (1.0, 0.8, 0.25), (1.0, 0.35, 0.3), (0.6, 0.6, 0.62)


def status_lines(s, cam):
    """(text, color) lines: tracking, depth, map; then corrections and notes in grey."""
    if s is None:
        return [("Starting camera, ORB-SLAM3 and FFS ...", GREY)]
    if cam is None or cam["ok"]:
        track = ("TRACKING", GREEN if cam is None or cam["matched"] >= LOW_MATCHES else YELLOW)
    else:
        track = ("ON THE IMU ONLY", YELLOW) if cam.get("imu_only") else ("OTHER MAP", YELLOW) if cam["other_map"] \
            else ("LOST: go back to a place seen before", RED)
    feats = f", {cam['matched']}/{cam['features']} features matched" if cam else ""
    lines = [(f"{track[0]}  {s['tracked']} frames{feats}", track[1]),
             (f"DEPTH  {s['depth_hz']:.1f} Hz, FFS {s['ffs_ms']:.0f} ms", GREEN if s["depth"] else GREY),
             (f"MAP  {s['triangles']:,} triangles, {s['stored']} frames, {s['latency_ms']:.0f} ms behind"
              + (f", rebuilding {s['rebuild']}" if s["rebuild"] else ""), GREEN if s["triangles"] else GREY)]
    if s["stopping"] and not s.get("done"):
        lines.append(("Stopping: final trajectory and mesh ...", YELLOW))
    lines += [(f"{t:6.1f} s  {kind}", GREY) for kind, t, v in s["events"][-4:]]
    lines += [(n, YELLOW) for n in s.get("notes", [])]
    if s.get("done"):
        lines.append((s["done"], GREEN if s["done"].startswith("Saved") else YELLOW))
    return lines


def view_process(map_q, cam_q, ui_q, st, z_up):
    """st: the IR camera (meta["stereo"]); z_up: leveled world (else the first camera's frame, y down)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    win = app.create_window("realsense-to-usd", 1800, 1000)
    em = win.theme.font_size
    up = np.array([0.0, 0, 1]) if z_up else np.array([0.0, -1, 0])
    color = lambda c: gui.Color(*c)  # noqa: E731

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([0.1, 0.1, 0.12, 1])
    scene.scene.show_axes(True)
    if z_up:
        scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XY)
    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultUnlit"
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2 * win.scaling

    # header: session and steps
    title = gui.Label("")
    stages = gui.Horiz(0.3 * em)
    stage_labels = {}
    for i, (key, name) in enumerate(STAGES):
        if i:
            stages.add_child(gui.Label(">"))
        stage_labels[key] = gui.Label(name)
        stages.add_child(stage_labels[key])

    # actions: only the buttons that apply now are shown
    actions = gui.Horiz(0.5 * em)
    stop, discard = gui.Button("  Stop  "), gui.Button("Discard & restart")
    usd, new = gui.Button("  3DGS -> USD  "), gui.Button("New recording")
    stop.background_color, usd.background_color = color((0.7, 0.2, 0.2)), color((0.2, 0.42, 0.75))
    eta = gui.Label("")  # the export's estimated time, next to its button
    eta.text_color = color(GREY)
    for b in (stop, discard, usd, eta, new):
        actions.add_child(b)
    actions.add_stretch()
    progress, progress_label = gui.ProgressBar(), gui.Label("")
    progress_label.text_color = color(GREY)

    cam_img = gui.ImageWidget(o3d.geometry.Image(np.zeros((st["height"], st["width"], 3), np.uint8)))
    depth_img = gui.ImageWidget(o3d.geometry.Image(np.zeros((st["height"] // 2, st["width"] // 2, 3), np.uint8)))
    cam_label, depth_label = gui.Label("Tracked IR image"), gui.Label("Depth over color (near red, far blue)")
    for lbl in (cam_label, depth_label):
        lbl.text_color = color(GREY)
    status = gui.Vert(0.15 * em)
    status_rows = []
    for _ in range(STATUS_LINES):
        lbl = gui.Label("")
        status_rows.append(lbl)
        status.add_child(lbl)

    # map view settings
    view = gui.Horiz(0.5 * em)
    mode = gui.Combobox()
    for m in ("Free", "Follow", "Camera"):
        mode.add_item(m)
    mode.selected_index = 1  # Follow: behind the camera
    reset = gui.Button("Reset")
    view.add_child(gui.Label("View"))
    view.add_child(mode)
    view.add_child(reset)
    show = {}
    for name in ("mesh", "trajectory", "camera"):
        cb = gui.Checkbox(name.capitalize())
        cb.checked = show[name] = True
        cb.set_on_checked(lambda on, n=name: (show.__setitem__(n, on), scene.scene.show_geometry(n, on)
                                               if scene.scene.has_geometry(n) else None))
        view.add_child(cb)

    def set_stage(key):
        reached = True
        for k, label in stage_labels.items():
            if k == key:
                label.text_color, reached = color(YELLOW), False
            else:
                label.text_color = color((0.4, 0.4, 0.42)) if reached else color((0.8, 0.8, 0.8))

    def set_status(lines):
        for i, lbl in enumerate(status_rows):
            text, c = lines[i] if i < len(lines) else ("", GREY)
            lbl.text, lbl.text_color = text, color(c)

    set_stage("start")
    set_status(status_lines(None, None))
    panel = (title, stages, actions, progress, progress_label, cam_label, cam_img, depth_label, depth_img, status, view)
    for w in (scene,) + panel:
        win.add_child(w)

    def on_layout(ctx):
        r = win.content_rect
        pw = min(int(r.width * 0.36), 660)
        scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        m, lh = int(0.6 * em), int(1.4 * em)
        x, y, iw = r.get_right() - pw + m, r.y + m, pw - 2 * m
        bh = actions.calc_preferred_size(ctx, gui.Widget.Constraints()).height
        vh = view.calc_preferred_size(ctx, gui.Widget.Constraints()).height
        sh = status.calc_preferred_size(ctx, gui.Widget.Constraints()).height
        ih = int(max(min(iw * st["height"] / st["width"], (r.height - 6 * lh - bh - sh - vh - 11 * m) / 2), 4 * em))
        rows = [(title, lh), (stages, lh), (actions, bh), (progress, int(0.6 * em)), (progress_label, lh),
                (cam_label, lh), (cam_img, ih), (depth_label, lh), (depth_img, ih), (status, sh), (view, vh)]
        for w, h in rows:
            w.frame = gui.Rect(x, y, iw, h)
            y += h + (m if w not in (cam_label, depth_label, progress) else int(0.2 * em))
    win.set_on_layout(on_layout)

    state = {"pose": None, "status": None, "cam": None, "center": np.zeros(3), "shown": False, "out": ""}

    def free_view():
        cam = scene.scene.camera
        cam.set_projection(60, max(scene.frame.width, 1) / max(scene.frame.height, 1), 0.05, 200,
                           rendering.Camera.FovType.Vertical)
        c = state["center"]
        cam.look_at(c, c + (np.array([-4.0, 0, 3]) if z_up else np.array([0, -3.0, -4])), up)

    def place_camera():
        T = state["pose"]
        if T is None or mode.selected_text == "Free":
            return
        cam = scene.scene.camera
        p, fwd = T[:3, 3], T[:3, 2]
        if mode.selected_text == "Camera":
            fov = np.degrees(2 * np.arctan(st["height"] / 2 / st["fy"]))
            cam.set_projection(fov, max(scene.frame.width, 1) / max(scene.frame.height, 1), 0.05, 200,
                               rendering.Camera.FovType.Vertical)
            cam.look_at(p + fwd, p, -T[:3, 1])
        else:  # Follow: 2.5 m behind, 1.5 m above
            back = fwd - (fwd @ up) * up
            back /= max(np.linalg.norm(back), 1e-6)
            cam.look_at(p + 0.5 * back, p - 2.5 * back + 1.5 * up, up)

    def on_mode(text, i):
        if text == "Free":
            free_view()
        scene.scene.show_axes(text != "Camera")
        if z_up:
            scene.scene.show_ground_plane(text != "Camera", rendering.Scene.GroundPlane.XY)
        if scene.scene.has_geometry("camera"):
            scene.scene.show_geometry("camera", show["camera"] and text != "Camera")
        place_camera()
    mode.set_on_selection_changed(on_mode)
    reset.set_on_clicked(free_view)

    def running(on):
        stop.visible = discard.visible = on
        usd.visible = new.visible = eta.visible = not on
        stop.enabled = discard.enabled = usd.enabled = new.enabled = True
        win.set_needs_layout()

    def send(cmd, *disable):
        ui_q.put(cmd)
        for b in disable:
            b.enabled = False

    def confirm_discard():
        dlg = gui.Dialog("Discard")
        box = gui.Vert(em, gui.Margins(em, em, em, em))
        box.add_child(gui.Label(f"Delete {state['out']} and start a new recording?"))
        row = gui.Horiz(em)
        yes, no = gui.Button("Discard & restart"), gui.Button("Cancel")
        yes.set_on_clicked(lambda: (win.close_dialog(), send("discard", stop, discard)))
        no.set_on_clicked(win.close_dialog)
        row.add_child(yes)
        row.add_child(no)
        box.add_child(row)
        dlg.add_child(box)
        win.show_dialog(dlg)

    def set_estimate(sec):
        """None: nothing to export."""
        usd.visible = sec is not None
        eta.text = "" if sec is None else f"~{max(sec / 60, 1):.0f} min"
        win.set_needs_layout()

    def set_progress(frac, text, finished):
        progress.visible = bool(text)  # its slot stays: no layout shift
        progress.value = frac
        progress_label.text = text if finished else f"3DGS -> USD {frac * 100:.0f}%: {text}"
        progress_label.text_color = color(GREEN if finished and frac >= 1 else RED if finished else GREY)
        if finished:
            new.enabled = True
            if text:
                usd.enabled = frac < 1  # a failed export can be retried

    stop.set_on_clicked(lambda: send("stop", stop, discard))
    discard.set_on_clicked(confirm_discard)
    new.set_on_clicked(lambda: send("new", new, usd))
    usd.set_on_clicked(lambda: (send("usd", new, usd), set_progress(0.0, "starting", False)))
    running(False)
    set_progress(0.0, "", True)

    def replace(name, geom, mat):
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, geom, mat)
        scene.scene.show_geometry(name, show[name] and not (name == "camera" and mode.selected_text == "Camera"))

    def clear(out=None):
        for name in ("mesh", "trajectory", "camera"):
            if scene.scene.has_geometry(name):
                scene.scene.remove_geometry(name)
        state.update(pose=None, status=None, cam=None, shown=False)
        set_stage("start")
        set_progress(0.0, "", True)
        if out is not None:
            state["out"] = out
            title.text = out
            running(True)
        set_status(status_lines(None, None))

    def run_on_main(fn):
        done = threading.Event()
        app.post_to_main_thread(win, lambda: (fn(), done.set()))
        done.wait()  # one update at a time: the reader skips to the newest item meanwhile

    def map_reader():
        while True:
            # Session and progress messages in order; of the meshes only the newest after them.
            items = [map_q.get()]
            while not map_q.empty():
                items.append(map_q.get())
            if None in items:
                return
            last = max((i for i, it in enumerate(items) if it[0] != "map"), default=-1)
            for it in items[:last + 1]:
                if it[0] == "session":
                    run_on_main(lambda o=it[1]: clear(o))
                elif it[0] == "reset":
                    run_on_main(clear)
                elif it[0] == "progress":
                    run_on_main(lambda a=it[1:]: set_progress(*a))
                elif it[0] == "estimate":
                    run_on_main(lambda sec=it[1]: set_estimate(sec))
            maps = [it for it in items[last + 1:] if it[0] == "map"]
            if not maps:
                continue
            _, v, f, c, cams, s, frame = maps[-1]
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                             o3d.utility.Vector3iVector(f.astype(np.int32)))
            mesh.vertex_colors = o3d.utility.Vector3dVector(c[:, :3].astype(np.float64) / 255)
            path = None
            if len(cams) > 1:
                path = o3d.geometry.LineSet(o3d.utility.Vector3dVector(cams), o3d.utility.Vector2iVector(
                    np.stack([np.arange(len(cams) - 1), np.arange(1, len(cams))], 1)))
                path.paint_uniform_color([1, 0.15, 0.1])
            img = None if frame is None else o3d.geometry.Image(draw_depth(*frame))

            def apply():
                if len(f):
                    replace("mesh", mesh, mesh_mat)
                    if not state["shown"]:
                        state["center"] = v.mean(0)
                        free_view()  # also sets the projection that Follow keeps
                        place_camera()
                        state["shown"] = True
                if path is not None:
                    replace("trajectory", path, line_mat)
                if img is not None:
                    depth_img.update_image(img)
                state["status"] = s
                set_status(status_lines(s, state["cam"]))
                set_stage(s.get("stage", "record"))
                if s.get("done"):
                    running(False)
            run_on_main(apply)

    def cam_reader():
        while True:
            item = cam_q.get()
            while item is not None and not cam_q.empty():  # skip to the newest
                item = cam_q.get()
            if item is None:
                return
            _, img, kps, T, track_ms, tracking, dps = item
            imu_only = tracking == "imu"
            other_map = bool(tracking) and T is None and not imu_only
            rgb, matched = draw_camera(img, kps, bool(tracking), track_ms, other_map, dps, imu_only)
            image = o3d.geometry.Image(rgb)
            col = [0.1, 1, 0.2] if T is not None else [1, 0.55, 0] if imu_only else [1, 0.85, 0] if other_map else [1, 0.1, 0.1]

            def apply():
                cam_img.update_image(image)
                state["cam"] = {"ok": T is not None, "matched": matched, "features": len(kps),
                                "other_map": other_map, "imu_only": imu_only}
                if T is not None:
                    state["pose"] = T
                pose = state["pose"]
                if pose is not None:  # lost: the last pose in red
                    pts, lines = frustum(pose, st)
                    g = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
                    g.paint_uniform_color(col)
                    replace("camera", g, line_mat)
                    if T is not None:
                        place_camera()
                set_status(status_lines(state["status"], state["cam"]))
            run_on_main(apply)

    threading.Thread(target=map_reader, daemon=True).start()
    threading.Thread(target=cam_reader, daemon=True).start()
    app.run()
