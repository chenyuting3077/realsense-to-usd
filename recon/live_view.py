"""Window of recon/live.py (own process, see live_workers.py), laid out like RTAB-Map's:

  map view (left)      the nvblox mesh in the world (map) frame, the trajectory (red) and the
                       current camera (green frustum; red while tracking is lost). View modes:
                       Free (mouse), Follow (behind and above the camera) and Camera (first
                       person, from the current pose with the IR camera's field of view)
  camera (right top)   the left IR image ORB-SLAM3 tracks, ~15 Hz, with its ORB features: green
                       matched to a map point, orange not; the border turns yellow with fewer than
                       LOW_MATCHES matches or turning faster than FAST_DPS[0], red when turning
                       faster than FAST_DPS[1] or tracking is lost
  depth (right)        the newest frame given to the mapper: FFS depth (turbo, near red to far
                       blue) over the color image, updated with the mesh; no overlay where there is
                       no depth or it is beyond DEPTH_RANGE
  status, controls     tracking / depth / map numbers, the trajectory corrections (loop closure,
                       map merge, new map, re-level), layer toggles and the session buttons:
                         Stop               end the session and save it (= Ctrl+C in the terminal)
                         Discard & restart  throw the session away (its output dir deleted, after
                                            a confirmation) and start a new one right away
                         New recording      after a session ended: start the next one
                         Re-level (IMU)     with the D455 held still: level the world again from
                                            the accelerometer (the map is rebuilt in the new world)

map_q (from the mapper): ("session", out dir) when a session starts, ("reset",) before the next
one, ("map", vertices, triangles, colors uint8, camera centers, status dict, (color, depth) or
None). cam_q: ("cam", left IR image, keypoints Nx3 (x, y, matched), left-IR camera-to-world or
None, track ms, ORB-SLAM3 tracking ok or "imu" (carried on the IMU alone, not mapped), turning speed
deg/s) from the tracker; ok without a pose: tracking in a map
ORB-SLAM3 started after losing track, not used until merged into the main one. ui_q (to the mapper): "stop", "discard", "new", "level".
"""
import signal
import threading

import numpy as np

LOW_MATCHES = 50
FAST_DPS = (20, 40)  # turning speed (deg/s): warn, too fast (blur, tracking lost)
DEPTH_RANGE = (0.2, 2.5)  # as the mapper (--depth-max): farther is left uncolored
STATUS_LINES = 10
# The session's steps (live.py's status "stage"), shown in a row with the current one highlighted.
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
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    return pts, lines


def status_text(s, cam):
    if s is None:
        return "starting: camera, ORB-SLAM3, FFS ..."
    lines = [f"{s['t']:.0f} s" + ("  - stopping, final map ..." if s["stopping"] else ""),
             f"tracking  {s['tracked']} IR pairs, {s['skipped']} skipped, lag {s['lag_ms']:.0f} ms"]
    if cam is not None:
        st_ = "OK" if cam["ok"] else "on the IMU alone (not mapped)" if cam.get("imu_only") else \
            "in another map (not used until merged)" if cam["other_map"] else "LOST"
        lines.append(f"          {st_}, {cam['matched']}/{cam['features']} features matched")
    lines += [f"depth     {s['depth']} frames ({s['depth_hz']:.1f} Hz), FFS {s['ffs_ms']:.0f} ms",
              f"map       v{s['map_version']}, {s['stored']} frames, {s['triangles']} triangles, "
              f"capture -> map {s['latency_ms']:.0f} ms"]
    if s["rebuild"]:
        lines.append(f"          rebuilding with {s['rebuild']}")
    lines.append("trajectory corrections:" if s["events"] else "trajectory corrections: none yet")
    lines += [f"  {t:6.1f} s  v{v} {kind}" for kind, t, v in s["events"][-6:]]
    lines += [f"note: {n}" for n in s.get("notes", [])]
    if s.get("done"):
        lines += ["", f"saved: {s['done']}", "New recording for another session, or close the window to exit"]
    return "\n".join(lines)


def view_process(map_q, cam_q, ui_q, st, z_up, camera):
    """st: the IR camera (meta["stereo"]); z_up: leveled world (else the first camera's frame, y down);
    camera: live from the D455 (re-level needs its accelerometer)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    win = app.create_window("live reconstruction", 1800, 1000)
    em = win.theme.font_size
    up = np.array([0.0, 0, 1]) if z_up else np.array([0.0, -1, 0])

    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    scene.scene.set_background([0.12, 0.12, 0.14, 1])
    scene.scene.show_axes(True)
    if z_up:
        scene.scene.show_ground_plane(True, rendering.Scene.GroundPlane.XY)
    mesh_mat = rendering.MaterialRecord()
    mesh_mat.shader = "defaultUnlit"
    line_mat = rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 2 * win.scaling

    cam_img = gui.ImageWidget(o3d.geometry.Image(np.zeros((st["height"], st["width"], 3), np.uint8)))
    depth_img = gui.ImageWidget(o3d.geometry.Image(np.zeros((st["height"] // 2, st["width"] // 2, 3), np.uint8)))
    cam_label, depth_label = gui.Label("Camera: left IR, ORB features"), gui.Label("Depth: FFS over color")
    status = gui.Label(status_text(None, None))
    controls = gui.Vert(0.4 * em)
    mode = gui.Combobox()
    for m in ("Free", "Follow", "Camera"):
        mode.add_item(m)
    row = gui.Horiz(0.5 * em)
    row.add_child(gui.Label("Map view"))
    row.add_child(mode)
    controls.add_child(row)
    toggles = gui.Horiz(0.5 * em)
    show = {}
    for name in ("mesh", "trajectory", "camera"):
        cb = gui.Checkbox(name.capitalize())
        cb.checked = show[name] = True
        cb.set_on_checked(lambda on, n=name: (show.__setitem__(n, on), scene.scene.show_geometry(n, on)
                                               if scene.scene.has_geometry(n) else None))
        toggles.add_child(cb)
    controls.add_child(toggles)
    buttons, session = gui.Horiz(0.5 * em), gui.Horiz(0.5 * em)
    reset, relevel = gui.Button("Reset view"), gui.Button("Re-level (IMU)")
    stop, discard, new = gui.Button("Stop"), gui.Button("Discard & restart"), gui.Button("New recording")
    for b in (reset, relevel):
        buttons.add_child(b)
    for b in (stop, discard, new):
        session.add_child(b)
    controls.add_child(buttons)
    controls.add_child(session)
    title = gui.Label("")
    stages = gui.Horiz(0.3 * em)
    stage_labels = {}
    for i, (key, name) in enumerate(STAGES):
        if i:
            stages.add_child(gui.Label(">"))
        stage_labels[key] = gui.Label(name)
        stages.add_child(stage_labels[key])

    def set_stage(key):
        """Steps before the current one dimmed, the current one highlighted, the rest plain."""
        reached = True
        for k, label in stage_labels.items():
            if k == key:
                label.text_color, reached = gui.Color(1.0, 0.8, 0.2), False
            else:
                label.text_color = gui.Color(0.45, 0.45, 0.45) if reached else gui.Color(0.85, 0.85, 0.85)

    set_stage("start")
    for w in (scene, title, stages, cam_label, cam_img, depth_label, depth_img, status, controls):
        win.add_child(w)

    def on_layout(ctx):
        r = win.content_rect
        pw = min(int(r.width * 0.4), 640)
        scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        x, y, m = r.get_right() - pw, r.y, int(0.5 * em)
        iw, lh = pw - 2 * m, int(1.5 * em)
        ch = controls.calc_preferred_size(ctx, gui.Widget.Constraints()).height
        # The two images share what the labels, controls and STATUS_LINES of status leave.
        ih = int(max(min(iw * st["height"] / st["width"],
                         (r.height - 4 * lh - ch - STATUS_LINES * 1.3 * em - 7 * m) / 2), 4 * em))
        for w, h in ((title, lh), (stages, lh), (cam_label, lh), (cam_img, ih), (depth_label, lh), (depth_img, ih),
                     (controls, ch)):
            w.frame = gui.Rect(x + m, y, iw, int(h))
            y += int(h) + m
        status.frame = gui.Rect(x + m, y, iw, max(r.get_bottom() - y, 0))
    win.set_on_layout(on_layout)

    state = {"pose": None, "status": None, "cam": None, "center": np.zeros(3), "shown": False, "out": ""}

    def free_view():
        cam = scene.scene.camera
        cam.set_projection(60, max(scene.frame.width, 1) / max(scene.frame.height, 1), 0.05, 200,
                           rendering.Camera.FovType.Vertical)
        c = state["center"]
        eye = c + (np.array([-4.0, 0, 3]) if z_up else np.array([0, -3.0, -4]))
        cam.look_at(c, eye, up)

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
        else:
            back = fwd - (fwd @ up) * up
            back /= max(np.linalg.norm(back), 1e-6)
            cam.look_at(p + 0.5 * back, p - 2.5 * back + 1.5 * up, up)

    def on_mode(text, i):
        if text == "Free":
            free_view()
        scene.scene.show_axes(text != "Camera")  # world axes and ground would cut through the first-person view
        if z_up:
            scene.scene.show_ground_plane(text != "Camera", rendering.Scene.GroundPlane.XY)
        scene.scene.show_geometry("camera", show["camera"] and text != "Camera") \
            if scene.scene.has_geometry("camera") else None
        place_camera()
    mode.set_on_selection_changed(on_mode)
    reset.set_on_clicked(free_view)

    def running(on):
        """Buttons of a running session (on) or of an ended one."""
        stop.enabled = discard.enabled = on
        relevel.enabled = on and camera
        new.enabled = not on

    def send(cmd, *disable):
        ui_q.put(cmd)
        for b in disable:
            b.enabled = False

    def confirm_discard():
        dlg = gui.Dialog("Discard")
        box = gui.Vert(em, gui.Margins(em, em, em, em))
        box.add_child(gui.Label(f"Delete this session ({state['out']}) and start a new one?"))
        row = gui.Horiz(em)
        yes, no = gui.Button("Discard & restart"), gui.Button("Cancel")
        yes.set_on_clicked(lambda: (win.close_dialog(), send("discard", stop, discard, relevel)))
        no.set_on_clicked(win.close_dialog)
        row.add_child(yes)
        row.add_child(no)
        box.add_child(row)
        dlg.add_child(box)
        win.show_dialog(dlg)

    stop.set_on_clicked(lambda: send("stop", stop, discard, relevel))
    discard.set_on_clicked(confirm_discard)
    new.set_on_clicked(lambda: send("new", new))
    relevel.set_on_clicked(lambda: send("level"))
    running(False)

    def replace(name, geom, mat):
        if scene.scene.has_geometry(name):
            scene.scene.remove_geometry(name)
        scene.scene.add_geometry(name, geom, mat)
        scene.scene.show_geometry(name, show[name] and not (name == "camera" and mode.selected_text == "Camera"))

    def newest(q):
        """Blocks for an item, then skips to the newest one queued; None (the end) is kept."""
        item = q.get()
        while item is not None and not q.empty():
            item = q.get()
        return item

    def clear(out=None):
        for name in ("mesh", "trajectory", "camera"):
            if scene.scene.has_geometry(name):
                scene.scene.remove_geometry(name)
        state.update(pose=None, status=None, cam=None, shown=False)
        set_stage("start")
        if out is not None:
            state["out"] = out
            title.text = f"session: {out}"
            running(True)
        status.text = status_text(None, None)

    def run_on_main(fn):
        done = threading.Event()
        app.post_to_main_thread(win, lambda: (fn(), done.set()))
        done.wait()  # one update at a time: the reader skips to the newest item meanwhile

    def map_reader():
        while True:
            # Session messages are applied in order; of the meshes only the newest after them.
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
                        free_view()
                        state["shown"] = True
                if path is not None:
                    replace("trajectory", path, line_mat)
                if img is not None:
                    depth_img.update_image(img)
                state["status"] = s
                status.text = status_text(s, state["cam"])
                set_stage(s.get("stage", "record"))
                if s.get("done"):
                    running(False)
                win.set_needs_layout()
            run_on_main(apply)

    def cam_reader():
        while True:
            item = newest(cam_q)
            if item is None:
                return
            _, img, kps, T, track_ms, tracking, dps = item
            imu_only = tracking == "imu"
            other_map = bool(tracking) and T is None and not imu_only
            rgb, matched = draw_camera(img, kps, bool(tracking), track_ms, other_map, dps, imu_only)
            image = o3d.geometry.Image(rgb)
            fr = None
            if T is not None:
                pts, lines = frustum(T, st)
                fr = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
            col = [0.1, 1, 0.2] if T is not None else [1, 0.55, 0] if imu_only else [1, 0.85, 0] if other_map else [1, 0.1, 0.1]

            def apply():
                cam_img.update_image(image)
                state["cam"] = {"ok": T is not None, "matched": matched, "features": len(kps),
                                "other_map": other_map, "imu_only": imu_only}
                if fr is not None:
                    fr.paint_uniform_color(col)
                    state["pose"] = T
                    replace("camera", fr, line_mat)
                    place_camera()
                elif scene.scene.has_geometry("camera"):  # lost: the last pose in red
                    pts, lines = frustum(state["pose"], st) if state["pose"] is not None else (None, None)
                    if pts is not None:
                        g = o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))
                        g.paint_uniform_color(col)
                        replace("camera", g, line_mat)
                status.text = status_text(state["status"], state["cam"])
            run_on_main(apply)

    threading.Thread(target=map_reader, daemon=True).start()
    threading.Thread(target=cam_reader, daemon=True).start()
    app.run()
