
def get_atd_paths(game=GAME, data_dir=None):
    """Return (meta, raw) ATD paths, or raise a readable error."""
    require(["atd_meta", "atd_raw"], game, data_dir, "tactical")  # noqa: F821
    from games import DATA_DIR
    paths = {
        "meta": os.path.join(DATA_DIR, "tactical", "BAR-ATH_tactical_atd.xml"),
        "raw":  os.path.join(DATA_DIR, "tactical", "BAR-ATH_tactical_atd.txt"),
    }
    for name in ("meta", "raw"):
        if not os.path.isfile(paths[name]):
            raise IOError(
                "ATD file not found: {}\n\n"
                "The match data is gated -- see DATA.md. Unpack the bundle into "
                "./data or point MTAH_DATA_DIR at where you put it.".format(paths[name])
            )
    return paths["meta"], paths["raw"]


def read_atd_window(frame_range, game=GAME, data_dir=None):
    """Stream the tactical ATD and return per-frame arrays for the requested
    video frames (0-based, inclusive).

    Returns a dict with:
      frame_ids  -- list of 1-based frame ids in the ATD file
      players    -- list of (n_tracks, 2) arrays of (x, y) in 0-1, None where missing
      ball       -- list of (2,) arrays of ball (x, y), None where missing
      n_tracks   -- number of track slots in this export
    """
    meta_path, raw_path = get_atd_paths(game, data_dir, "tactical")
    with open(meta_path) as handle:
        tree = ET.parse(handle)
    n_tracks = len(tree.getroot().iter("Player"))

    frame_ids = []
    players = []
    ball = []
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            if frame_id < frame_range[0] or frame_id > frame_range[1]:
                continue

            fields = parts[1].split(";")
            if len(fields) != n_tracks:
                raise ValueError(
                    "Expected {} track slots, got {}. Line: {}".format(
                        n_tracks, len(fields), line[:80])
                )

            frame_ids.append(frame_id)
            players.append([None] * n_tracks)
            for slot, group in enumerate(fields):
                if group[0] == MISSING:
                    continue
                comma = group.index(",")
                x = float(group[:comma])
                y = float(group[comma + 1:group.index(",", comma + 1)])
    print("Getting ATD paths...")
    meta_path, raw_path = get_atd_paths(game, data_dir, "tactical")
    print("  meta:", meta_path)
    print("  raw:", raw_path)
                players[-1][slot] = (x, y)

            # Ball is parts[2] -- format "x,y,speed" or "NaN,NaN,NaN"
            ball_field = parts[2]
            if ball_field[0] == MISSING:
                ball.append(None)
            else:
                comma = ball_field.index(",")
                bx = float(ball_field[:comma])
                by = float(ball_field[comma + 1:ball_field.index(",", comma + 1)])
                ball.append((bx, by))

    return {
        "frame_ids": frame_ids,
        "players": players,
        "ball": ball,

def draw_frame(ax, frame_idx, vis):
    """Draw one frame onto ax: pitch background, players, and the ball."""
    ax.clear()
    ax.set_xlim(-3, PITCH_LENGTH + 3)
    ax.set_ylim(-3, PITCH_WIDTH + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "video frame {:>6}  (ATD record {})".format(
            vis["frame_ids"][frame_idx], vis["frame_ids"][frame_idx]),
        fontsize=10,

def generate_animation(png_files, output_dir, fps=25):
    """Create an MP4 (ffmpeg) or GIF (matplotlib) from the PNG sequence."""
    # Try ffmpeg first (MP4)
    mp4_path = os.path.join(output_dir, "animation.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps),
             "-i", os.path.join(output_dir, "frame_%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4_path],
            check=True, capture_output=True, timeout=120,
        )
        print("  wrote {}".format(mp4_path))
        return
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fall back to matplotlib GIF
    try:
        import matplotlib.animation as animation
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        def update(k):
            ax.clear()
            ax.set_xlim(0, PITCH_LENGTH)
            ax.set_ylim(0, PITCH_WIDTH)
            ax.set_aspect("equal")
            ax.set_title("animation frame {}".format(k))
            return [ax]
        anim = animation.FuncAnimation(
            fig, update, frames=len(png_files), interval=1000 // fps, blit=False
        )
        gif_path = os.path.join(output_dir, "animation.gif")
        anim.save(gif_path, writer="pillow", fps=fps)
        plt.close(fig)
        print("  wrote {}".format(gif_path))
    except Exception as exc:  # pragma: no cover
        print("  animation fallback failed: {}".format(exc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise tracking for a 1-minute window of the match"
    )
    parser.add_argument(
        "--frames", type=int, nargs=2, metavar=("START", "END"),
        help="video frames (inclusive) to render",
    )
    parser.add_argument(
        "--minutes", type=float,
        help="window of this many minutes, centred on frame 90000",
    )
    parser.add_argument(
        "--output", default="visual_frames",
        help="output directory for PNGs",
    )
    parser.add_argument(
        "--trail-frames", type=int, default=30,
        help="how many future frames to look ahead for trails",
    )
    args = parser.parse_args()

    if args.frames:
        frame_range = (args.frames[0], args.frames[1])
    elif args.minutes:
        centre = 90000
        n_frames = int(args.minutes * FPS)
        frame_range = (centre - n_frames // 2, centre + n_frames // 2)
    else:
        frame_range = (11185, 11185 + FPS * 60)  # default: first graded minute

    print("Rendering {} frames ({}..{})".format(
        frame_range[1] - frame_range[0] + 1, *frame_range))

    vis = read_atd_window(frame_range)
    print("Loaded {} valid frames".format(len(vis["frame_ids"])))

    render_window(
        vis, list(range(frame_range[0], frame_range[1] + 1)),
        args.output, trail_frames=args.trail_frames,
    )
    print("Done.  PNGs in {}".format(args.output))

    )

    # Pitch background
    from matplotlib.patches import Rectangle, Circle
    ax.add_patch(Rectangle(
        (0, 0), PITCH_LENGTH, PITCH_WIDTH,
        facecolor="#3f8f4f", edgecolor="#ffffff", zorder=0,
    ))
    ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH],
            color="#ffffff", zorder=1)
    ax.add_patch(Circle(
        (PITCH_LENGTH / 2, PITCH_WIDTH / 2), 9.15,
        fill=False, edgecolor="#ffffff", zorder=1,
    ))
    for side in (0, 1):
        for depth, box_width in ((16.5, 40.3), (5.5, 18.3)):
            x = 0 if side == 0 else PITCH_LENGTH - depth
            ax.add_patch(Rectangle(
                (x, (PITCH_WIDTH - box_width) / 2), depth, box_width,
                fill=False, edgecolor="#ffffff", zorder=1,
            ))

    # Players -- neutral colour
    for slot in range(vis["n_tracks"]):
        px, py = vis["players"][frame_idx][slot]
        if px is None or py is None:
            continue
        ax.scatter(px * PITCH_LENGTH, py * PITCH_WIDTH, s=70, zorder=3,

def render_window(vis, frame_indices, output_dir, trail_frames=30):
    """Render every frame in frame_indices as a PNG pitch overlay.

    trail_frames: when set to N > 0, look ahead N future frames to draw a
    faint trail of where players and the ball have been -- this is the
    "future data" trick you mentioned, purely for display.
    """
    import matplotlib
    matplotlib.use(os.environ.get("MPL_BACKEND", "Agg"))
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    n = len(frame_indices)
    print("Rendering {} frames to {}".format(n, output_dir))

    for k, frame_idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        draw_frame(ax, k, vis)
        # Draw a faint trail on the same axes (looks ahead into the future)
        if trail_frames > 0 and k + trail_frames < n:
            start = max(0, k)
            end = min(n, k + trail_frames + 1)
            for j in range(start, end):
                px, py = vis["players"][j]
                for slot in range(vis["n_tracks"]):
                    x, y = px[slot]
                    if x is None or y is None:
                        continue
                    ax.plot(x * PITCH_LENGTH, y * PITCH_WIDTH,
                            color="gray", alpha=0.15, linewidth=0.8, zorder=2)
            bx, by = vis["ball"][k + 10]  # mid-point of trail window
            if bx is not None and by is not None:
                ax.plot(bx * PITCH_LENGTH, by * PITCH_WIDTH,
                        color="gray", alpha=0.15, linewidth=0.8, zorder=2)

        out_path = os.path.join(
            output_dir, "frame_{:05d}.png".format(frame_idx)
        )
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        if (k + 1) % 100 == 0:
            print("  wrote {} ({:.0f}% of {}, frame {})".format(
                os.path.basename(out_path), 100 * (k + 1) / n,
                n, frame_idx,
            ))

    # Assemble a GIF animation from the PNGs (requires ffmpeg or PIL)
    png_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    if png_files:
        generate_animation(png_files, output_dir, fps=FPS)

    return output_dir

                   facecolor="#d62728", edgecolor="black", linewidth=0.5)

    # Ball -- white dot
    bx, by = vis["ball"][frame_idx]
    has_ball = bx is not None and by is not None
    if has_ball:
        ax.scatter(bx * PITCH_LENGTH, by * PITCH_WIDTH, s=80, zorder=4,
                   facecolor="white", edgecolor="black", linewidth=1.2)

    ax.set_title(
        ax.get_title() + ("   -- ball tracked" if has_ball else "   -- ball NOT tracked")
    )

        "n_tracks": n_tracks,
    }
