import cv2
import numpy as np

def create_slide_video(lr_path, sr_path, output_path='sr_demo.mp4', duration=3, fps=40, repeat=2):
    target_size = (512, 128)  # (width, height)

    # Load LR and SR images
    lr = cv2.imread(lr_path)
    sr = cv2.imread(sr_path)

    if lr is None or sr is None:
        raise ValueError("Could not load images. Check the file paths.")

    # Resize both to the target size
    lr = cv2.resize(lr, target_size)
    sr = cv2.resize(sr, target_size)

    w, h = target_size
    h, w = lr.shape[:2]
    transition_frames = int(duration * fps)
    pause_frames = int(2 * fps)  # 2 seconds pause

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    def pause_on_frame(img):
        for _ in range(pause_frames):
            video.write(img)

    def slide_transition(img1, img2, direction='left_to_right'):
        for i in range(transition_frames):
            alpha = i / transition_frames
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            if direction == 'left_to_right':
                boundary = int(alpha * w)
                frame[:, :boundary] = img2[:, :boundary]
                frame[:, boundary:] = img1[:, boundary:]
                cv2.line(frame, (boundary, 0), (boundary, h), (255, 0, 0), 2)  # Blue vertical line
            else:  # right_to_left
                boundary = int((1 - alpha) * w)
                frame[:, :boundary] = img1[:, :boundary]
                frame[:, boundary:] = img2[:, boundary:]
                cv2.line(frame, (boundary, 0), (boundary, h), (255, 0, 0), 2)  # Blue vertical line

            video.write(frame)

    # Perform transitions
    for _ in range(repeat):
        pause_on_frame(lr)
        slide_transition(lr, sr, direction='left_to_right')
        pause_on_frame(sr)
        slide_transition(sr, lr, direction='right_to_left')

    video.release()
    print(f"✅ Saved demo video to {output_path} with size {w}x{h}")

# --- Run this part to generate the video ---
if __name__ == '__main__':
    lr_image_path = 'test.png'           # 🔁 Replace with your LR image path
    sr_image_path = 'output_mynet.png'   # 🔁 Replace with your SR image path
    create_slide_video(lr_image_path, sr_image_path)
