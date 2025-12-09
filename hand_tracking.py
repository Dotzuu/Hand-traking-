import cv2
import mediapipe as mp
import numpy as np
import time

# ------------------ INIT ------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

canvas = None
prev_x, prev_y = None, None

def finger_up(lm, tip, pip):
    return lm[tip][1] < lm[pip][1]

# ------------------ LOOP ------------------
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # CREȘTE luminozitatea camerei
    frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=40)

    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    drawing = False
    erasing = False

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]

        lm = [(int(pt.x * w), int(pt.y * h)) for pt in hand.landmark]
        x, y = lm[8]

        index_up = finger_up(lm, 8, 6)
        middle_up = finger_up(lm, 12, 10)

        # -------- DRAW --------
        if index_up and not middle_up:
            drawing = True
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 7)
            prev_x, prev_y = x, y

        # -------- ERASE --------
        elif index_up and middle_up:
            erasing = True
            cv2.circle(canvas, (x, y), 40, (0, 0, 0), -1)
            prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Schelet mână (super clar)
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style()
        )

    # Combinare frame + desen
    combined = cv2.add(frame, canvas)

    # FPS counter
    cur = time.time()
    fps = int(1 / (cur - prev_time)) if prev_time != 0 else 0
    prev_time = cur

    cv2.putText(combined, f"FPS: {fps}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    if drawing:
        cv2.putText(combined, "Drawing...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    elif erasing:
        cv2.putText(combined, "Erasing...", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Hand Drawing ULTRA Smooth", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
