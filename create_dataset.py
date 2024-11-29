import os
import pickle

import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
total_images = sum(
    len(files) for _, _, files in os.walk(DATA_DIR) if files
)
processed_count = 0  # Counter for processed images

OUTPUT_DIR = './output_images'  # Directory to save processed images
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip non-directory files like .DS_Store
        continue

    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            print(f"Skipping corrupt or unreadable image: {img_path}")
            continue
        img = cv2.resize(img, (640, 480))  # Resize to a smaller resolution

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract normalized coordinates
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

            # Save or display the image
            output_path = os.path.join(OUTPUT_DIR, f'{dir_}_{img_path}')
            cv2.imwrite(output_path, img)  # Save image with landmarks
            # cv2.imshow('Hand Landmarks', img)  # Display the image
            # cv2.waitKey(1)  # Adjust display time as needed
            processed_count += 1
            print(f"Processed {processed_count}/{total_images} images. Remaining: {total_images - processed_count}")

cv2.destroyAllWindows()

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

