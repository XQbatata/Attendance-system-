
import cv2 as cv
import os

# To get the file's path
current_path = os.getcwd()

new_path = os.path.join(current_path, "CollectedFaces")

#os.mkdir(new_path)

# Your name here....
name = input("Enter your name to be recognized: ")

full_path = os.path.join(new_path, name)

os.mkdir(full_path)

cam = cv.VideoCapture(0)

img_counter = 0

while True:
    is_success, frame = cam.read()
    if not is_success:
        print("failed to capture image.")
        break
    cv.imshow("Collecting Data Phase | Press space to capture", frame)

    # Stores the pressed key by the user.
    k = cv.waitKey(1)

    # Check if ESC is pressed
    if k % 256 == 27:
        print("ESC is pressed... Cancel")
        break

    # Check if space is pressed
    elif k % 256 == 32:
        img_name = f"{full_path}/{name}_img_{img_counter}.jpg"
        cv.imwrite(img_name, frame)
        print(img_name)
        print(f"Image #{img_counter} saved!")
        img_counter += 1

cam.release()
cv.destroyAllWindows()