import cv2 
# image
img_file='car4.jpg'
video= cv2.VideoCapture('people1.mp4')

# https://www.youtube.com/watch?v=d4L1Pte7zVc
# https://www.youtube.com/watch?v=WriuvU1rXkc


# our pre-trained model
car_tracker_file= 'cars.xml'
people_tracker_file= 'people.xml'


# car and people classifier
car_tracker =cv2.CascadeClassifier(car_tracker_file)
people_tracker = cv2.CascadeClassifier(people_tracker_file)


#  run forever until car stops
while True:
    # read the current frame
    (read_successful, frame)= video.read()


    # safe coding
    if read_successful:
        # must convert to greyscale
        greyscaled_frame =cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # car detect and people
    cars=car_tracker.detectMultiScale(greyscaled_frame)
    people=people_tracker.detectMultiScale(greyscaled_frame)
    # print(cars)

    # draw rectangle around cars
    for (x, y, w, h,) in cars:
        cv2.rectangle(frame, (x+1 ,y+1), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (0, 0, 255), 2)


    # drow rectangle around people
    for (x, y, w, h,) in people:
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (0, 255, 255), 2)



    # display the image with the faces spotted
    cv2.imshow('self ', frame)


    # dont autoclose wait here in the code and listen the key press
    key=cv2.waitKey()


    # stop if q key is pressed
    if key==81 or key==113:
        break
video.release()

'''
# create an opencv image
im = cv2.imread(img_file)



# conver file into black and white
black_and_white = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# car classifier
car_tracker =cv2.CascadeClassifier(classfier_file)

# car detect
cars=car_tracker.detectMultiScale(black_and_white)

# print(cars)

for (x, y, w, h,) in cars:
    cv2.rectangle(im, (x ,y), (x+w, y+h), (0, 0, 255), 2)

# car2 = cars[1]
# (x, y, w, h) = car2




# display the image with the faces spotted
cv2.imshow('car detector', im)


# dont autoclose wait here in the code and listen the key press
cv2.waitKey()

print('program is completed')'''