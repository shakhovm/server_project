from clearml import Task
from clearml import StorageManager
import torch
import cv2
import json
import numpy as np
import sys
Task.init(project_name='myproject', task_name='initial_version')


# StorageManager.\
#     upload_file('00_42_36.264','http://172.25.21.178:8081/manual_artifacts/00_42_36.264')
# StorageManager.


# local_path = Dataset.get(dataset_id='834d11b8999940b386327133fa043b22').get_local_copy()
# print(local_path)

if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or 'yolov3_spp', 'yolov3_tiny'
    k = 10
    cap = cv2.VideoCapture(f"00_42_36.264")
    print(cap.isOpened())

    # im = ImagePreprocessor()
    results = []
    print("Model Loaded!")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = frame
        # img_first = im.to_rgb(img)
        # img = im.adjust_gamma(frame, 0.5)
        # helper.imshow(img)
        bbox = model(img)
        # bbox.print()


        bboxes = np.array(bbox.xyxy[0].cpu())
        # print(bboxes)
        # helper.display_bboxes(frame, bboxes)
        lst = []
        for bbox in bboxes:
            lst.append(list(map(float, bbox)))
        results.append(lst)
        # bboxes = helper.get_human(bboxes)
        # if len(bboxes) > 0:
        #     results.append([list(map(float, bbox)) for bbox in bboxes])
        # else:
        #     results.append([])
        # helper.imshow(img_first)
        # cv2.imshow("hello", helper.resize(frame, 0.5))
        # key = cv2.waitKey(1)
        # if key & 0xFF == ord('q'):
        #     break
        if k < 0:
            break
        k -= 1
    # print(results)
    with open("2020-02-20.json", 'w') as f:
        json.dump(results, f, indent=2)

