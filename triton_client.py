import cv2
import numpy as np
import sys
from utils import *
from tqdm import tqdm
import tritonclient.grpc as tritongrpcclient


if __name__=="__main__":
    image_name = "the_meeting.jpg"

    anchors = np.load("anchors.npy")

    url = 'localhost:8001'
    model_name = 'blazeface_onnx'

    total_requests = 100

    input_img_orig = cv2.imread('the_meeting.jpg')
    input_img_orig = cv2.cvtColor(input_img_orig, cv2.COLOR_BGR2RGB)
    orig_size =min(input_img_orig.shape[0],input_img_orig.shape[1])
    input_img, shifts = resize_and_crop_image(input_img_orig,128)

    try:
        triton_client = tritongrpcclient.InferenceServerClient(
            url=url,
            verbose=False
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    inputs = []
    outputs = []

    inputs.append(tritongrpcclient.InferInput('cropped_face', input_img.shape, "FP32"))

    outputs.append(tritongrpcclient.InferRequestedOutput('detection_coords'))
    outputs.append(tritongrpcclient.InferRequestedOutput('detection_score'))

    for i in tqdm(range(total_requests)):
        inputs[0].set_data_from_numpy(input_img)
        results = triton_client.infer(model_name=model_name,
                                      inputs=inputs,
                                      outputs=outputs)

        bboxes = results.as_numpy('detection_coords')
        scores = results.as_numpy('detection_score')

        detections = tensors_to_detections(bboxes,scores,anchors)
        filtered_detections = filter_detections(detections)
        copyed = input_img_orig.copy()
        draw_detection_single(copyed,filtered_detections,shifts,orig_size,"results/results_{}.jpg".format(i))

    from datetime import datetime
    print("Finished execution at ", datetime.now().time())

