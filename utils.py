import numpy as np
import cv2

configs = {
    "score_clipping_thresh": 100.0,
    "x_scale": 128.0,
    "y_scale": 128.0,
    "h_scale": 128.0,
    "w_scale": 128.0,
    "min_score_thresh": 0.75,
    "min_suppression_threshold": 0.3
}

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_and_crop_image(image, dim):
    if image.shape[0] > image.shape[1]:
        img = image_resize(image, width=dim)
        yshift, xshift = (image.shape[0] - image.shape[1]) // 2, 0
        y_start = (img.shape[0] - img.shape[1]) // 2
        y_end = y_start + dim
        input_img = img[y_start:y_end, :, :]
        input_img = input_img.astype(np.float32)
        input_img = input_img[np.newaxis,...]
        input_img = np.transpose(input_img,(0,3,1,2))
        return input_img, (xshift, yshift)
    else:
        img = image_resize(image, height=dim)
        yshift, xshift = 0, (image.shape[1] - image.shape[0]) // 2
        x_start = (img.shape[1] - img.shape[0]) // 2
        x_end = x_start + dim
        input_img = img[:, x_start:x_end, :]
        input_img = input_img.astype(np.float32)
        input_img = input_img[np.newaxis,...]
        input_img = np.transpose(input_img,(0,3,1,2))
        return input_img, (xshift, yshift)



def tensors_to_detections(raw_boxes, raw_scores, anchors):
    """The output of the neural network is a tensor of shape (b, 896, 16)
    containing the bounding box regressor predictions, as well as a tensor
    of shape (b, 896, 1) with the classification confidences.

    This function converts these two "raw" tensors into proper detections.
    Returns a list of (num_detections, 17) arrays, one for each image in
    the batch.
    """


    detection_boxes = decode_boxes(raw_boxes, anchors)

    thresh = configs['score_clipping_thresh']
    raw_scores = raw_scores.clip(-thresh, thresh)
    detection_scores = sigmoid(raw_scores)
    detection_scores = detection_scores.squeeze(axis=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    mask = detection_scores >= configs['min_score_thresh']

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    output_detections = []

    for i in range(raw_boxes.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]]
        scores = np.expand_dims(scores,-1)
        #output_detections.append(torch.cat((boxes, scores), dim=-1).to('cpu'))
        output_detections.append(np.concatenate((boxes, scores), axis=-1))

    return output_detections
    
def decode_boxes(raw_boxes, anchors):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    x_scale = configs['x_scale']
    y_scale = configs['y_scale']
    w_scale = configs['w_scale']
    h_scale = configs['h_scale']


    boxes = np.zeros_like(raw_boxes)

    x_center = raw_boxes[..., 0] / x_scale* \
        anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / y_scale * \
        anchors[:, 3] + anchors[:, 1]
    w = raw_boxes[..., 2] / w_scale * anchors[:, 2]
    h = raw_boxes[..., 3] / h_scale * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(6):
        offset = 4 + k * 2
        keypoint_x = raw_boxes[..., offset] / \
            x_scale * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / \
            y_scale * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes


def filter_detections(detections):
    """filter available detections using NMS and output of a list
    of detection result tensors, one for each image
    """
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i])
        faces = np.stack(faces) if len(
            faces) > 0 else np.zeros((0,17))
        filtered_detections.append(faces)
    return filtered_detections


def weighted_non_max_suppression(detections):
    """The alternative NMS method as mentioned in the BlazeFace paper:

    "We replace the suppression algorithm with a blending strategy that
    estimates the regression parameters of a bounding box as a weighted
    mean between the overlapping predictions."

    The original MediaPipe code assigns the score of the most confident
    detection to the weighted detection, but we take the average score
    of the overlapping detections.

    The input detections should be a Ndarray of shape (count, 17).

    Returns a list of numpy arrays, one for each detected face.

    """
    if len(detections) == 0:
        return []

    output_detections = []

    # Sort the detections from highest to lowest score.
    remaining = np.argsort(detections[:, 16])

    while len(remaining) > 0:
        detection = detections[remaining[-1]]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)
        first_box = detection[:4]
        other_boxes = detections[remaining, :4]
        ious = overlap_similarity(first_box, other_boxes)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        mask = ious > configs['min_suppression_threshold']
        overlapping = remaining[mask]
        remaining = remaining[~mask]

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = np.copy(detection)
        if len(overlapping) > 1:
            coordinates = detections[overlapping, :16]
            scores = detections[overlapping, 16:17]
            total_score = scores.sum()
            weighted = (coordinates * scores).sum(axis=0) / total_score
            weighted_detection[:16] = weighted
            weighted_detection[16] = total_score / len(overlapping)

        output_detections.append(weighted_detection)

    return output_detections


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2]:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
        box_a: (Ndarray) bounding boxes, Shape: [A,4].
        box_b: (Ndarray) bounding boxes, Shape: [B,4].
    Return:
        (Ndarray) intersection area, Shape: [A,B].
    """
    A = box_a.shape[0]
    B = box_b.shape[0]

    max_box_a = box_a[:,2:]
    max_box_a = np.expand_dims(max_box_a,axis=1)
    max_box_a = np.broadcast_to(max_box_a,(A,B,2))

    min_box_a = box_a[:,:2]
    min_box_a = np.expand_dims(min_box_a,axis=1)
    min_box_a = np.broadcast_to(min_box_a,(A,B,2))

    max_box_b = box_b[:,2:]
    max_box_b = np.expand_dims(max_box_b,axis=0)
    max_box_b = np.broadcast_to(max_box_b,(A,B,2))

    min_box_b = box_b[:,:2]
    min_box_b = np.expand_dims(min_box_b,axis=0)
    min_box_b = np.broadcast_to(min_box_b,(A,B,2))
    

    max_xy = np.minimum(max_box_a,max_box_b)
  
    
    min_xy = np.maximum(min_box_a,min_box_b)

    inter = np.clip((max_xy - min_xy),0,None)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (Ndarray) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (Ndarray) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (Ndarray) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
   
    area_a = ((box_a[:, 2] - box_a[:, 0])
                * (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    
    area_a = np.expand_dims(area_a,axis=1)
    area_a = np.broadcast_to(area_a,inter.shape)

    area_b = ((box_b[:, 2] - box_b[:, 0])
                * (box_b[:, 3] - box_b[:, 1])) # [A,B]
    area_b = np.expand_dims(area_b,axis=0)
    area_b = np.broadcast_to(area_b,inter.shape)
    
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def overlap_similarity(box, other_boxes):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box[np.newaxis,...], other_boxes).squeeze(axis=0)


def draw_detection_single(img, preds, shifts, orig_size, save_path):
    """Draw all detection results on the target image
    """
    shift = np.array(shifts * 2)
    
    for pred in preds:
        for face in pred:
            xmin = face[1] * orig_size + shift[0]
            ymin = face[0] * orig_size + shift[1]
            xmax = face[3] * orig_size + shift[0]
            ymax = face[2] * orig_size + shift[1]
            cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,255,0), 2)
            for key in range(4,16,2):
                kp = (int(face[key] * orig_size + shift[0]), int(face[key+1] * orig_size + shift[1]))
                cv2.circle(img,kp,2,(0,255,0),2)
    img = img[...,::-1]
    cv2.imwrite(save_path,img)






