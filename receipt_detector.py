import os
import cv2
import numpy as np
from ultralytics import YOLO
from pdf2image import convert_from_bytes

class ReceiptDetector:
    """
    A class used to detect and crop receipts from images and PDF files using YOLO model.

    Attributes:
    ----------
    yolo_model_path : str
        The path to the YOLO model.
    confidence_threshold : float
        The confidence threshold for YOLO model predictions.
    yolo_model : YOLO
        The loaded YOLO model.

    Methods:
    -------
    image_to_bytes(image)
        Converts a CV2 image to bytes.
    crop_receipts(image)
        Detects and crops receipts from the given image.
    process_pdf(pdf_bytes)
        Converts PDF file bytes to a list of images.
    process_image_bytes(image_bytes)
        Processes image or PDF file bytes and returns a list of images.
    """

    def __init__(self, yolo_model_path, confidence_threshold=0.8):
        """
        Initializes the ReceiptDetector with the given YOLO model path and confidence threshold.

        Parameters:
        ----------
        yolo_model_path : str
            The path to the YOLO model.
        confidence_threshold : float
            The confidence threshold for YOLO model predictions.
        """
        self.yolo_model_path = yolo_model_path
        self.confidence_threshold = confidence_threshold
        
        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path)
    
    def image_to_bytes(self, image):
        """
        Converts a CV2 image to bytes.

        Parameters:
        ----------
        image : ndarray
            The image to convert.

        Returns:
        -------
        bytes
            The image in bytes.
        """
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
    
    def crop_receipts(self, image):
        """
        Detects and crops receipts from the given image.

        Parameters:
        ----------
        image : ndarray
            The image in which to detect receipts.

        Returns:
        -------
        list of bytes
            A list of cropped receipt images in bytes, or None if no receipts are detected.
        """
        results = self.yolo_model(image, conf=self.confidence_threshold)
        
        if results is None or len(results) == 0:
            return None
        
        cropped_images_bytes = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped_image = image[y1:y2, x1:x2]
                cropped_images_bytes.append(self.image_to_bytes(cropped_image))
        
        return cropped_images_bytes

    def process_pdf(self, pdf_bytes):
        """
        Converts PDF file bytes to a list of images.

        Parameters:
        ----------
        pdf_bytes : bytes
            The PDF file in bytes.

        Returns:
        -------
        list of ndarray
            A list of images converted from the PDF.
        """
        images = convert_from_bytes(pdf_bytes)
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

    def process_image_bytes(self, image_bytes):
        """
        Processes image or PDF file bytes and returns a list of images.

        Parameters:
        ----------
        image_bytes : bytes
            The image or PDF file in bytes.

        Returns:
        -------
        list of ndarray
            A list of images.
        """
        if image_bytes[:4] == b'%PDF':
            # Process PDF
            return self.process_pdf(image_bytes)
        else:
            # Process image
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            return [image]

def receipt_detector(file_bytes, yolo_model_path):
    """
    Main function to detect and crop receipts from a given file.

    Parameters:
    ----------
    file_path : bytes
        bytes from file (image or PDF).
    yolo_model_path : str
        The path to the YOLO model.

    Returns:
    -------
    list of bytes
        A list of cropped receipt images in bytes, or the original image bytes if no receipts are detected.
    """
    detector = ReceiptDetector(yolo_model_path=yolo_model_path)
    results = []
    
    images = detector.process_image_bytes(file_bytes)

    for idx, image in enumerate(images):
        # Crop receipts
        cropped_images_bytes = detector.crop_receipts(image)

        if cropped_images_bytes:
            results.extend(cropped_images_bytes)
        else:
            results.append(detector.image_to_bytes(image))

    return results
