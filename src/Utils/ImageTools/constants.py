from os import environ

backend = environ.get('IMAGE_BACKEND', 'cv2').lower()
