import falcon
import mimetypes
import io
import os
import uuid
import cv2
import numpy as np
from sudoku_detector.main import process_sudoku_image
from neural_network.run import classify_numbers
from celery.result import AsyncResult
from falcon_cors import CORS
from utils import RedisService
import base64
import json

service = RedisService()

class SudokuSolveResource(object):
    def on_post(self, req, resp):
        ext = mimetypes.guess_extension(req.content_type)
        nparr = np.frombuffer(req.stream.read(), np.uint8)
        img_np = cv2.imdecode(nparr, 1)
        number_images = process_sudoku_image(img_np)
        serialized_number_images = []
        for number_image in number_images:
            print(type(number_image))
            serialized_number_image = base64.b64encode(number_image.dumps()).decode()
            serialized_number_images.append(serialized_number_image)

        redis_key_scan = f'{uuid.uuid1()}'
        task_scan = classify_numbers.delay(serialized_number_images)
        resp.status = falcon.HTTP_202
        resp.body = json.dumps({
            'id_task_scan': task_scan.id
        })


class SudokuSolveItem(object):
    def on_get(self, req, resp, scan_id):
        scan_result = AsyncResult(scan_id)
        result = {
            'status': scan_result.status,
            'scan_result': scan_result.result
        }
        resp.status = falcon.HTTP_200

        resp.body = json.dumps(result)


images = SudokuSolveResource()
app = falcon.API(middleware=[
    CORS(
        allow_all_origins=True,
        allow_all_methods=True,
        allow_all_headers=True
    ).middleware
])
app.add_route('/sudoku-solve', images)
app.add_route('/sudoku-solve/{scan_id}', SudokuSolveItem())

