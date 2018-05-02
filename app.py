# app.py 파일
# 웹 프론트엔드와 각 모듈을 중계해준다.

from flask import Flask, render_template, request
import cv2
import numpy as np
from daemon import client

app = Flask(__name__)

# 파일 사이즈를
# Human readable format으로 변경
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# 프론트엔드 index
@app.route('/')
def view_index():
    return render_template('index.html')

# Upload 엔트리
# 이미지 파일을 받아 numpy array로 디코드
# 사이즈와 이름 등 기본 정보를 분석 후
# analysis()를 호출
@app.route('/upload', methods=['POST'])
def view_upload():
    f = request.files['image']
    blob = f.read()
    size = len(blob)
    blob_array = np.asarray(bytearray(blob), dtype=np.uint8)

    #img = cv2.imdecode(blob_array, cv2.IMREAD_COLOR) # 이미지를 디코드 후 numpy array로 변환
    analyzed = client.send(bytearray(blob))

    ret = ( "name : %s\n" % f.filename +
            "size : %s\n" % sizeof_fmt(size) +
            #dimension : %d X %d\n" % (img.shape[1], img.shape[0]) +
            "\n<Analysis>\n%s" % analyzed)
    return ret

# 웹서버를 통하지 않고 python 인터프리터로 바로 실행되었을 때의 상황.
if __name__ == '__main__':
    app.run("0.0.0.0", debug=True)
