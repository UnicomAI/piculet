from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang="ch") 
result = ocr.ocr('testocr_2.jpg', cls=False)
print(result)

result = ocr.ocr('testocr_2.jpg', cls=False)
print(result)

result = ocr.ocr('testocr_2.jpg', cls=False)
print(result)