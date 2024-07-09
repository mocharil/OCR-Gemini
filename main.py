from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from mimetypes import guess_type
from receipt_detector import receipt_detector
from utils import new_pipeline, split_pdf_per_15_pages_to_images

app = FastAPI()
print('V<<<<<<<<<<<<<<<')
@app.post("/process/")    
def main_process(   content: UploadFile = File(...),
                    custome_field_headers: list[str] = Form([]),
                    custome_field_items: list[str] = Form([])):
    
    
    custome_field_headers_list = [j for i in custome_field_headers if i for j in i.split(',')]
    custome_field_items_list = [j for i in custome_field_items if i for j in i.split(',')]    
    
    content_file = content.file.read()
    mime_type, _ = guess_type(content.filename)
    print(mime_type)

# try:
    if mime_type == 'application/pdf':

        list_bytes = split_pdf_per_15_pages_to_images(content_file)
        result = new_pipeline(list_bytes, custome_field_headers=custome_field_headers_list, custome_field_items=custome_field_items_list)

    else:
        yolo_model_path = 'models/receipt_detector.pt'
        list_bytes = receipt_detector(content_file, yolo_model_path)
        print(f'detected {len(list_bytes)} receipt')

        result = []
        for LB in list_bytes: 
            dt = new_pipeline([LB], custome_field_headers=custome_field_headers_list, custome_field_items=custome_field_items_list)

            result.append(dt)
    return result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
