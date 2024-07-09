import io
import re
from pdf2image import convert_from_bytes
from PyPDF2 import PdfReader
from vertexai.generative_models import Image, GenerationConfig, HarmCategory, HarmBlockThreshold

from models import multimodal_model

safety_config = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

config = GenerationConfig(temperature=0.0, top_p=1, top_k=32)

def new_pipeline(list_bytes, custome_field_headers=None, custome_field_items=None):
    output_receipt = {'document_type':'receipt',
                     'cash_amount': '<amount that user should pay if any>',
                     'change_amount': '<change amount>',
                     'currency': '<currency>',
                     'line_item': [{'amount': '<total amount for the product>',
                                   'description': '<product name or description>',
                                   'price_per_unit': '<price per unit item>',
                                   'quantity': '<quantity of the item>'}],
                     'subtotal_amount': '<net amount or subtotal>',
                     'payment_type': '<payment type>',
                     'purchase_time': '<purchase time>',
                     'receipt_date': '<receipt date in YYYY-MM-DD format>',
                     'rounding_amount': '<rounding amount>',
                     'service_charge_amount': '<service charge amount>',
                     'supplier_address': '<supplier address>',
                     'supplier_name': '<supplier name>',
                     'supplier_phone': '<supplier phone>',
                     'total_amount': '<total amount>',
                     'tax_amount': '<total tax amount>'}

    output_invoice = {'document_type':'invoice',
                      'customer_address':'<customer_address>',
                        'customer_email':'<customer_email>',
                        'customer_name':'<customer_name>',
                        'customer_phone':'<customer_phone>',
                        'delivery_fee_amount':'<delivery_fee_amount>',
                         "fully_paid_amount": "<Total amount paid by the user, fully settling the invoice. This may be referred to as 'Lunas' or other similar terms indicating complete payment.>",
                      'down_payment_amount':"<down_payment_amount>",
                        'discount_amount':'<discount_amount>',
                        'grand_total':'<grand_total>',
                        'invoice_date':'<invoice_date YYYY-MM-DD format>',
                        'invoice_due_date':'<invoice_due_date YYYY-MM-DD format>',
                        'invoice_number':'<invoice_number>',
                        'items': [{ 'item_code':'<item_code>',
                                    'item_product_name':'<product name>',
                                    'item_description':'<description product>',
                                    'item_discount':'<item_discount>',
                                    'item_price_unit':'<item_price_unit>',
                                    'item_quantity':'<item_quantity>',
                                    'item_tax':'<item_tax>',
                                    'item_total_amount':'<item_total_amount>',
                                    'item_unit':'<item_unit>' }],
                        'purchase_order_number':'<purchase_order_number>',
                        'subtotal_amount':'<subtotal_amount>',
                        'supplier_account_name':'<supplier_account_name>',
                        'supplier_address':'<supplier_address>',
                        'supplier_bank':'<supplier_bank>',
                        'supplier_bank_account':'<supplier_bank_account>',
                        'supplier_email':'<supplier_email>',
                        'supplier_name':'<supplier_name>',
                        'supplier_npwp':'<supplier_npwp>',
                        'supplier_phone':'<supplier_phone>',
                        'tax_amount':'<tax_amount>',
                        'tax_inclusive_amount':'<tax_inclusive_amount>',
                        'total_amount':'<total_amount>'}  

    for field in custome_field_headers:
        output_receipt.update({field: f"<{field}>"})
        output_invoice.update({field: f"<{field}>"})
        
    for field in custome_field_items:
        output_receipt['line_item'][0].update({field: f"<{field}>"})
        output_invoice['items'][0].update({field: f"<{field}>"})            

    contents = [Image.from_bytes(content) for content in list_bytes]

    prompt = f"""
    Please analyze the following image above using OCR and classify the document_type of that image. 
    Based on your analysis, return the appropriate result:

    If the document_type is "receipt":
    Return a JSON response:
    {output_receipt}

    Else if the document_type is "invoice":
    Return a JSON response:
    {output_invoice}

    Else if not "receipt" and "invoice":
    Return a JSON response:
    {{
      "document_type": "<document_type>",
      "confidence_score": <value between 0 and 1>,
      "explanation": "reason"
    }}
    """

    contents.append(prompt)

    responses = multimodal_model.generate_content(contents,
                                                  safety_settings=safety_config, 
                                                  generation_config=config, 
                                                  stream=True)
    
    full_result = ''
    for response in responses:
        full_result += response.text
    
    usage = {line.split(':')[0].strip(): int(line.split(':')[1].strip()) for line in str(response.usage_metadata).split('\n') if line.strip()}

    null = '-'
    
    try:
        result = eval(re.findall(r'\{.*\}', full_result, flags=re.I|re.S)[0])
    except Exception as e:
        print(e)
        try:
            result = eval(re.findall(r'\[\{.*\}\]', full_result, flags=re.I|re.S)[0])
        except:
            return full_result
    result.update({'usage': usage})
    return result

def split_pdf_per_15_pages_to_images(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    num_files = (total_pages + 14) // 15
    
    all_images_bytes = []

    for i in range(num_files):
        start_page = i * 15
        end_page = min(start_page + 15, total_pages)
        
        images = convert_from_bytes(pdf_bytes, first_page=start_page + 1, last_page=end_page)
        
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            all_images_bytes.append(img_bytes)
        
        print(f"Part {i + 1} processed with pages from {start_page + 1} to {end_page}.")
    
    return all_images_bytes
