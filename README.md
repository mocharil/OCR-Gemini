# OCR Invoice and Receipt using Gemini

This project demonstrates how to process receipts and invoices using FastAPI and the YOLO model for receipt detection. It also includes functionalities to process PDF files and convert them into images for further processing.

## Features

- **Receipt Detection:**
  - Uses YOLO model to detect receipts in images.
  - Processes PDF files and converts them to images for receipt detection.
- **Invoice and Receipt Processing:**
  - Analyzes images of invoices and receipts.
  - Uses Google's Vertex AI for document analysis and OCR.
  - Provides detailed results including document type and content analysis.

## Endpoints

### POST /process

#### Request

Form data parameters:

- `file`: The file to be processed (PDF or image).
- `custome_field_headers`: Custom field headers, separated by commas.
- `custome_field_items`: Custom field items, separated by commas.

#### Response

The response is a JSON object containing the results of the receipt or invoice processing.

```json
{
  "document_type": "receipt",
  "total_amount": "<total amount>",
  "line_items": [
    {
      "description": "<item description>",
      "amount": "<item amount>"
    }
  ],
  "usage": {
    "prompt_token_count": 100,
    "candidates_token_count": 200,
    "total_token_count": 300
  }
}
```

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory with the following variables:

    ```env
    PROJECT_ID=your_project_id
    CREDENTIALS_FILE_PATH=/path/to/your/credentials.json
    GEMINI_MODEL=gemini-model-name
    YOLO_MODEL_PATH=models/receipt_detector.pt
    ```

5. Run the FastAPI application:

    ```sh
    uvicorn main:app --reload
    ```

## Project Structure

- `main.py`: Main FastAPI application.
- `config.py`: Configuration settings loaded from the `.env` file.
- `models.py`: Initialization and setup for AI models.
- `utils.py`: Utility functions for PDF and image processing.
- `receipt_detector.py`: YOLO model for receipt detection.

## Usage

1. Start the FastAPI server.
2. Send a POST request to `/process` with the necessary form data.
3. Receive the receipt or invoice processing results in the response.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
