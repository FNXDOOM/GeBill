# Bill Processing API with Gen AI

A FastAPI-based application that processes bills and invoices using Google's Gemini AI for data extraction. The application can handle various file formats, extract key information, and log the results in an Excel file. Supports concurrent processing of multiple bills while maintaining data integrity.

## Features

- **Multiple File Upload Support**: Process multiple bills simultaneously
- **Concurrent Processing**: Efficient parallel processing of multiple files
- **Data Integrity**: Thread-safe Excel operations to prevent data corruption
- **File Upload Support**: Accepts PNG, JPG, JPEG, WEBP, and PDF files
- **AI-Powered Extraction**: Uses Google's Gemini AI to extract:
  - Vendor Name
  - Invoice Number
  - Invoice Date
  - Total Amount
- **Excel Logging**: Automatically logs extracted data to an Excel file
- **Secure File Handling**: Implements secure filename handling and proper file storage
- **CORS Support**: Configured for cross-origin requests
- **Error Handling**: Comprehensive error handling and validation

## Prerequisites

- Python 3.7+
- Google Gemini API key
- Required Python packages (see Installation)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install fastapi uvicorn python-dotenv google-generativeai openpyxl pillow werkzeug
```

4. Create a `.env` file in the root directory with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Project Structure

```
.
├── app.py              # Main FastAPI application
├── .env               # Environment variables
├── uploads/           # Directory for uploaded files
├── processed/         # Directory for processed files
├── processed_bills_log.xlsx  # Excel log file
└── index.html         # Frontend interface
```

## Usage

1. Start the server:
```bash
uvicorn app:app --reload
```

2. The API will be available at `http://localhost:8000`

### API Endpoints

- `POST /upload-bill/`: Upload and process multiple bill files
  - Accepts multipart form data with:
    - `files`: List of bill files to process (multiple files supported)
    - `company_name` (optional): Company name
    - `notes` (optional): Additional notes

- `GET /`: Root endpoint for API status check

### Example Response for Multiple Files

```json
{
    "total_files": 3,
    "processed_files": 2,
    "results": [
        {
            "upload_status": "success",
            "message": "Bill upload processed.",
            "file_id": "unique_id_1",
            "filename": "processed_filename_1",
            "original_filename": "original_filename_1",
            "content_type": "file_content_type",
            "company_name": "optional_company_name",
            "notes": "optional_notes",
            "processing_status": "success",
            "processing_message": null,
            "extracted_data": {
                "vendor_name": "Vendor Name 1",
                "invoice_number": "INV-123",
                "invoice_date": "2024-04-11",
                "total_amount": 123.45
            },
            "excel_write_status": "success",
            "excel_write_message": "Data successfully appended to Excel."
        },
        {
            "upload_status": "success",
            "message": "Bill upload processed.",
            "file_id": "unique_id_2",
            "filename": "processed_filename_2",
            "original_filename": "original_filename_2",
            "content_type": "file_content_type",
            "company_name": "optional_company_name",
            "notes": "optional_notes",
            "processing_status": "success",
            "processing_message": null,
            "extracted_data": {
                "vendor_name": "Vendor Name 2",
                "invoice_number": "INV-456",
                "invoice_date": "2024-04-11",
                "total_amount": 678.90
            },
            "excel_write_status": "success",
            "excel_write_message": "Data successfully appended to Excel."
        }
    ]
}
```

## Error Handling

The application includes comprehensive error handling for:
- Invalid file types
- File upload failures
- AI processing errors
- Excel writing issues
- API key configuration problems

## Security Considerations

- Secure filename handling using `secure_filename`
- Environment variable configuration for sensitive data
- Proper file storage and cleanup
- CORS configuration (configure appropriately for production)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For support, please contant us at nikhilkumar66513@gmail.com.
