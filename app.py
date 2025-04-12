from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Integer, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import uuid
import json
import google.generativeai as genai
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Configure Google Gemini AI
try:
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Gemini AI configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}")
    gemini_model = None

# Initialize FastAPI
app = FastAPI(
    title="Bill Processing API",
    description="API for processing bills using Gemini AI with SQLite storage and templates",
    version="1.0.2"
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define directories & settings
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
DB_URL = os.getenv("DB_URL", "sqlite:///bill_logs.db")

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"Upload directory created: {UPLOAD_DIR}")

# SQLite setup
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

class BillLog(Base):
    __tablename__ = "bill_logs"
    id = Column(Integer, primary_key=True)
    processed_timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    status = Column(String)
    original_filename = Column(String)
    saved_filename = Column(String)
    file_id = Column(String)
    message = Column(String, nullable=True)
    error_type = Column(String, nullable=True)
    extracted_data = Column(JSON, nullable=True)
    template_id = Column(Integer, nullable=True)  # Link to template used

class Template(Base):
    __tablename__ = "templates"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)  # e.g., "Water Bill", "Electricity Bill"
    fields = Column(JSON)  # List of field names, e.g., ["vendor_name", "bill_date", "amount"]

# Database migration logic
def migrate_database():
    """Check and migrate the database schema if needed."""
    inspector = inspect(engine)
    if inspector.has_table("bill_logs"):
        # Check if template_id column exists
        columns = [col["name"] for col in inspector.get_columns("bill_logs")]
        if "template_id" not in columns:
            logger.info("bill_logs table exists without template_id. Dropping and recreating...")
            Base.metadata.drop_all(engine, tables=[BillLog.__table__])
            Base.metadata.create_all(engine)
            logger.info("bill_logs table recreated with template_id column.")
    else:
        # If table doesn't exist, create all tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created.")

# Run migration on startup
migrate_database()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("SQLite database initialized")

def allowed_file(filename: str) -> bool:
    """Check if the filename has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def extract_bill_data_with_gemini(file_content: bytes, mime_type: str, fields: List[str]) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract bill data using Gemini AI with retries, based on specified fields."""
    if not gemini_model:
        return None, "Gemini AI model not configured"

    file_part = {"mime_type": mime_type, "data": file_content}
    fields_str = "\n- " + "\n- ".join(fields)
    prompt = f"""
    You are an expert data extraction assistant specialized in invoices and bills.
    Analyze the provided document (image or PDF).
    Extract the following fields:
    {fields_str}

    Output *strictly* in JSON format:
    {{
      {", ".join(f'"{field}": "..."' for field in fields)}
    }}
    Use null for unextractable fields.
    """
    try:
        response = await gemini_model.generate_content_async([prompt, file_part])
        json_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        if not json_text:
            return None, "Empty AI response"
        data = json.loads(json_text)
        required_keys = set(fields)
        if not required_keys.issubset(data.keys()):
            return {k: data.get(k, None) for k in required_keys}, "Incomplete AI response"
        return data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON from AI: {str(e)}"
    except Exception as e:
        return None, f"AI processing error: {str(e)}"

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bill Processing API"}

@app.post("/templates/")
async def create_template(name: str = Form(...), fields: str = Form(...)):
    """Create a new template for bill data extraction."""
    try:
        db = SessionLocal()
        # Parse fields from a comma-separated string (e.g., "vendor_name, bill_date, amount")
        field_list = [field.strip() for field in fields.split(",") if field.strip()]
        if not field_list:
            raise HTTPException(status_code=400, detail="At least one field is required")
        
        # Check if template name already exists
        if db.query(Template).filter(Template.name == name).first():
            raise HTTPException(status_code=400, detail="Template name already exists")
        
        template = Template(name=name, fields=field_list)
        db.add(template)
        db.commit()
        db.refresh(template)
        db.close()
        return {"id": template.id, "name": template.name, "fields": template.fields}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create template")

@app.get("/templates/")
async def get_templates():
    """Retrieve all available templates."""
    try:
        db = SessionLocal()
        templates = db.query(Template).all()
        result = [{"id": t.id, "name": t.name, "fields": t.fields} for t in templates]
        db.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching templates: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch templates")

@app.post("/upload-bill/")
@limiter.limit("10/minute")
async def upload_bill(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    company_name: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    template_id: Optional[int] = Form(None)  # Optional template selection
):
    """Upload and process multiple bill files with an optional template."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Default fields if no template is specified
        default_fields = ["vendor_name", "invoice_number", "invoice_date", "total_amount"]
        fields = default_fields

        # Load template fields if template_id is provided
        if template_id:
            db = SessionLocal()
            template = db.query(Template).filter(Template.id == template_id).first()
            db.close()
            if not template:
                raise HTTPException(status_code=400, detail=f"Template with ID {template_id} not found")
            fields = template.fields

        results = []
        processed_timestamp = datetime.now(timezone.utc)
        db = SessionLocal()

        for file in files:
            try:
                if not file.filename:
                    results.append({"status": "error", "message": "No filename", "filename": "unknown"})
                    continue

                if not allowed_file(file.filename):
                    results.append({
                        "status": "error",
                        "message": f"Invalid file type: {file.filename}",
                        "filename": file.filename
                    })
                    continue

                # Check file size
                file_content = await file.read()
                if len(file_content) > MAX_FILE_SIZE:
                    results.append({
                        "status": "error",
                        "message": f"File {file.filename} exceeds 10MB",
                        "filename": file.filename
                    })
                    continue

                filename_secure = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_id = str(uuid.uuid4())[:8]
                extension = os.path.splitext(filename_secure)[1]
                saved_filename = f"{timestamp}_{file_id}{extension}"
                file_path = os.path.join(UPLOAD_DIR, saved_filename)

                # Save file
                with open(file_path, "wb") as buffer:
                    buffer.write(file_content)

                # Process with Gemini AI using specified fields
                extracted_data, processing_error = await extract_bill_data_with_gemini(
                    file_content=file_content,
                    mime_type=file.content_type,
                    fields=fields
                )
                status = "success" if not processing_error else "error"
                message = processing_error or "Successfully processed"

                # Log to SQLite
                log_entry = BillLog(
                    processed_timestamp=processed_timestamp,
                    status=status.upper(),
                    original_filename=file.filename,
                    saved_filename=saved_filename,
                    file_id=file_id,
                    message=message,
                    error_type="AIProcessingError" if processing_error else None,
                    extracted_data=extracted_data,
                    template_id=template_id
                )
                db.add(log_entry)

                results.append({
                    "status": status,
                    "message": message,
                    "file_id": file_id,
                    "original_filename": file.filename,
                    "saved_filename": saved_filename,
                    "extracted_data": extracted_data,
                    "template_id": template_id
                })

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                results.append({
                    "status": "error",
                    "message": f"Unexpected error: {str(e)}",
                    "filename": file.filename,
                    "error_type": type(e).__name__
                })
            finally:
                await file.close()

        db.commit()
        db.close()

        successful_uploads = [r for r in results if r["status"] == "success"]
        if not successful_uploads:
            raise HTTPException(status_code=400, detail={
                "message": "No files processed successfully",
                "errors": results
            })

        # Clean up old files in background
        background_tasks.add_task(cleanup_old_files, UPLOAD_DIR)

        return {
            "total_files": len(files),
            "processed_files": len(successful_uploads),
            "failed_files": len(results) - len(successful_uploads),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail={"message": "Internal error"})

@app.get("/history/")
async def get_history():
    """Retrieve bill processing history."""
    try:
        db = SessionLocal()
        logs = db.query(BillLog).order_by(BillLog.processed_timestamp.desc()).limit(100).all()
        result = [{
            "processed_timestamp": log.processed_timestamp.isoformat(),
            "status": log.status,
            "original_filename": log.original_filename,
            "extracted_data": log.extracted_data,
            "template_id": log.template_id
        } for log in logs]
        db.close()
        return result
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@app.get("/health")
async def health_check():
    """Check API health."""
    return {
        "status": "healthy",
        "gemini_configured": bool(gemini_model),
        "db_connected": engine.dialect.has_table(engine, "bill_logs")
    }

def cleanup_old_files(directory: str, max_age_days: int = 30):
    """Remove files older than max_age_days."""
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.getmtime(file_path) < time.time() - max_age_days * 86400:
                os.remove(file_path)
                logger.info(f"Deleted old file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )