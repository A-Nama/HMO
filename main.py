import os
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import io
from urllib.parse import quote
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import uuid

# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck PDF Generator & Viewer API",
    description="An API to generate, save, and view pitch decks.",
    version="4.0.0"
)

# --- CORS Configuration ---
origins = ["http://localhost", "http://localhost:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Client Initialization ---
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    print("FATAL: Supabase credentials not found. Application cannot function without them.")

# --- Pydantic Models ---
class PitchDeckSlide(BaseModel):
    title: str
    content: str
    
class PitchDeckContent(BaseModel):
    company_name: str
    tagline: str
    slides: List[PitchDeckSlide]

# --- AI & PDF Helper Functions (These remain the same) ---

def create_pitch_deck_prompt(idea_text: str) -> str:
    return f"""
    You are an expert startup consultant. Based on the user's idea: "{idea_text}", generate the content for a 10-slide pitch deck.
    Return a single, valid JSON object with the structure: {{"company_name": "...", "tagline": "...", "slides": [{{"title": "...", "content": "..."}}]}}.
    The 10 slides MUST be: 1. The Problem, 2. Our Solution, 3. Product/Service, 4. Market Size, 5. Business Model, 6. Go-to-Market Strategy, 7. Competitive Landscape, 8. The Team, 9. Financial Projections, 10. The Ask.
    """

async def _generate_deck_content(idea_text: str) -> dict:
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = create_pitch_deck_prompt(idea_text)
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text content: {e}")

async def _generate_image_for_slide(slide_content: str, client: httpx.AsyncClient) -> bytes:
    if not HUGGING_FACE_API_KEY:
        raise ValueError("HUGGING_FACE_API_KEY not found. Please add it to your .env file.")
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    image_prompt = f"A professional, minimalist, abstract vector art representing the concept of: {slide_content}"
    try:
        response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        if response.status_code == 503:
            await asyncio.sleep(15)
            response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"Hugging Face image generation failed: {e}")
        fallback_url = f"https://placehold.co/800x400/EEE/31343C?text=AI+Image+Failed"
        response = await client.get(fallback_url)
        return response.content

def _create_pitch_deck_pdf(deck_content: dict, images: list) -> io.BytesIO:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    styles = getSampleStyleSheet()
    c.setFont("Helvetica-Bold", 36)
    c.drawCentredString(width / 2.0, height - 150, deck_content['company_name'])
    c.setFont("Helvetica", 18)
    c.drawCentredString(width / 2.0, height - 200, deck_content['tagline'])
    c.showPage()
    for i, slide in enumerate(deck_content['slides']):
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 100, slide['title'])
        p = Paragraph(slide['content'], styles['Normal'])
        p.wrapOn(c, width - 100, height - 450)
        p.drawOn(c, 50, height - 350)
        if i < len(images):
            image_reader = ImageReader(io.BytesIO(images[i]))
            c.drawImage(image_reader, 50, 80, width=width-100, height=200, preserveAspectRatio=True, anchor='n')
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# --- API Endpoints ---

@app.post("/decks", tags=["Pitch Decks"])
async def generate_and_save_deck(audio_file: UploadFile = File(...)):
    """
    Generates a new pitch deck from audio, saves it to the database and
    storage, and returns its metadata.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        # Steps 1-4: Transcribe, Generate Content, Generate Images, Create PDF
        transcription_model = genai.GenerativeModel('gemini-1.5-flash')
        audio_bytes = await audio_file.read()
        uploaded_audio = genai.upload_file(path="temp_audio", content=audio_bytes)
        response = await transcription_model.generate_content_async(["Transcribe this startup idea.", uploaded_audio])
        transcribed_idea = response.text.strip()

        deck_content = await _generate_deck_content(transcribed_idea)

        async with httpx.AsyncClient() as client:
            image_tasks = [_generate_image_for_slide(slide['content'], client) for slide in deck_content['slides']]
            images = await asyncio.gather(*image_tasks)

        pdf_buffer = _create_pitch_deck_pdf(deck_content, images)
        pdf_bytes = pdf_buffer.getvalue()
        
        # Step 5: Upload PDF to Supabase Storage
        file_path = f"public/{uuid.uuid4()}.pdf"
        # The bucket name is 'pitch_decks'
        supabase.storage.from_("pitch_decks").upload(file=pdf_bytes, path=file_path, file_options={"content-type": "application/pdf"})

        # Step 6: Get Public URL for the PDF
        res = supabase.storage.from_("pitch_decks").get_public_url(file_path)
        pdf_url = res

        # Step 7: Save Metadata to Supabase Database
        deck_to_save = {
            "company_name": deck_content.get("company_name"),
            "tagline": deck_content.get("tagline"),
            "original_idea": transcribed_idea,
            "pdf_url": pdf_url,
            "storage_path": file_path
        }
        db_response = supabase.table("pitch_decks").insert(deck_to_save).execute()
        
        # Step 8: Return the newly created record
        new_deck = db_response.data[0]
        return JSONResponse(status_code=201, content=new_deck)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decks", tags=["Pitch Decks"])
async def get_all_decks():
    """Retrieves a list of all saved pitch decks."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    try:
        response = supabase.table("pitch_decks").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/decks/{deck_id}", tags=["Pitch Decks"])
async def delete_deck(deck_id: int):
    """Deletes a pitch deck from the database and its file from storage."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available.")
    try:
        # First, find the record to get the storage path
        select_res = supabase.table("pitch_decks").select("storage_path").eq("id", deck_id).execute()
        if not select_res.data:
            raise HTTPException(status_code=404, detail="Deck not found.")
        
        storage_path = select_res.data[0]['storage_path']

        # Delete from storage
        supabase.storage.from_("pitch_decks").remove([storage_path])

        # Delete from database
        supabase.table("pitch_decks").delete().eq("id", deck_id).execute()
        
        return {"status": "success", "message": f"Deck {deck_id} deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
