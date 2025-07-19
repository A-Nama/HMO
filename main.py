import os
import json
import asyncio
import httpx
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --- Environment Variable Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Pitch Deck PDF Generator & Viewer API (Public)",
    description="A public API to generate, save, and view pitch decks from audio or text.",
    version="6.0.0" # Version for public/no-auth
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
    raise ValueError("FATAL: Supabase credentials not found. Application cannot function.")

# --- Pydantic Models & Helper Functions ---
class PitchDeckSlide(BaseModel):
    title: str
    content: str
class PitchDeckContent(BaseModel):
    company_name: str
    tagline: str
    slides: List[PitchDeckSlide]

def create_pitch_deck_prompt(idea_text: str) -> str:
    return f"""
    You are an expert startup consultant. Based on the user's idea: "{idea_text}", generate the content for a 10-slide pitch deck.
    Return a single, valid JSON object with the structure: {{"company_name": "...", "tagline": "...", "slides": [{{"title": "...", "content": "..."}}]}}.
    The 10 slides MUST be: 1. The Problem, 2. Our Solution, 3. Product/Service, 4. Market Size, 5. Business Model, 6. Go-to-Market Strategy, 7. Competitive Landscape, 8. Traction & Milestones, 9. Financial Projections, 10. The Ask.
    For the 'Traction & Milestones' slide, invent some plausible early achievements and future goals.
    """
async def _generate_deck_content(idea_text: str) -> dict:
    model = genai.GenerativeModel('gemini-pro')
    prompt = create_pitch_deck_prompt(idea_text)
    response = await model.generate_content_async(prompt)
    return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())

async def _generate_image_for_slide(slide_content: str, client: httpx.AsyncClient) -> bytes:
    if not HUGGING_FACE_API_KEY: 
        raise ValueError("HUGGING_FACE_API_KEY not found.")
        
    api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1-base"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    image_prompt = f"A professional, minimalist, abstract vector art representing: {slide_content}"
    
    async def get_fallback_image():
        fallback_url = "https://placehold.co/800x400/EEE/31343C?text=AI+Image+Failed"
        return (await client.get(fallback_url)).content

    try:
        response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        if response.status_code == 503:
            await asyncio.sleep(15)
            response = await client.post(api_url, headers=headers, json={"inputs": image_prompt}, timeout=120.0)
        
        if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
            return response.content
        else:
            print(f"Hugging Face returned non-image data. Status: {response.status_code}. Body: {response.text}")
            return await get_fallback_image()
    except Exception as e:
        print(f"Hugging Face image generation failed: {e}")
        return await get_fallback_image()

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

# --- API Endpoints (No Authentication) ---

@app.post("/decks", tags=["Pitch Decks"])
async def generate_and_save_deck(
    audio_file: Optional[UploadFile] = File(None),
    idea_text: Optional[str] = Form(None)
):
    """
    Generates a new pitch deck from EITHER audio OR text and saves it.
    This endpoint is public.
    """
    if not audio_file and not idea_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or typed text.")
    if audio_file and idea_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or typed text, not both.")

    try:
        final_idea_text = ""
        if audio_file:
            if not audio_file.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid file type for audio.")
            
            transcription_model = genai.GenerativeModel('gemini-1.5-flash')
            audio_bytes = await audio_file.read()
            audio_part = {"mime_type": audio_file.content_type, "data": audio_bytes}
            response = await transcription_model.generate_content_async(["Transcribe this startup idea.", audio_part])
            final_idea_text = response.text.strip()
            
        elif idea_text:
            if len(idea_text) < 20:
                 raise HTTPException(status_code=400, detail="Text idea is too short. Please provide more detail.")
            final_idea_text = idea_text

        deck_content = await _generate_deck_content(final_idea_text)
        async with httpx.AsyncClient() as client:
            image_tasks = [_generate_image_for_slide(slide['content'], client) for slide in deck_content['slides']]
            images = await asyncio.gather(*image_tasks)
        pdf_buffer = _create_pitch_deck_pdf(deck_content, images)
        pdf_bytes = pdf_buffer.getvalue()
        
        # Save to a generic public folder, not a user-specific one
        file_path = f"public/{uuid.uuid4()}.pdf"
        supabase.storage.from_("pitch_decks").upload(file=pdf_bytes, path=file_path, file_options={"content-type": "application/pdf"})
        pdf_url = supabase.storage.from_("pitch_decks").get_public_url(file_path)

        deck_to_save = {
            # "user_id" is no longer needed
            "company_name": deck_content.get("company_name"),
            "tagline": deck_content.get("tagline"),
            "original_idea": final_idea_text,
            "pdf_url": pdf_url,
            "storage_path": file_path
        }
        db_response = supabase.table("pitch_decks").insert(deck_to_save).execute()
        
        return JSONResponse(status_code=201, content=db_response.data[0])

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decks", tags=["Pitch Decks"])
async def get_all_decks():
    """Retrieves a list of ALL saved pitch decks."""
    response = supabase.table("pitch_decks").select("*").order("created_at", desc=True).execute()
    return response.data

@app.delete("/decks/{deck_id}", tags=["Pitch Decks"])
async def delete_deck(deck_id: int):
    """Deletes any pitch deck by its ID."""
    select_res = supabase.table("pitch_decks").select("storage_path").eq("id", deck_id).single().execute()
    if not select_res.data:
        raise HTTPException(status_code=404, detail="Deck not found.")
    
    storage_path = select_res.data['storage_path']
    supabase.storage.from_("pitch_decks").remove([storage_path])
    supabase.table("pitch_decks").delete().eq("id", deck_id).execute()
    
    return {"status": "success", "message": f"Deck {deck_id} deleted."}
